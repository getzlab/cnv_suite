from collections import defaultdict
from dataclasses import dataclass
import dataclasses
from pathlib import Path
import random
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import numpy as np
from intervaltree import IntervalTree, Interval
import pandas as pd
from simulate import cnv_profile
from simulate.cnv_profile import (
    MATERNAL,
    PATERNAL,
    CNV_Profile,
    simulate_coverage_and_depth,
)
from utils import PathLike, gen_id, DATA_PATH, OUT_PATH
from utils.simulation_utils import (
    Haplotype,
    dump_tsv,
    read_coverage,
    read_snvs,
    switch_contigs,
)


class W:
    """
    A wrapper for objects stored in IntervalTrees.

    The IntervalTree class stores things internally as sets of intervals, which might cause equal intervals to be dropped by accident.
    To prevent this from happening, wrap all objects you put in the tree with W, and use
    `W.unwrap` or `W.split_merge_unwrap` to extract a reduced tree in the end.
    """

    def __init__(self, val):
        self.val = val

    def __repr__(self) -> str:
        return f"W({self.val})"

    def __str__(self) -> str:
        return f"W({self.val})"

    @staticmethod
    def unwrap(tree: IntervalTree) -> IntervalTree:
        """
        Given a tree whose values are wrapped using W,
        returns a tree whose values are unwrapped.
        """
        res = IntervalTree()
        for inter in tree:
            assert isinstance(inter.data, W)
            res[inter.begin : inter.end] = inter.data.val

        return res

    @staticmethod
    def split_merge_unwrap(
        tree: IntervalTree, merge_func: Callable
    ) -> IntervalTree:
        """
        Given a tree whose values are wrapped using W,
        returns a tree whose values are unwrapped.
        """
        tree.split_overlaps()
        tree.merge_overlaps(lambda x, y: W(merge_func(x.val, y.val)))
        return W.unwrap(tree)


@dataclass(
    frozen=True,
)
class DnaSegment:
    """
    A class storing a DNA segment.
    The DNA segments can have a source from which they were cloned. Mutations occur in DNA segments.
    """

    source: "Union[DnaSegment, Tuple[str, Haplotype]]"
    """
    The DNA segment from which the DNA segment is derived.
    A DNA segment can be either a chromosome, or a subset of another DNA segment.
    """
    start: int
    end: int

    id: int = dataclasses.field(default_factory=gen_id)

    def __post_init__(self):
        assert 0 <= self.start <= self.end
        if isinstance(self.source, Chromosome):
            assert self.end <= len(self.source), (
                "Attempted to create a DNA segment out of bounds!"
            )

    def __len__(self) -> int:
        return self.end - self.start

    def map_to_source(self, idx: int) -> Tuple[int, Tuple[str, Haplotype]]:
        """
        Maps a position in the DNA segment to the chromosome it came from.
        """
        assert 0 <= idx < len(self)
        if isinstance(self.source, DnaSegment):
            return self.source.map_to_source(idx + self.start)
        else:
            return idx, self.source

    def clone(self):
        """Returns a copy of the DNA segment."""
        return DnaSegment(self.source, self.start, self.end)

    def __repr__(self) -> str:
        if isinstance(self.source, DnaSegment):
            return f"DnaSegment(id={self.id}, {self.start}:{self.end}, ...)"
        else:
            return f"DnaSegment(id={self.id}, {self.start}:{self.end}, {self.source})"

    def __eq__(self, other) -> bool:
        return isinstance(other, DnaSegment) and self.id == other.id

    def __hash__(self) -> int:
        return self.id


class Chromosome:
    """
    A single chromosome or extrachromosomal DNA segment after some genome alterations.
    DNA segments could be derived from DNA segments.
    """

    id: int
    """A unique ID for the DNA segment."""
    segments: "List[DnaSegment]" = dataclasses.field(repr=False)
    """The segments from which this DNA segment is composed."""
    cent_loc: int
    """The center of the chromosome."""

    def __init__(
        self, dna_segments: Optional[Iterable[DnaSegment]] = None, cent_loc=None
    ):
        self.id = gen_id()

        self.segments = []
        if dna_segments:
            self.segments.extend(dna_segments)

        if cent_loc is not None:
            self.cent_loc = cent_loc
        elif self.segments:
            self.cent_loc = len(self) // 2
        else:
            self.cent_loc = 0

    def __len__(self) -> int:
        return sum(len(seg) for seg in self.segments)

    def flat_slice(self, start: int, end: int) -> "Chromosome":
        """
        Gets a slice of the given chromosomes.
        All segments in the result point to the sources of the segments in `self`.
        Flat slices are used when doing manipulations on a subclone, since mutations in one segment of the subclone shouldn't affect other segments.
        """
        res_segments = []

        pos = 0
        for seg in self.segments:
            s = max(start - pos, 0)
            e = min(end - pos, len(seg))
            if s < e:
                res_segments.append(
                    DnaSegment(seg.source, seg.start + s, seg.start + e)
                )
            pos += len(seg)

        return Chromosome(res_segments)

    def ref(self) -> "Chromosome":
        return Chromosome(
            (DnaSegment(seg, 0, len(seg)) for seg in self.segments),
            self.cent_loc,
        )

    def clone(self) -> "Chromosome":
        """
        Returns a deep copy of the DNA segment.
        Note that the DNA segments do not need to be deep-copied because they are frozen.
        """
        return Chromosome([seg.clone() for seg in self.segments])

    def __eq__(self, other) -> bool:
        return isinstance(other, Chromosome) and self.id == other.id

    def __hash__(self) -> int:
        return self.id

    def flat_representation(self) -> "Chromosome":
        """
        Represents the chromosome as copies of the original chromosomes, without taking the internal evolution into account.
        Essentially flattens the tree to depth 1.
        """
        res = []
        for seg in self.segments:
            start, source = seg.map_to_source(0)
            res.append(DnaSegment(source, start, start + len(seg)))

        return Chromosome(res)

    def delete(self, start: int, end: int):
        """
        Deletes a segment.
        """
        self.cent_loc = self.map_center_on_deletion(start, end)
        self.segments = (
            self.flat_slice(0, start).segments
            + self.flat_slice(end, len(self)).segments
        )

    def insert(self, pos: int, segments: List[DnaSegment]):
        """
        Inserts some segments to the chromosomes.
        """
        self.cent_loc = self.map_center_on_insertion(
            pos, sum(map(len, segments))
        )
        self.segments = (
            self.flat_slice(0, pos).segments
            + segments
            + self.flat_slice(pos, len(self)).segments
        )

    def substitute(self, start: int, end: int, segments: List[DnaSegment]):
        sub_length = sum(map(len, segments))
        self.cent_loc = (
            self.cent_loc
            if self.cent_loc < start
            else start
            if self.cent_loc < end
            else self.cent_loc - end + start + sub_length
        )
        self.segments = (
            self.flat_slice(0, start).segments
            + segments
            + self.flat_slice(end, len(self)).segments
        )

    def map_center_on_deletion(self, start: int, end: int) -> int:
        """
        Returns the new cent_loc index after a deletion.
        """
        if self.cent_loc < start:
            return self.cent_loc
        if self.cent_loc < end:
            return start
        return self.cent_loc - end + start

    def map_center_on_insertion(self, pos: int, length: int) -> int:
        """
        Returns the new cent_loc index after an insertion.
        """
        if self.cent_loc < pos:
            return self.cent_loc
        return self.cent_loc + length


class Subclone:
    """
    A tumor subclone.
    """

    id: int
    """A unique ID for the subclone."""
    parent: "Optional[Subclone]"
    """The subclone from which this subclone is derived."""
    children: "List[Subclone]"
    """The subclones derived from this subclone."""
    chromosomes: Set[Chromosome]
    """The chromosomes in the cells of the subclone."""
    ccf: float
    """The fraction of the tumor cells that come from the subclone."""
    mutation_weight: float
    """An abstract term combining the mutation rate of the subclone and the evolutionary time it had.
    Used to determine how many mutations originate in the subclone."""

    def __init__(
        self, parent: "Optional[Subclone]", ccf: float, mutation_weight: float
    ):
        """
        Initializes a new subclone.
        Initially, its chromosomes are a copy of the parent chromosomes and it has no child subclones.
        """
        self.id = gen_id()
        self.children = []
        self.ccf = ccf
        self.mutation_weight = mutation_weight
        self.chromosomes = set()
        self.parent = None

        if parent is not None:
            self.set_parent(parent)
            for chrom in parent.chromosomes:
                self.chromosomes.add(chrom.ref())

    def __eq__(self, other) -> bool:
        return isinstance(other, Subclone) and self.id == other.id

    def __hash__(self) -> int:
        return self.id

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def set_parent(self, parent: "Subclone"):
        """
        Sets the given subclone to be a child of this subclone.
        """
        assert self.parent is None
        self.parent = parent
        parent.children.append(self)

    def preorder_iter(self) -> "Iterator[Subclone]":
        """Returns an iterator over the subtree defined by the subclone."""
        yield self
        for child in self.children:
            yield from child.preorder_iter()

    def postorder_iter(self) -> "Iterator[Subclone]":
        """Returns an iterator over the subtree defined by the subclone."""
        for child in self.children:
            yield from child.preorder_iter()
        yield self

    def residual_ccf(self) -> float:
        """The CCF of this subclone that is not part of its child subclones."""
        return self.ccf - sum(child.ccf for child in self.children)

    def add_chromosome(self, chromosome: Chromosome):
        """
        Adds the chromosome to the subclone.
        A chromosome cannot be added once child nodes were created.
        """
        assert self.is_leaf(), (
            "Chromosomes cannot be added to a non-leaf Subclone, since they would have to affect its children!"
        )
        self.chromosomes.add(chromosome)

    def __repr__(self) -> str:
        return f"Subclone({self.id}, parent={self.parent.id if self.parent is not None else None}, ccf={self.ccf})"

    def add_arm(
        self,
        chrom: Optional[Chromosome] = None,
        p_whole=0.5,
        p_q=0.5,
        p_deletion=0.6,
    ):
        """Add an arm level copy number event to the profile given the specifications.

        Will not add an arm-level homozygous deletion."""

        if chrom is None:
            chrom = random.choice(list(self.chromosomes))

        # The chromosome ploidities dict stores for each chromosome its ploidity.
        # This is used to ensure that we don't reach a subclone missing all copies of some chromosome, which makes little sense.
        chromosome_ploidities: dict[str, IntervalTree] = defaultdict(
            IntervalTree
        )
        for chr in self.chromosomes:
            for seg in chr.flat_representation().segments:
                assert not isinstance(seg.source, DnaSegment)
                chromosome_ploidities[seg.source[0]][seg.start : seg.end] = W(1)

        chromosome_ploidities = {
            chr: W.split_merge_unwrap(tree, np.add)
            for chr, tree in chromosome_ploidities.items()
        }

        start = 0
        end = len(chrom)
        whole = np.random.rand() < p_whole
        # choose arm-level vs. whole chromosome event
        if not whole:
            if np.random.rand() > p_q:
                start = chrom.cent_loc
            else:
                end = chrom.cent_loc

        if np.random.rand() < p_deletion:  # P(deletion)
            # Checking if this would result in total deletion of chromosome segments, as this is selected against.
            deleted_segment = chrom.flat_slice(start, end).flat_representation()
            deletions: defaultdict[str, IntervalTree] = defaultdict(
                IntervalTree
            )
            for seg in deleted_segment.segments:
                assert not isinstance(seg.source, DnaSegment)
                deletions[seg.source[0]][seg.start : seg.end] = 1

            for tree in deletions.values():
                tree.split_overlaps()
                tree.merge_overlaps(np.add)

            zeroed_length = 0

            deleted_interval: Interval
            original_ploidities: Interval
            for chr, tree in deletions.items():
                for deleted_interval in tree:
                    for original_ploidities in chromosome_ploidities[chr][
                        deleted_interval.begin : deleted_interval.end
                    ]:
                        if original_ploidities.data == deleted_interval.data:
                            zeroed_length += (
                                deleted_interval.end - deleted_interval.begin
                            )

            if zeroed_length / (end - start) > 0.3:
                print(
                    f"Homozygous deletion will not be added for chrom {chrom}."
                )
                return

            # Performing the deletion.
            self.chromosomes.remove(chrom)
            if not whole:
                if start > 0:
                    self.chromosomes.add(chrom.flat_slice(0, start))
                else:
                    self.chromosomes.add(chrom.flat_slice(end, len(chrom)))

        else:
            # The duplicated arms are usually independent.
            self.chromosomes.add(chrom.flat_slice(start, end))

    def add_focal(
        self,
        median_focal_length=1.8 * 10**6,
        cnv_lambda=0.8,
        chrom: Optional[Chromosome] = None,
        p_deletion=0.5,
        position: Optional[Tuple[int, int]] = None,
        cnv_level: Optional[int] = None,
    ):
        """Add a focal copy number event to the profile, according to the specifications.

        :returns (start_position, end_position), for ease of calling same (random) CN event on both alleles"""
        if not chrom:  # choose chromosome
            chrom = random.choice(list(self.chromosomes))

        if not position:
            # choose length of event - from exponential
            focal_length_rate = median_focal_length / np.log(2)
            focal_length = np.floor(
                np.random.exponential(focal_length_rate)
            ).astype(int)
            start_pos = np.random.randint(0, max(2, len(chrom) - focal_length))
            end_pos = start_pos + focal_length
        else:
            start_pos = position[0]
            end_pos = position[1]

        if (np.random.rand() < p_deletion) or (cnv_level == 0):
            # Focal deletion.
            chrom.delete(start_pos, end_pos)
        else:
            # Focal amplification. I model this as remaining the the same chromosome.
            if cnv_level is None:
                cnv_level = np.random.poisson(cnv_lambda) + 1

            assert cnv_level > 0
            chrom.insert(
                start_pos,
                [
                    seg.clone()
                    for seg in chrom.flat_slice(start_pos, end_pos).segments
                    * (cnv_level - 1)
                ],
            )

        return start_pos, end_pos

    def add_wgd(self, both_alleles=True):
        """Add whole genome doubling for the specified cluster."""
        assert both_alleles
        self.chromosomes.update(chrome.clone() for chrome in self.chromosomes)

        # alleles = [PATERNAL, MATERNAL]
        # shuffle(alleles)
        # for chrom in self.chromosomes.keys():
        #     # apply whole arm amplification to each chromosome
        #     self.add_arm(cluster_num, 1, chrom=chrom, p_deletion=0, allele=alleles[0])
        #     if both_alleles:
        #         self.add_arm(cluster_num, 1, chrom=chrom, p_deletion=0, allele=alleles[1])

    def add_chromothripsis(
        self,
        chrom: Optional[Chromosome] = None,
        cn_states=2,
        num_events: Optional[int] = None,
        median_focal_length=1.8 * 10**6,
    ):
        """Adds a Chromothripsis event."""
        if not chrom:
            chrom = random.choice(list(self.chromosomes))

        # get number of events
        if not num_events:
            num_events = np.random.randint(20, 70)

        # generate sizes of events
        focal_length_rate = median_focal_length / np.log(2)
        sizes = np.floor(
            np.random.exponential(focal_length_rate, num_events)
        ).astype(int)
        states = np.random.randint(cn_states, size=num_events)

        if sizes.sum() >= len(chrom):
            print("Failed to add chromothripsis on extremely short chromosome.")
            return

        start_pos = np.random.randint(0, len(chrom) - sizes.sum())
        end_pos = start_pos + sizes.sum()

        starts = np.concatenate([[0], np.cumsum(sizes)])
        fragments: List[Chromosome] = [
            chrom.flat_slice(s, e) for s, e in zip(starts, starts[1:])
        ]
        random.shuffle(fragments)
        merged: List[DnaSegment] = sum(
            (frag.segments * state for frag, state in zip(fragments, states)),
            [],
        )

        chrom.substitute(start_pos, end_pos, merged)

    def add_cn_loh(
        self,
        p_whole=0.5,
        chrom: Optional[Chromosome] = None,
        focal=False,
        origin_length_cutoff=0.7,
        median_focal_length=1.8 * 10**6,
    ):
        """
        Add loss of heterozygosity event (deletion of one allele, amplification of the other)
        Call add_arm (default) or add_focal (if focal attribute is set to True) twice, once for each allele.

        To do this properly, I need to find two homologous segments, which is non-trivial.
        """
        if chrom is None:
            chrom = random.choice(list(self.chromosomes))

        if not focal:  # for chromosome level event
            # When not handling a focal event, we pair chromosomes by sort-of chromosome of origin, then delete one and duplicate the other.
            # We also find the origin of the given chromosome.
            by_origin = defaultdict(list)
            origin: Optional[Tuple[str, Haplotype]] = None
            for curr_chrom in self.chromosomes:
                curr_length = len(curr_chrom)
                flat = curr_chrom.flat_representation()
                counter: defaultdict[Tuple[str, Haplotype], int] = defaultdict(
                    int
                )
                for seg in flat.segments:
                    assert not isinstance(seg, DnaSegment)
                    counter[seg.source] += len(seg)
                for frag_origin, frag_length in counter.items():
                    if frag_length / curr_length > origin_length_cutoff:
                        by_origin[frag_origin].append(curr_chrom)
                        if curr_chrom == chrom:
                            origin = frag_origin

            if origin is None:
                print(
                    "Failed to add a loss-of-homozygosity event since the chromosome did not have a specific origin."
                )
                return

            target_origin = (origin[0], origin[1].other())

            if len(by_origin[target_origin]) == 1:
                print(
                    "Failed to add a loss-of-homozygosity event since there is no appropriate homologous protein."
                )
                return

            # Choosing a chromosome
            target: Chromosome = random.choice(by_origin[target_origin])

            # choose arm-level vs. whole chromosome event
            if np.random.rand() > p_whole:
                if random.random() < 0.5:
                    target.segments = (
                        chrom.flat_slice(0, chrom.cent_loc).segments
                        + target.flat_slice(
                            target.cent_loc, len(target)
                        ).segments
                    )
                    target.cent_loc = chrom.cent_loc
                else:
                    target.segments = (
                        target.flat_slice(0, target.cent_loc).segments
                        + chrom.flat_slice(chrom.cent_loc, len(chrom)).segments
                    )
            else:
                self.chromosomes.remove(target)
                self.chromosomes.add(chrom.clone())
        else:
            # When doing a focal LOH event, we start by sampling the focus, then search for another chromosome where this focus can be substituted.
            # Horridly generic.
            focal_length_rate = median_focal_length / np.log(2)
            focal_length = np.floor(
                np.random.exponential(focal_length_rate)
            ).astype(int)

            # Sampling the area undergoing LOH.
            weights = np.array([len(interval) for interval in chrom.segments])
            weights -= focal_length
            weights = np.maximum(weights, 0)
            weights /= weights.sum()

            seg = chrom.segments[np.random.choice(len(weights), weights)]
            start = np.random.randint(len(seg) - focal_length)
            end = start + focal_length

            flat_start, src = seg.map_to_source(start)
            flat_end = flat_start + focal_length

            other_src = (src[0], src[1].other())

            other_chromosomes = list(self.chromosomes)
            random.shuffle(other_chromosomes)

            for other_chrom in self.chromosomes:
                positions = []
                pos = 0
                for seg in other_chrom.flat_representation().segments:
                    if (
                        seg.source == other_src
                        and seg.start <= flat_start
                        and seg.end >= flat_end
                    ):
                        positions.append(
                            (
                                pos + flat_start - seg.start,
                                pos + flat_end - seg.start,
                            )
                        )
                if positions:
                    (s, e) = random.choice(positions)
                    other_chrom.substitute(
                        s, e, chrom.flat_slice(start, end).segments
                    )


class Phylogeny:
    """
    A class storing the whole process of the evolution of the tumor.
    """

    num_subclones: int
    """The number of subclones."""
    subclones: List[Subclone]
    """A list of the subclones sorted such that if some subclone was derived from another, it will come after it in the list."""
    chromosomes: Dict[Tuple[str, Haplotype], DnaSegment]
    """A map from the chromosome description to the DNA segment representing it."""

    def __init__(
        self,
        num_subclones: int,
        chromosomes: List[Tuple[str, Haplotype]],
        chromosome_sizes: Dict[str, int],
        arm_num: int,
        focal_num: int,
        p_whole: float,
        ratio_clonal: float,
        median_focal_length=1.8 * 10**6,
        chromothripsis=False,
        wgd=False,
    ):
        """
        Initializes aphylogeny with the given number of subclones.
        TODO: Improve the algorithm. For large number of `num_subclones`, this gives a line instead of a tree.
        """
        self.num_subclones = num_subclones

        # If there are no subclones, all events occur in the clonal node.
        if self.num_subclones == 0:
            ratio_clonal = 1

        self.chromosomes = {}
        for s, h in chromosomes:
            self.chromosomes[s, h] = DnaSegment((s, h), 0, chromosome_sizes[s])

        # Initializing the tumor subclone before any CNA.
        self.subclones = [Subclone(None, 1, 1)]
        for chr in self.chromosomes.values():
            self.subclones[-1].add_chromosome(Chromosome([chr]))
        # Initializing the last common ancestor of the tumor.
        self.subclones.append(Subclone(self.subclones[-1], 1.0, 1.0))

        for _ in np.arange(arm_num * ratio_clonal):
            self.subclones[-1].add_arm(p_whole=p_whole)
        for _ in np.arange(focal_num * ratio_clonal):
            self.subclones[-1].add_focal(median_focal_length)
        if wgd:
            self.subclones[-1].add_wgd()
        if chromothripsis:
            self.subclones[-1].add_chromothripsis()

        if num_subclones > 0:
            subclone_arm_mean = arm_num * (1 - ratio_clonal) / num_subclones
            subclone_focal_mean = arm_num * (1 - ratio_clonal) / num_subclones

        # Initializing subclones.
        ccfs = sorted(np.random.rand(self.num_subclones))
        par_idx = 1
        while ccfs:
            rem_ccf = self.subclones[par_idx].ccf
            ccf_idx = len(ccfs) - 1
            while ccf_idx >= 0:
                if ccfs[ccf_idx] < rem_ccf:
                    rem_ccf -= ccfs[ccf_idx]
                    self.subclones.append(
                        Subclone(
                            self.subclones[par_idx],
                            ccfs.pop(ccf_idx),
                            random.random(),
                        )
                    )

                    for _ in np.arange(np.random.poisson(subclone_arm_mean)):
                        self.subclones[-1].add_arm()
                    for _ in np.arange(np.random.poisson(subclone_focal_mean)):
                        self.subclones[-1].add_focal()

                ccf_idx -= 1
            par_idx += 1

    def get_chromosome_ploidities(
        self,
    ) -> Dict[Tuple[str, Haplotype], IntervalTree]:
        """
        Computes for each chromosome and haplotype its ploidity in each subclone.
        """
        ploidities: defaultdict[Tuple[str, Haplotype], IntervalTree] = (
            defaultdict(IntervalTree)
        )

        for subclone in self.subclones:
            ccf = subclone.residual_ccf()
            for chrom in subclone.chromosomes:
                for seg in chrom.flat_representation().segments:
                    assert isinstance(seg.source, tuple), (
                        f"The flat representation contains an illegal interval: {seg.source}"
                    )
                    ploidities[seg.source][seg.start : seg.end] = W(ccf)

        return {
            chr: W.split_merge_unwrap(tree, np.add)
            for chr, tree in ploidities.items()
        }

    def get_segment_ploidities(self) -> Dict[DnaSegment, IntervalTree]:
        """
        Computes the number of times each DNA segment appears:
        For every DNA segment, this will contain an IntervalTree whose data is the ploidity of the mutation band of that DNA segment.
        """
        res = defaultdict(IntervalTree)

        for sc in self.subclones:
            ccf = sc.residual_ccf()
            for chrom in sc.chromosomes:
                for seg in chrom.segments:
                    s = 0
                    e = len(seg)
                    while isinstance(seg, DnaSegment):
                        res[seg][s:e] = W(ccf)

                        # Note: this might become buggy when reversals become legal,
                        # and the correct thing to do might be a "map_segment_to_source" method.
                        s += seg.start
                        e += seg.start
                        seg = seg.source

        res = {
            key: W.split_merge_unwrap(val, np.add) for key, val in res.items()
        }

        return res

    def get_mutation_bands(self) -> Dict[str, IntervalTree]:
        """
        Computes the observed mutation bands.
        Returns a dictionary, mapping each chromosome to an interval tree, whose data is a [2, bands] NP-array of the band ploidity and the band weight.
        """
        segment_ploidities = self.get_segment_ploidities()
        mutation_bands: Dict[str, IntervalTree] = defaultdict(IntervalTree)

        # Computing the mutation bands for chromosome segments.
        for sc in self.subclones:
            weight = sc.ccf * sc.mutation_weight
            for chr in sc.chromosomes:
                for seg in chr.segments:
                    tree = segment_ploidities[seg]
                    for inter in tree:
                        offset, (source_chr, _) = seg.map_to_source(0)
                        mutation_bands[source_chr][
                            offset + inter.begin : offset + inter.end
                        ] = W(np.array([[inter.data], [weight]]))

        # Combining the data to a single NP array for each segment.
        return {
            chr: W.split_merge_unwrap(
                tree, lambda x, y: np.concatenate([x, y], axis=1)
            )
            for chr, tree in mutation_bands.items()
        }

    def to_profile(self) -> CNV_Profile:
        """
        Makes a CNV profile for the given phylogeny.
        """
        res = CNV_Profile(0)
        res.cent_loc = {}
        res.chromosomes = {}
        res.chromosome_size = {}
        raw_ploidities = self.get_chromosome_ploidities()
        ploidities = {}
        for k, v in raw_ploidities.items():
            # Converting the ploidities from ints specifying the ploidity to `Event`s.
            t = IntervalTree()
            for inter in v:
                t[inter.begin : inter.end] = cnv_profile.Event(
                    "", k[1], 1, inter.data
                )
            ploidities[k] = t

        sc = self.subclones[0]
        for chr in sc.chromosomes:
            s0 = chr.segments[0]
            assert not isinstance(s0.source, DnaSegment)
            chromosome_name, haplotype = s0.source
            res.cent_loc[chromosome_name] = chr.cent_loc
            res.chromosome_size[chromosome_name] = len(chr)

        for chr_name in res.cent_loc:
            res.chromosomes[chr_name] = cnv_profile.Chromosome(
                chr_name, res.chromosome_size[chr_name]
            )
            # If a chromosome is missing, we still have to input it into the tree, or suffer the consequences.
            res.chromosomes[chr_name].paternal_tree = ploidities.get(
                (chr_name, PATERNAL)
            ) or IntervalTree(
                [
                    Interval(
                        0,
                        res.chromosome_size[chr_name],
                        cnv_profile.Event("No chromosome", PATERNAL, 0, 0),
                    )
                ]
            )
            res.chromosomes[chr_name].maternal_tree = ploidities.get(
                (chr_name, MATERNAL)
            ) or IntervalTree(
                [
                    Interval(
                        0,
                        res.chromosome_size[chr_name],
                        cnv_profile.Event("No chromosome", MATERNAL, 0, 0),
                    )
                ]
            )

        # Creating the phylogeny object.
        res.phylogeny.num_subclones = self.num_subclones
        subclone_ids = [subclone.id for subclone in self.subclones]
        id_map = {id: idx for idx, id in enumerate(subclone_ids)}

        for sc in self.subclones:
            res.phylogeny.ccfs[id_map[sc.id]] = sc.ccf
            if sc.parent is None:
                res.phylogeny.parents[id_map[sc.id]] = None
            else:
                res.phylogeny.parents[id_map[sc.id]] = id_map[sc.parent.id]

        # Creating the mutation bands.
        res.mutation_bands = self.get_mutation_bands()

        return res


def read_chr_length(path: PathLike) -> Dict[str, int]:
    """
    Reads the lengths of chromosomes from a file describing them.
    """
    res = {}
    for row in open(path).readlines():
        chr, length = row.split()
        res[chr] = int(length)
    return res


def simulate_tumor(
    name: str,
    coverages: Path,
    snps: Path,
    target_read_depths: Path,
    purity: float,
    *,
    num_subclones=0,
    **kwargs,
):
    """
    Simulates a tumor sample.
    """
    # Reading the tumor sizes from the SNPs and the coverages.
    snv_df = read_snvs(snps)
    cov_df = read_coverage(coverages)
    read_depth_df = switch_contigs(
        pd.read_csv(
            target_read_depths,
            sep="\t",
            header=0,
            names=["CHROM", "POS", "DEPTH"],
            dtype={"CHROM": str},
        )
    )

    max_snv: Dict[str, int] = snv_df.groupby("CHROM").max().to_dict()["POS"]
    max_cov: Dict[str, int] = cov_df.groupby("chrom").max().to_dict()["end"]
    max_read: Dict[str, int] = (
        read_depth_df.groupby("CHROM").max().to_dict()["POS"]
    )

    chromosome_names: Set[str] = set(max_cov.keys())  # type: ignore
    chromosome_sizes = {
        c: max(max_snv.get(c) or 0, max_cov.get(c) or 0, max_read.get(c) or 0)
        for c in chromosome_names
    }

    snv_df = snv_df.query("CHROM in @chromosome_names")
    cov_df = cov_df.query("chrom in @chromosome_names")
    read_depth_df = read_depth_df.query("CHROM in @chromosome_names")

    is_male = len(cov_df["chrom"].unique()) == 24

    chromosomes = [
        (f"{i}", hap)
        for i in range(1, 23)
        for hap in [MATERNAL, PATERNAL]
        if str(i) in chromosome_names
    ] + [("23", MATERNAL)]

    if is_male:
        chromosomes.append(("24", PATERNAL))
        # No SNPs in the X chromosome in males.
        snv_df = snv_df.query('CHROM != "23"')
    else:
        chromosomes.append(("23", PATERNAL))
        print("female", snv_df["CHROM"].unique())

    ph = Phylogeny(num_subclones, chromosomes, chromosome_sizes, **kwargs)

    prof = ph.to_profile()
    prof._calculate_cnv_profile()
    prof._calculate_df_profiles()
    assert prof.cnv_profile_df is not None, (
        "CNV Profile dataframe was not calculated!"
    )

    out_path = OUT_PATH / name
    if out_path.exists():
        return
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)

    prof.to_pickle(OUT_PATH / f"{name}_profile.pickle")

    dump_tsv(
        prof.generate_random_mutation_file(1e-6, 25),
        OUT_PATH / f"{name}_mutation_plan.tsv",
    )
    dump_tsv(
        prof.generate_mutations(OUT_PATH / f"{name}_mutation_plan.tsv", purity),
        OUT_PATH / f"{name}_mutations.tsv",
    )
    (OUT_PATH / f"{name}_mutation_plan.tsv").unlink()

    simulate_coverage_and_depth(
        (OUT_PATH / f"{name}_profile.pickle").open("rb"),
        cov_df,
        snv_df,
        read_depth_df,
        purity,
        OUT_PATH / f"{name}_coverage.vcf",
        OUT_PATH / f"{name}_output_hets.vcf",
        do_parallel=False,
    )

    prof.cnv_profile_df.to_csv(
        OUT_PATH / f"{name}_profile.tsv", sep="\t", index=False
    )


def simulate_tumor_na12878(name: str, purity: float, num_subclones, **kwargs):
    """
    Simulates a tumor with some phylogeny.
    """
    ph = Phylogeny(
        num_subclones,
        [(f"{i}", hap) for i in range(1, 24) for hap in [MATERNAL, PATERNAL]],
        read_chr_length(DATA_PATH / "NA12878_csizes.tsv"),
        **kwargs,
    )

    prof = ph.to_profile()
    prof._calculate_cnv_profile()
    prof._calculate_df_profiles()
    assert prof.cnv_profile_df is not None, (
        "CNV Profile dataframe was not calculated!"
    )

    out_path = OUT_PATH / name
    if out_path.exists():
        return
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)

    prof.to_pickle(OUT_PATH / f"{name}_profile.pickle")

    dump_tsv(
        prof.generate_random_mutation_file(1e-6, 25),
        OUT_PATH / f"{name}_mutation_plan.tsv",
    )
    dump_tsv(
        prof.generate_mutations(OUT_PATH / f"{name}_mutation_plan.tsv", purity),
        OUT_PATH / f"{name}_mutations.tsv",
    )
    (OUT_PATH / f"{name}_mutation_plan.tsv").unlink()

    simulate_coverage_and_depth(
        (OUT_PATH / f"{name}_profile.pickle").open("rb"),
        DATA_PATH / "NA12878_platinum_realigned_covcollect.bed",
        DATA_PATH / "NA12878.vcf",
        OUT_PATH / "read_depth.tsv",
        purity,
        OUT_PATH / f"{name}_coverage.vcf",
        OUT_PATH / f"{name}_output_hets.vcf",
        do_parallel=False,
    )


def main():
    """
    Running simulations as in the HapASeg paper (Except I don't have the TWIST capture y)
    """

    params = [
        (
            "ch1022gl",
            DATA_PATH / "CH1022GL_NA12878_covcollect_chrs_excluded.bed",
            DATA_PATH / "NA12878.vcf",
            DATA_PATH / "read_depth.tsv",
        ),
        (
            "ch1032ln",
            DATA_PATH / "CH1032LN_NA12878_covcollect_chrs_excluded_2.bed",
            DATA_PATH / "NA12878.vcf",
            DATA_PATH / "read_depth.tsv",
        ),
        (
            "na12878",
            DATA_PATH / "NA12878_platinum_realigned_covcollect.bed",
            DATA_PATH / "NA12878.vcf",
            DATA_PATH / "read_depth.tsv",
        ),
    ]
    print(params)
    for base_name, coverage, snps, read_depths in params:
        for purity in np.linspace(0.1, 0.9, 9):
            for i in range(10):
                arm_n = np.random.choice(np.arange(20, 300))
                focal_n = np.random.choice(np.arange(20, 500))
                num_subclones = np.random.choice(3)

                arm_num = np.random.binomial(arm_n, 0.25)
                focal_num = np.random.binomial(focal_n, 0.1)
                p_whole = np.random.beta(2, 2)
                ratio_clonal = (
                    np.random.beta(4, 1) if num_subclones > 0 else 1.0
                )

                name = f"mass_2/{base_name}/pur_{int(round(purity * 10)):02}/s_{i}/sim"

                simulate_tumor(
                    name,
                    coverages=coverage,
                    snps=snps,
                    target_read_depths=read_depths,
                    purity=purity,
                    arm_num=arm_num,
                    focal_num=focal_num,
                    p_whole=p_whole,
                    ratio_clonal=ratio_clonal,
                    median_focal_length=1.8 * 10**6,
                    num_subclones=num_subclones,
                )

                with open(OUT_PATH / f"{name}_params.txt", "w") as f:
                    f.write(
                        str(
                            dict(
                                arm_num=arm_num,
                                focal_num=focal_num,
                                p_whole=p_whole,
                                ratio_clonal=ratio_clonal,
                            )
                        )
                    )

    # print("Simulating CH1032LN")
    # for purity in tqdm(np.linspace(0.025, 0.975, 20)):
    #     name = f"ch1032ln/pur_{int(round(purity * 1000)):04}/sim"
    #     simulate_tumor(
    #         name,
    #         coverages=DATA_PATH / "CH1032LN_NA12878_covcollect_chrs_excluded_2.bed",
    #         snps=DATA_PATH / "NA12878.vcf",
    #         target_read_depths=DATA_PATH / "read_depth.tsv",
    #         purity=purity,
    #         arm_num=20,
    #         focal_num=600,
    #         p_whole=0.6,
    #         ratio_clonal=0.5,
    #         median_focal_length=1.8 * 10**6,
    #         num_subclones=2,
    #     )

    # simulate_tumor_na12878(
    #     "test",
    #     0.7,
    #     0,
    #     arm_num=20,
    #     focal_num=600,
    #     p_whole=0.6,
    #     ratio_clonal=0.5,
    #     median_focal_length=1.8 * 10**6,
    # )

    # for purity in np.linspace(0.025, 0.975, 20):
    #     simulate_tumor_na12878(
    #         f"pur_{int(round(purity*1000)):04}/sim",
    #         purity,
    #         2,
    #         arm_num=20,
    #         focal_num=600,
    #         p_whole=0.6,
    #         ratio_clonal=0.5,
    #         median_focal_length=1.8 * 10**6,
    #     )


if __name__ == "__main__":
    main()

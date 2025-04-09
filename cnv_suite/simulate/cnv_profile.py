#!/bin/bash/env python3

from dataclasses import dataclass
import io
import os.path
from pathlib import Path
import random
import pandas as pd
import numpy as np
from intervaltree import IntervalTree, Interval
from collections import defaultdict, deque
from random import choice, shuffle
from natsort import natsort_keygen
from pandarallel import pandarallel
import pickle
import tqdm
import scipy.stats as s
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from utils import PathLike, switch_contigs
from utils.simulation_utils import (
    SexChromosomes,
    get_alt_count,
    get_contigs_from_header,
    get_average_ploidy,
    read_coverage,
    read_snvs,
    single_allele_ploidy,
    Haplotype,
)

MATERNAL = Haplotype.MATERNAL
PATERNAL = Haplotype.PATERNAL


@dataclass
class Event:
    type: str
    """The type of the CNA event, such as 'focal' or 'arm'."""
    allele: Haplotype
    """The paternal or maternal allele."""
    cluster_num: int
    """The subclone in which the event had happened."""
    cn_change: int
    """The change in the chromosome number caused by the event."""

    # TODO: Add a consistent timing to the events.


DEFAULT_CSIZE = {
    "1": 249250621,
    "2": 243199373,
    "3": 198022430,
    "4": 191154276,
    "5": 180915260,
    "6": 171115067,
    "7": 159138663,
    "8": 146364022,
    "9": 141213431,
    "10": 135534747,
    "11": 135006516,
    "12": 133851895,
    "13": 115169878,
    "14": 107349540,
    "15": 102531392,
    "16": 90354753,
    "17": 81195210,
    "18": 78077248,
    "19": 59128983,
    "20": 63025520,
    "21": 48129895,
    "22": 51304566,
    "23": 156040895,
    "24": 57227415,
}


class CNV_Profile:
    cent_loc: Dict[str, int]
    chromosome_size: Dict[str, int]

    chromosomes: "Dict[str, Chromosome]"
    """A dictionary from a chromosome name to the chromosome object tracking the maternal and paternal ploidities."""
    mutation_bands: "Dict[str, IntervalTree]"
    """A dictionary from a chromosomes to a [2, bands] NP-arry. An array B represents that with weight B[i, 1] we get a mutation with ploidity B[i, 0]."""
    phylogeny: "Phylogeny"
    cnv_trees: "Optional[Dict[str, Tuple[IntervalTree, IntervalTree]]]"
    cnv_profile_df: Optional[pd.DataFrame]
    phased_profile_df: Optional[pd.DataFrame]

    def __init__(
        self,
        num_subclones=3,
        csize: dict[str, int] | PathLike | None = None,
        cent_loc: dict[str, int] | None = None,
    ):
        """Create a simulated CNV profile, built with random Phylogeny and copy number alterations.

        :param num_subclones: Desired number of phylogenetic subclones, as an int.
        :param csize: Chromosome sizes, given as dict('chr': size), genome or bed file path, or
            pandas DataFrame with columns 'chr', 'len'; default is hg19 chr sizes.
        :param cent_loc: Location of centromeres, given as dict('chr': loc), tsv file, or
            pandas DataFrame with columns 'chr', 'pos'; default is halfway point of chromosomes.
        """
        # Getting the chromosome size.
        if csize is None:
            csize = dict(DEFAULT_CSIZE)
        elif not isinstance(csize, dict):
            if isinstance(csize, PathLike) and os.path.exists(csize):
                _, ext = os.path.splitext(csize)
                if ext == ".bed":
                    columns = [
                        "chr",
                        "start",
                        "len",
                    ]  # three columns if bed file
                else:
                    columns = ["chr", "len"]
                csize_df = pd.read_csv(
                    csize, sep="\t", header=None, names=columns
                )
            elif isinstance(csize, pd.DataFrame):
                csize_df = csize.copy()
            else:
                raise ValueError(
                    "csize input must be one of [None, dict, file path, pandas DataFrame]"
                )
            csize_df.set_index("chr", inplace=True)
            csize = csize_df.to_dict()["len"]
        assert csize is not None
        # Getting the location of the centromere.
        if not cent_loc:
            cent_loc = {chrom: int(size / 2) for chrom, size in csize.items()}
        elif not isinstance(cent_loc, dict):
            if isinstance(cent_loc, PathLike) and os.path.exists(cent_loc):
                cent_loc_df = pd.read_csv(
                    cent_loc, sep="\t", header=None, names=["chr", "pos"]
                )
            elif isinstance(cent_loc, pd.DataFrame):
                cent_loc_df = cent_loc.copy()
                cent_loc_df.columns = ["chr", "pos"]
            else:
                raise ValueError(
                    "cent_loc input must be one of [None, dict, file path, pandas DataFrame]"
                )
            cent_loc_df.set_index("chr", inplace=True)
            cent_loc = cent_loc_df.to_dict()["pos"]
        assert cent_loc is not None

        self.cent_loc = switch_contigs(cent_loc)
        self.chromosome_size = switch_contigs(csize)

        self.chromosomes = self._init_all_chrom(self.chromosome_size)
        self.phylogeny = Phylogeny(num_subclones)
        self.cnv_trees = None
        self.cnv_profile_df = None
        self.phased_profile_df = None
        self.mutation_bands = {}

    def _init_all_chrom(
        self, chromosome_size: Dict[str, int]
    ) -> "Dict[str, Chromosome]":
        """
        Initialize event tree dictionaries with Chromosomes containing a single haploid interval for each allele.

        :param chromosome_size: A dictionary mapping each chromosome to the number of base pairs in it.
        """
        tree_dict = {}
        for chrom, size in chromosome_size.items():
            if chrom in (23, 24):
                continue
            tree = Chromosome(chrom, size)
            tree.add_seg("haploid", MATERNAL, 0, 1, 1, size)
            tree.add_seg("haploid", PATERNAL, 0, 1, 1, size)
            tree_dict[chrom] = tree

        if len(chromosome_size) == 24:
            tree = Chromosome("23", chromosome_size["23"])
            tree.add_seg("haploid", MATERNAL, 0, 1, 1, chromosome_size["23"])
            tree_dict["23"] = tree
            tree = Chromosome("24", chromosome_size["24"])
            tree.add_seg("haploid", PATERNAL, 0, 1, 1, chromosome_size["24"])
            tree_dict["24"] = tree
        else:
            tree = Chromosome("23", chromosome_size["23"])
            tree.add_seg("haploid", MATERNAL, 0, 1, 1, chromosome_size["23"])
            tree.add_seg("haploid", PATERNAL, 0, 1, 1, chromosome_size["23"])
            tree_dict["23"] = tree

        return tree_dict

    def add_cnv_events(
        self,
        arm_num: int,
        focal_num: int,
        p_whole: float,
        ratio_clonal: float,
        median_focal_length=1.8 * 10**6,
        chromothripsis=False,
        wgd=False,
    ):
        """General helper to add CNV events according to criteria.

        Whole genome doubling and chromothripsis both applied as final clonal events if specified.

        :param arm_num: total number of arm level events
        :param focal_num: total number of focal level events
        :param p_whole: probability of whole-chromosome (vs. arm-level) CN event, given as float between [0, 1]
        :param ratio_clonal: ratio of clonal events out of all events, given as float between [0, 1]
        :param median_focal_length: median length of focal CN events; default is 1.8 Mb
        :param chromothripsis: boolean if clonal Chromothripsis event is desired; default False
        :param wgd: boolean if clonal Whole Genome Doubling event is desired; default False"""
        # add clonal events
        for _ in np.arange(arm_num * ratio_clonal):
            self.add_arm(1, p_whole)
        for _ in np.arange(focal_num * ratio_clonal):
            self.add_focal(1, median_focal_length)
        if wgd:
            self.add_wgd(1)
        if chromothripsis:
            self.add_chromothripsis(1, median_focal_length=median_focal_length)

        # add subclonal events
        for cluster in np.arange(2, self.phylogeny.num_subclones + 2):
            for _ in np.arange(
                arm_num * (1 - ratio_clonal) / self.phylogeny.num_subclones
            ):
                self.add_arm(cluster, p_whole)
            for _ in np.arange(
                focal_num * (1 - ratio_clonal) / self.phylogeny.num_subclones
            ):
                self.add_focal(cluster, median_focal_length)

    def add_arm(
        self,
        cluster_num: int,
        p_whole=0.5,
        p_q=0.5,
        chrom=None,
        p_deletion=0.6,
        allele=None,
    ):
        """Add an arm level copy number event to the profile given the specifications.

        Will not add an arm-level homozygous deletion."""
        if not chrom:
            chrom = choice(list(self.chromosomes.keys()))
        start = 1
        end = self.chromosomes[chrom].length

        # choose arm-level vs. whole chromosome event
        if np.random.rand() > p_whole:
            if np.random.rand() > p_q:
                start = self.cent_loc[chrom]
            else:
                end = self.cent_loc[chrom]

        # choose maternal vs. paternal
        if not allele:
            allele = PATERNAL if np.random.rand() > 0.5 else MATERNAL

        # choose level (based on current CN and cluster number)
        pat_int, mat_int = self.calculate_cnv_lineage(
            chrom, start, end, cluster_num
        )
        desired_int = pat_int if allele == PATERNAL else mat_int
        other_int = pat_int if allele == MATERNAL else mat_int

        # if deletion:
        # - only delete so that current intervals + deletion >= -1
        # - can delete up to that value (max(1, current_intervals + 1))
        # if amplification:
        # - should probably double whatever is in current intervals
        if np.random.rand() < p_deletion:  # P(deletion)
            # check for arm level deletion in other interval list
            weighted_del = 0
            for o in other_int:
                weighted_del += o.data.cn_change == 0 * (o.end - o.begin) / (
                    end - start
                )
            deletion_adjust = 0 if weighted_del > 0.3 else 1
            if deletion_adjust == 0:
                print(
                    f"Homozygous deletion will not be added for chrom {chrom}."
                )

            if np.random.rand() < 0.3:  # delete fully
                for i in desired_int:
                    self.chromosomes[chrom].add_seg_interval(
                        "arm",
                        cluster_num,
                        -i.data.cn_change * deletion_adjust,
                        i,
                    )
            else:  # delete one copy (is there a way to delete multiple focal copies?)  - maybe get full interval tree here too to check arm vs. focal events
                current_levels = [i.data.cn_change for i in desired_int]
                desired_change = [
                    0 if lev == 0 or deletion_adjust == 0 else -1
                    for lev in current_levels
                ]
                for i, level in zip(desired_int, desired_change):
                    self.chromosomes[chrom].add_seg_interval(
                        "arm", cluster_num, level, i
                    )
        else:
            for i in desired_int:
                self.chromosomes[chrom].add_seg_interval(
                    "arm", cluster_num, i.data.cn_change, i
                )

    def add_focal(
        self,
        cluster_num: int,
        median_focal_length=1.8 * 10**6,
        cnv_lambda=0.8,
        chrom: Optional[str] = None,
        p_deletion=0.5,
        allele: Optional[Haplotype] = None,
        position: Optional[Tuple[int, int]] = None,
        cnv_level=None,
    ):
        """Add a focal copy number event to the profile, according to the specifications.

        :returns (start_position, end_position), for ease of calling same (random) CN event on both alleles"""
        if not chrom:  # choose chromosome
            chrom = choice(list(self.chromosomes.keys()))

        if not position:
            # choose length of event - from exponential
            focal_length_rate = median_focal_length / np.log(2)
            focal_length = np.floor(
                np.random.exponential(focal_length_rate)
            ).astype(int)
            start_pos = np.random.randint(
                1, max(2, self.chromosome_size[chrom] - focal_length)
            )
            end_pos = start_pos + focal_length
        else:
            start_pos = position[0]
            end_pos = position[1]

        # choose maternal vs. paternal
        if not allele:
            allele = PATERNAL if np.random.rand() > 0.5 else MATERNAL

        # get current CN intervals for this branch of phylogenetic tree
        pat_int, mat_int = self.calculate_cnv_lineage(
            chrom, start_pos, end_pos, cluster_num
        )
        desired_int = pat_int if allele == PATERNAL else mat_int

        if np.random.rand() < p_deletion:
            # lean towards fully deleting intervals (if there is already an amplification)
            for i in desired_int:
                curr_level = i.data.cn_change
                chosen_del = (
                    max(1, curr_level - np.random.poisson(curr_level / 10))
                    if curr_level != 0
                    else 0
                )
                self.chromosomes[chrom].add_seg_interval(
                    "focal", cluster_num, -chosen_del, i
                )
        else:
            if not cnv_level:
                cnv_level = np.random.poisson(cnv_lambda) + 1

            # if amplification, just add equal number for entire interval unless already fully deleted
            for i in desired_int:
                chosen_amp = cnv_level if i.data.cn_change != 0 else 0
                self.chromosomes[chrom].add_seg_interval(
                    "focal", cluster_num, chosen_amp, i
                )

        return start_pos, end_pos

    def add_wgd(self, cluster_num: int, both_alleles=True):
        """Add whole genome doubling for the specified cluster."""
        alleles = [PATERNAL, MATERNAL]
        shuffle(alleles)
        for chrom in self.chromosomes.keys():
            # apply whole arm amplification to each chromosome
            self.add_arm(
                cluster_num, 1, chrom=chrom, p_deletion=0, allele=alleles[0]
            )
            if both_alleles:
                self.add_arm(
                    cluster_num, 1, chrom=chrom, p_deletion=0, allele=alleles[1]
                )

    def add_chromothripsis(
        self,
        cluster_num: int,
        chrom=None,
        cn_states=2,
        allele=None,
        num_events=None,
        median_focal_length=1.8 * 10**6,
    ):
        """Add whole genome doubling for the specified cluster following specifications."""
        if not chrom:
            chrom = choice(list(self.chromosomes.keys()))
        if (
            not allele
        ):  # assuming all events happen on single chromatid (allele)
            allele = PATERNAL if np.random.rand() > 0.5 else MATERNAL

        # get number of events
        if not num_events:
            num_events = np.random.randint(20, 70)

        # generate sizes of events
        focal_length_rate = median_focal_length / np.log(2)
        sizes = np.floor(
            np.random.exponential(focal_length_rate, num_events)
        ).astype(int)

        # assign states (alternating if 2, more complicated if 3+)
        if cn_states == 2:
            states = [0, 1] * int(num_events / 2) + [0]
        else:
            all_states = list(range(cn_states))
            select_states = all_states[:]
            select_states.remove(1)
            states = [1] * num_events
            for i in range(num_events):
                new_state = np.random.choice(select_states)
                states[i] = new_state
                select_states = all_states[:]
                select_states.remove(new_state)

        start_pos = np.random.randint(
            1, max(2, self.chromosome_size[chrom] - sizes.sum())
        )
        for this_size, this_state in zip(sizes, states):
            end_pos = start_pos + this_size
            if this_state == 0:  # deletion
                p_deletion = 1
            else:
                p_deletion = 0

            if this_state != 1:  # skip sections of state == 1
                _, _ = self.add_focal(
                    cluster_num,
                    position=(start_pos, end_pos),
                    chrom=chrom,
                    p_deletion=p_deletion,
                    allele=allele,
                    cnv_level=this_state - 1,
                )

            start_pos = end_pos

    def add_cn_loh(
        self, cluster_num: int, p_whole=0.5, chrom=None, focal=False
    ):
        """Add loss of heterozygosity event (deletion of one allele, amplification of the other)

        Call add_arm (default) or add_focal (if focal attribute is set to True) twice, once for each allele.
        """
        alleles = [PATERNAL, MATERNAL]
        shuffle(alleles)

        if not chrom:
            chrom = choice(list(self.chromosomes.keys()))

        if not focal:  # for chromosome level event
            # choose arm-level vs. whole chromosome event
            if np.random.rand() > p_whole:
                p_q = 1 if np.random.rand() > 0.5 else 0
                p_whole = 0
            else:
                p_q = 0
                p_whole = 1

            self.add_arm(
                cluster_num,
                p_whole,
                p_q=p_q,
                chrom=chrom,
                p_deletion=0,
                allele=alleles[0],
            )
            self.add_arm(
                cluster_num,
                p_whole,
                p_q=p_q,
                chrom=chrom,
                p_deletion=1,
                allele=alleles[1],
            )
        else:  # for focal
            start_pos, end_pos = self.add_focal(
                cluster_num, chrom=chrom, p_deletion=0, allele=alleles[0]
            )
            _, _ = self.add_focal(
                cluster_num,
                position=(start_pos, end_pos),
                chrom=chrom,
                p_deletion=1,
                allele=alleles[1],
            )

    def calculate_cnv_lineage(self, chrom, start, end, cluster_num):
        """Get the CNV intervals effecting this loci in this given cluster_num and its phylogenetic parents.

        :returns (IntervalTree, IntervalTree): paternal CNV intervals, maternal CNV intervals"""
        return self.chromosomes[chrom].calc_current_cnv_lineage(
            start, end, cluster_num, self.phylogeny
        )

    def calculate_profiles(self):
        """Calculate CNV profiles based on the phylogeny and CNV events and generate CNV and phased dataframes.

        Run after adding all relevant CNV events. Can always be run again to generate profiles again if new events are added. Must be run before generating coverage or snvs."""
        self._calculate_cnv_profile()
        self._calculate_df_profiles()

    def _calculate_cnv_profile(self):
        """
        Calculates the CNV trees of the profile.
        """
        cnv_trees = {}
        for chrom, interval_tree in self.chromosomes.items():
            cnv_trees[chrom] = interval_tree.calc_full_cnv(self.phylogeny)

        self.cnv_trees = cnv_trees

    def _calculate_df_profiles(self):
        cnv_df = []
        phasing_df = []

        assert (
            self.cnv_trees is not None
        ), "Profile's CNV trees were not initialized!"
        for chrom, profile_tree in self.chromosomes.items():
            cnv_df.append(
                profile_tree.get_cnv_df(
                    self.cnv_trees[chrom][0], self.cnv_trees[chrom][1]
                )
            )
            phasing_df.append(
                profile_tree.get_phased_df(
                    self.cnv_trees[chrom][0], self.cnv_trees[chrom][1]
                )
            )

        self.cnv_profile_df = pd.concat(cnv_df).sort_values(
            ["Chromosome", "Start.bp"], key=natsort_keygen()
        )
        self.phased_profile_df = pd.concat(phasing_df).sort_values(
            ["Chromosome", "Start.bp"], key=natsort_keygen()
        )

    def generate_coverage(
        self,
        purity: float,
        cov_binned: PathLike | pd.DataFrame,
        x_coverage: Optional[bool] = None,
        sigma: Optional[float] = None,
        do_parallel=True,
    ):
        """Generate binned coverage profile based on purity and copy number profile.

        :param purity: desired purity/tumor fraction of tumor sample
        :param cov_binned: tsv file with binned coverage for genome
        :param x_coverage: optional integer to overwrite cov_binned coverage values with Log-Normal Poisson values with lambda=x_coverage
        :param sigma: optional value for Log-Normal sigma value
        :param do_parallel: boolean option to parallelize with pandarallel

        :return: pandas.DataFrame with coverage data corrected for given purity and this CN profile

        Notes:
        - The x_coverage is relative to a local ploidy of 2. Given the CN profile, it may be more or less than that.
        - local_coverage = x_coverage * ploidy / 2 where ploidy = pur*(mu_min+mu_maj) + (1-pur)*2
        - pandarallel natively uses /dev/shm for message passing and it is recommended to
          do use a single core on dockerized and memory-constrained systems
        """
        assert (
            self.cnv_trees is not None
        ), "cnv_trees not computed yet. Run calculate_profiles() before generating coverage."

        if isinstance(cov_binned, PathLike):
            x_coverage_df = read_coverage(cov_binned)
        else:
            x_coverage_df = cov_binned

        x_coverage_df = x_coverage_df[
            x_coverage_df["chrom"].isin(self.chromosomes.keys())
        ]
        if do_parallel:
            pandarallel.initialize(use_memory_fs=False)
            # bins in cov_collect bed file are inclusive, but end values should be exclusive to compare to intervals
            x_coverage_df["paternal_ploidy"] = x_coverage_df.parallel_apply(
                lambda x: single_allele_ploidy(
                    self.cnv_trees[x["chrom"]][0], x["start"], x["end"] + 1
                ),  # type: ignore
                axis=1,
            )
            x_coverage_df["maternal_ploidy"] = x_coverage_df.parallel_apply(
                lambda x: single_allele_ploidy(
                    self.cnv_trees[x["chrom"]][1], x["start"], x["end"] + 1
                ),  # type: ignore
                axis=1,
            )
        else:
            x_coverage_df["paternal_ploidy"] = x_coverage_df.apply(
                lambda x: single_allele_ploidy(
                    self.cnv_trees[x["chrom"]][0], x["start"], x["end"] + 1
                ),  # type: ignore
                axis=1,
            )
            x_coverage_df["maternal_ploidy"] = x_coverage_df.apply(
                lambda x: single_allele_ploidy(
                    self.cnv_trees[x["chrom"]][1], x["start"], x["end"] + 1
                ),  # type: ignore
                axis=1,
            )

        x_coverage_df["ploidy"] = get_average_ploidy(
            x_coverage_df["paternal_ploidy"].values,  # type: ignore
            x_coverage_df["maternal_ploidy"].values,  # type: ignore
            purity,  # type: ignore
        )

        if x_coverage:
            if not sigma:
                sigma = 1
            dispersion_norm = np.random.normal(0, sigma, x_coverage_df.shape[0])
            binned_coverage = (
                x_coverage * (x_coverage_df["end"] - x_coverage_df["start"]) / 2
            )
            this_chr_coverage = np.asarray(
                [
                    np.random.poisson(cov + np.exp(disp))
                    for cov, disp in zip(binned_coverage, dispersion_norm)
                ]
            )
            x_coverage_df["covcorr"] = this_chr_coverage

        # save original coverage values before scaling by ploidy
        x_coverage_df["covcorr_original"] = x_coverage_df["covcorr"]
        x_coverage_df["covcorr"] = np.floor(
            x_coverage_df["covcorr"].values * x_coverage_df["ploidy"].values / 2  # type: ignore
        ).astype(int)

        return x_coverage_df[
            [
                "chrom",
                "start",
                "end",
                "covcorr",
                "mean_fraglen",
                "sqrt_avg_fragvar",
                "n_frags",
                "tot_reads",
                "reads_flagged",
                "ploidy",
                "covcorr_original",
            ]
        ]

    def save_coverage_file(
        self,
        filename: PathLike,
        purity: float,
        cov_binned: PathLike | pd.DataFrame,
        x_coverage=None,
        sigma=None,
        do_parallel=True,
    ):
        """Generate coverage for given purity and binned coverage file and save output to filename"""
        cov_df = self.generate_coverage(
            purity,
            cov_binned,
            x_coverage=x_coverage,
            sigma=sigma,
            do_parallel=do_parallel,
        )
        assert isinstance(cov_df, pd.DataFrame), "Failed to generate coverage!"
        cov_df = cov_df.rename(columns={"chrom": "chr"})
        cov_df.to_csv(filename, sep="\t", index=False, header=False)

    def generate_snvs(
        self,
        snv_vcf: PathLike | pd.DataFrame,
        read_depths: PathLike | pd.DataFrame,
        purity: float,
        ref_alt=False,
        do_parallel=True,
    ):
        """Generate SNV read depths adjusted for CNV profile (and purity), with phasing from vcf file.

        :param snv_vcf: VCF file containing SNVs and haplotype of SNVs
        :param read_depths: bed file containing the read depths for all desired SNVs in original bam. The columns expected in the default settings are CHROM, POS, and DEPTH.
        :param purity: desired purity, given as float
        :param do_parallel: boolean option to parallelize with pandarallel
        :param ref_alt: True if bed file contains ref and alt counts vs. only depth counts (default False)
        """
        if self.cnv_trees is None:
            print(
                "cnv_trees not computed yet. Run calculate_profiles() before generating snvs."
            )
            return None, None

        # # check if VCF contigs given in header match contigs and lengths in self
        # vcf_contigs = switch_contigs(get_contigs_from_header(snv_vcf))
        # vcf_contigs_pertinent = {k: v for k, v in vcf_contigs.items() if k in self.chromosomes.keys()}
        # if vcf_contigs_pertinent.keys() != self.chromosomes.keys():
        #     print(
        #         f"WARNING: Not all defined contigs exist in VCF file. "
        #         f"Missing contigs: {set(self.chromosomes.keys()) - set(vcf_contigs_pertinent.keys())}"
        #     )
        # for k, v in vcf_contigs_pertinent.items():
        #     if v != self.chromosome_size[k]:
        #         print(
        #             f"WARNING: Contig length for chrom {k} in VCF file does not match CNV Profile "
        #             f"({v} vs. {self.chromosome_size[k]})."
        #         )

        if isinstance(snv_vcf, PathLike):
            snv_df = read_snvs(snv_vcf)
        else:
            snv_df = snv_vcf

        if isinstance(read_depths, PathLike):
            if ref_alt:
                bed_df = pd.read_csv(
                    read_depths,
                    sep="\t",
                    header=0,
                    names=["CHROM", "POS", "REF_BED", "ALT_BED"],
                    dtype={"CHROM": str},
                )
                bed_df["DEPTH"] = bed_df["REF_BED"] + bed_df["ALT_BED"]
            else:
                bed_df = pd.read_csv(
                    read_depths,
                    sep="\t",
                    header=0,
                    names=["CHROM", "POS", "DEPTH"],
                    dtype={"CHROM": str},
                )
        else:
            bed_df = read_depths

        # change contigs to [0-9]+ from chr[0-9XY]+ in input files
        bed_df = switch_contigs(bed_df)

        snv_df = snv_df.merge(bed_df, on=["CHROM", "POS"], how="inner")

        if do_parallel:
            pandarallel.initialize()
            snv_df["paternal_ploidy"] = snv_df.parallel_apply(
                lambda x: single_allele_ploidy(
                    self.cnv_trees[x["CHROM"]][0], x["POS"], x["POS"] + 1
                ),  # type: ignore
                axis=1,
            )
            snv_df["maternal_ploidy"] = snv_df.parallel_apply(
                lambda x: single_allele_ploidy(
                    self.cnv_trees[x["CHROM"]][1], x["POS"], x["POS"] + 1
                ),  # type: ignore
                axis=1,
            )
        else:
            snv_df["paternal_ploidy"] = snv_df.apply(
                lambda x: single_allele_ploidy(
                    self.cnv_trees[x["CHROM"]][0], x["POS"], x["POS"] + 1
                ),  # type: ignore
                axis=1,
            )
            snv_df["maternal_ploidy"] = snv_df.apply(
                lambda x: single_allele_ploidy(
                    self.cnv_trees[x["CHROM"]][1], x["POS"], x["POS"] + 1
                ),  # type: ignore
                axis=1,
            )

        snv_df["ploidy"] = get_average_ploidy(
            snv_df["paternal_ploidy"].values,  # type: ignore
            snv_df["maternal_ploidy"].values,  # type: ignore
            purity,
        )
        snv_df["maternal_prop"] = (
            snv_df["maternal_ploidy"].values * purity + (1 - purity)
        ) / snv_df["ploidy"].values  # type: ignore

        snv_df["paternal_prop"] = (
            snv_df["paternal_ploidy"].values * purity + (1 - purity)
        ) / snv_df["ploidy"].values  # type: ignore

        snv_df["maternal_present"] = snv_df["test"].apply(lambda x: x[0] == "1")
        snv_df["paternal_present"] = snv_df["test"].apply(lambda x: x[2] == "1")

        snv_df["adjusted_depth"] = np.floor(
            snv_df["DEPTH"].values * snv_df["ploidy"].values / 2
        ).astype(int)  # type: ignore

        # generate phase switch profile
        # chromosome interval trees: False if phase switched
        correct_phase_interval_trees = self.generate_phase_switching()

        # calculate alt counts for each SNV
        if do_parallel:
            snv_df["alt_count"] = snv_df.parallel_apply(
                lambda x: get_alt_count(
                    x["maternal_prop"],
                    x["paternal_prop"],
                    x["maternal_present"],
                    x["paternal_present"],
                    x["adjusted_depth"],
                    correct_phase_interval_trees[x["CHROM"]][x["POS"]]
                    .pop()
                    .data,
                ),
                axis=1,
            )  # type: ignore
        else:
            snv_df["alt_count"] = snv_df.apply(
                lambda x: get_alt_count(
                    x["maternal_prop"],
                    x["paternal_prop"],
                    x["maternal_present"],
                    x["paternal_present"],
                    x["adjusted_depth"],
                    correct_phase_interval_trees[x["CHROM"]][x["POS"]]
                    .pop()
                    .data,
                ),
                axis=1,
            )
        snv_df["ref_count"] = snv_df["adjusted_depth"] - snv_df["alt_count"]

        return snv_df, correct_phase_interval_trees

    def generate_random_mutation_file(
        self, mutation_density: float, coverage: float
    ) -> pd.DataFrame:
        """
        Generates a dataframe containing mutations in each chromosome.
        The keys of the returned dataframe are 'chr', 'pos', and 'depth'.
        """

        res = []
        for chr in self.chromosomes.values():
            pos = 0
            while True:
                pos += np.random.geometric(mutation_density)
                if pos >= chr.length:
                    break
                res.append([chr.name, pos, coverage])

        return pd.DataFrame(res, columns=["chr", "pos", "depth"])

    def generate_mutations(
        self, mut_file: PathLike, purity: float
    ) -> pd.DataFrame:
        """
        Generates the mutations.
        The mutations are randomly distributed along the chromosomes with the given density.

        :param mut_file: A file containing a dataframe of the mutations. The required fields are ['chr', 'pos', 'depth']
        :return: A dataframe with the fields ["contig", "position", "t_ref_count", "t_alt_count", "judgement"], specifying the generated mutations.
        The 'judgement' column is guaranteed to contain 'KEEP'.
        """

        mut_df = pd.read_csv(mut_file, sep="\t")
        mut_df["chr"] = mut_df["chr"].map(str)

        res = []

        total_ploidities = {
            chr_name: chr.calc_full_cnv(self.phylogeny)
            for chr_name, chr in self.chromosomes.items()
        }

        for _, row in mut_df.iterrows():
            chr = row["chr"]
            pos = row["pos"]
            depth = row["depth"]

            pat, mat = total_ploidities[chr]
            assert len(pat[pos]) == len(mat[pos]) == 1
            total = (
                next(iter(pat[pos])).data.cn_change
                + next(iter(mat[pos])).data.cn_change
            )

            band_set = self.mutation_bands[chr][pos]
            assert len(band_set) == 1
            bands, weights = next(iter(band_set)).data
            weights /= weights.sum()
            band = np.random.choice(bands, p=weights)
            p = band * purity / (total * purity + 2 * (1 - purity))

            reads = np.random.poisson(p * depth * total)
            nonreads = np.random.poisson((1 - p + 1e-6) * depth * total)
            if reads <= 3:
                continue
            res.append([chr, pos, nonreads, reads, "KEEP"])

        return pd.DataFrame(
            res,
            columns=[
                "contig",
                "position",
                "t_ref_count",
                "t_alt_count",
                "judgement",
            ],
        )

    def save_hets_file(
        self,
        out_file: PathLike,
        vcf: PathLike | pd.DataFrame,
        read_depths: PathLike | pd.DataFrame,
        purity: float,
        ref_alt=False,
        do_parallel=True,
    ):
        """
        Generate SNV adjusted depths for given purity for given bed file and save output to filename

        Parameters:
        * `filename`: ...
        * `vcf`: A VCF file describing the SNVs in the individual.
        * `read_depths`: A file describing the read depth of each SNV.
        * `purity`: The purity of the simulated tumor.
        """

        vcf_df, _ = self.generate_snvs(
            vcf, read_depths, purity, ref_alt=ref_alt, do_parallel=do_parallel
        )

        assert vcf_df is not None, "Failed to generate SNV file!"

        vcf_df.rename(
            columns={
                "CHROM": "CONTIG",
                "POS": "POSITION",
                "ref_count": "REF_COUNT",
                "alt_count": "ALT_COUNT",
            }
        )[["CONTIG", "POSITION", "REF_COUNT", "ALT_COUNT"]].to_csv(
            out_file, sep="\t", index=False
        )

    # generates seg file using poisson variance and beta noise for sigma
    def generate_profile_seg_file(
        self, filename, vcf, het_depth_bed, og_coverage_bed, purity
    ):
        snv_df, _ = self.generate_snvs(vcf, het_depth_bed, purity)
        assert snv_df is not None, "Failed to generate SNV file!"
        assert (
            self.cnv_profile_df is not None
        ), "Attempted to generate the profile seg file before generating the profile!"
        # get allele counts from snv_df
        A_alt_mask = snv_df.NA12878.apply(lambda x: int(x[0]) == 1)
        snv_df.loc[:, "A_count"] = 0
        snv_df.loc[:, "B_count"] = 0
        snv_df.loc[A_alt_mask, "A_count"] = snv_df.loc[A_alt_mask, "alt_count"]
        snv_df.loc[~A_alt_mask, "A_count"] = snv_df.loc[
            ~A_alt_mask, "ref_count"
        ]
        snv_df.loc[~A_alt_mask, "B_count"] = snv_df.loc[
            ~A_alt_mask, "alt_count"
        ]
        snv_df.loc[A_alt_mask, "B_count"] = snv_df.loc[A_alt_mask, "ref_count"]
        phased_counts = snv_df[["CHROM", "POS", "A_count", "B_count"]]

        Cov = pd.read_csv(
            og_coverage_bed,
            sep="\t",
            names=[
                "chr",
                "start",
                "end",
                "covcorr",
                "mean_frag_len",
                "std_frag_len",
                "num_frags",
                "tot_reads",
                "fail_reads",
            ],
            low_memory=False,
        )
        filt = Cov.loc[Cov.mean_frag_len > 0]
        # filter out zero bins
        mean_allele_cov = filt.covcorr.mean() / filt.mean_frag_len.mean() / 2

        prof_df = self.cnv_profile_df.reset_index(drop=True).rename(
            {"mu.major": "major_ploidy", "mu.minor": "minor_ploidy"}, axis=1
        )
        prof_df.loc[:, ["mu.major", "mu.minor"]] = np.nan
        prof_df.loc[:, ["A_count", "B_count"]] = 0
        prof_df.loc[:, ["sigma.major", "sigma.minor"]] = np.nan
        for i, row in tqdm.tqdm(prof_df.iterrows()):
            chrom, st, en, maj_ploidy, min_ploidy = row[:5]
            tot_ploidy = maj_ploidy + min_ploidy
            A, B = phased_counts.loc[
                (phased_counts.CHROM == chrom)
                & (phased_counts.POS >= st)
                & (phased_counts.POS < en),
                ["A_count", "B_count"],
            ].sum()
            prof_df.loc[i, ["A_count", "B_count"]] = A, B  # type: ignore
            if (A + B) == 0:
                continue
            A, B = (A, B) if A > B else (B, A)
            purity_corrected_cov = (mean_allele_cov * tot_ploidy * purity) + (
                mean_allele_cov * (1 - purity) * 2
            )
            major_samples: pd.DataFrame = s.poisson.rvs(
                purity_corrected_cov * s.beta.rvs(A, B, size=10000)
            )  # type: ignore
            major_mu, major_sigma = major_samples.mean(), major_samples.std()
            minor_samples: pd.DataFrame = s.poisson.rvs(
                purity_corrected_cov * s.beta.rvs(B, A, size=10000)
            )  # type: ignore
            minor_mu, minor_sigma = minor_samples.mean(), minor_samples.std()
            prof_df.at[i, ["mu.major", "sigma.major"]] = major_mu, major_sigma
            prof_df.at[i, ["mu.minor", "sigma.minor"]] = minor_mu, minor_sigma
        # prof_df.loc[:, 'sigma.minor'] = np.sqrt(prof_df.loc[:, 'mu.minor'])
        # prof_df.loc[:, 'sigma.major'] = np.sqrt(prof_df.loc[:, 'mu.major'])

        # prof_df[['Chromosome', 'Start.bp', 'End.bp', 'mu.major', 'mu.minor',
        #         'sigma.major', 'sigma.minor']].to_csv(filename, sep='\t', index=False)
        prof_df.to_csv(filename, sep="\t", index=False)

    def generate_phase_switching(self):
        phase_switches = {}
        for chrom, size in self.chromosome_size.items():
            tree = IntervalTree()
            start = 1
            correct_phase = True
            while start < size:
                interval_len = np.floor(np.random.exponential(1e6))
                if interval_len > 0:
                    tree[start : start + interval_len] = correct_phase
                    correct_phase = not correct_phase
                    start += interval_len

            phase_switches[chrom] = tree

        return phase_switches

    def to_pickle(self, filename):
        """Save self as a pickle to be imported elsewhere"""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def save_seg_file(self, filename, purity=1):
        assert 0 <= purity <= 1
        assert self.cnv_profile_df is not None
        local_cnv_profile_df = self.cnv_profile_df.copy()
        # adjust cnv profile by purity; if purity=1, profile remains the same
        local_cnv_profile_df["mu.major"] = local_cnv_profile_df[
            "mu.major"
        ] * purity + (1 - purity)
        local_cnv_profile_df["mu.minor"] = local_cnv_profile_df[
            "mu.minor"
        ] * purity + (1 - purity)
        local_cnv_profile_df.to_csv(filename, sep="\t", index=False)


class Chromosome:
    """
    An object tracking the ploidity of the maternal and paternal chromosomes.
    """

    name: str
    length: int
    paternal_tree: IntervalTree
    maternal_tree: IntervalTree

    def __init__(self, chr_name, chr_length):
        """
        A contig with IntervalTrees representing its copy number state

        :param chr_name: the given name for this contig, generally as a string
        :param chr_length: the length of this contig, as an int
        """
        self.name = chr_name
        self.length = chr_length

        # IntervalTree representing the copy number state of the paternal allele
        self.paternal_tree = IntervalTree()
        # IntervalTree representing the copy number state of the maternal allele
        self.maternal_tree = IntervalTree()

    def add_seg(
        self,
        type: str,
        allele: Haplotype,
        cluster_num: int,
        cn_change: int,
        start: int,
        end: int,
    ):
        if allele == PATERNAL:
            self.paternal_tree[start:end] = Event(
                type, allele, cluster_num, cn_change
            )
        else:
            self.maternal_tree[start:end] = Event(
                type, allele, cluster_num, cn_change
            )

    def add_seg_interval(
        self, type: str, cluster_num: int, cn_change: int, interval: Interval
    ):
        """Add segment to one of the alleles with given cluster, copy number change and interval"""
        self.add_seg(
            type,
            interval.data.allele,
            cluster_num,
            cn_change,
            interval.begin,
            interval.end,
        )

    def calc_current_cnv_lineage(
        self, start: int, end: int, cluster_num: int, phylogeny: "Phylogeny"
    ) -> Tuple[IntervalTree, IntervalTree]:
        """
        Computes a tree specifying the total ploidity at each point in the chromosome between `start` and `end`
        at the tumor subclone specified by `cluster_num`.
        """
        lineage_clusters = phylogeny.get_lineage(cluster_num)

        pat_intervals = self.paternal_tree.copy()
        pat_intervals.slice(start)
        pat_intervals.slice(end)
        pat_tree = IntervalTree()
        for i in pat_intervals.envelop(start, end):
            if i.data.cluster_num in lineage_clusters:
                pat_tree.add(i)
        pat_tree.split_overlaps()
        pat_tree.merge_overlaps(data_reducer=self.sum_levels)

        mat_intervals = self.maternal_tree.copy()
        mat_intervals.slice(start)
        mat_intervals.slice(end)
        mat_tree = IntervalTree()
        for i in mat_intervals.envelop(start, end):
            if i.data.cluster_num in lineage_clusters:
                mat_tree.add(i)
        mat_tree.split_overlaps()
        mat_tree.merge_overlaps(data_reducer=self.sum_levels)

        return pat_tree, mat_tree

    def calc_full_cnv(
        self, phylogeny: "Phylogeny"
    ) -> Tuple[IntervalTree, IntervalTree]:
        """
        Computes interval trees, specifying for each position along the chromosome the total (possibly fractional) number of paternal and maternal chromosomes,
        according to the given phylogeny and the already sampled events.

        :param phylogeny: An object containing the tumor subclones.

        :return: Trees containing the number of paternal and maternal chromosomes.
        """
        pat_tree = IntervalTree()
        for i in self.paternal_tree:
            weighted_cn = i.data.cn_change * phylogeny.ccfs[i.data.cluster_num]
            pat_tree[i.begin : i.end] = Event(
                i.data.type, i.data.allele, i.data.cluster_num, weighted_cn
            )
        pat_tree.split_overlaps()
        pat_tree.merge_overlaps(data_reducer=self.sum_levels)

        mat_tree = IntervalTree()
        for i in self.maternal_tree:
            weighted_cn = i.data.cn_change * phylogeny.ccfs[i.data.cluster_num]
            mat_tree[i.begin : i.end] = Event(
                i.data.type, i.data.allele, i.data.cluster_num, weighted_cn
            )
        mat_tree.split_overlaps()
        mat_tree.merge_overlaps(data_reducer=self.sum_levels)

        # could deliver a Chromosome (or child class) instead of just a tree
        return pat_tree, mat_tree

    def get_cnv_df(
        self, pat_tree: IntervalTree, mat_tree: IntervalTree
    ) -> pd.DataFrame:
        """
        Makes a dataframe specifying for each segment its major chromosome ploidity and its minor chromosome ploidity.
        This must be called in trees in which each segment is disjoint, ideally those created by `calc_full_cnv`.
        """
        both_alleles = IntervalTree(list(pat_tree) + list(mat_tree))
        both_alleles.split_overlaps()
        both_alleles.merge_overlaps(data_reducer=self.specify_levels)
        seg_df = []
        for segment in both_alleles:
            seg_df.append(
                [
                    self.name,
                    segment.begin,
                    segment.end,
                    segment.data["major"],
                    segment.data["minor"],
                ]
            )

        return pd.DataFrame(
            seg_df,
            columns=[
                "Chromosome",
                "Start.bp",
                "End.bp",
                "mu.major",
                "mu.minor",
            ],
        )

    def get_phased_df(
        self, pat_tree: IntervalTree, mat_tree: IntervalTree
    ) -> pd.DataFrame:
        both_alleles = IntervalTree(list(pat_tree) + list(mat_tree))
        both_alleles.split_overlaps()
        both_alleles.merge_overlaps(data_reducer=self.specify_phasing)
        seg_df = []
        for segment in both_alleles:
            seg_df.append(
                [
                    self.name,
                    segment.begin,
                    segment.end,
                    segment.data[PATERNAL],
                    segment.data[MATERNAL],
                ]
            )

        return pd.DataFrame(
            seg_df,
            columns=["Chromosome", "Start.bp", "End.bp", PATERNAL, MATERNAL],
        )

    @staticmethod
    def sum_levels(old: Event, new: Event) -> Event:
        """Returns an event specifying the sum of the CNA of the two events."""
        return Event(old.type, old.allele, 1, old.cn_change + new.cn_change)

    @staticmethod
    def specify_levels(old: Event, new: Event):
        return {
            "major": max(old.cn_change, new.cn_change),
            "minor": min(old.cn_change, new.cn_change),
        }

    @staticmethod
    def specify_phasing(old, new) -> Dict[Haplotype, int]:
        """Returns a dictionary containing the CNA of the maternal and paternal chromosome numbers individually."""
        return {
            PATERNAL: old.cn_change
            if old.allele == PATERNAL
            else new.cn_change,
            MATERNAL: old.cn_change
            if old.allele == MATERNAL
            else new.cn_change,
        }


class Phylogeny:
    """
    A class representing the subclones of a tumor.
    The subclones form a tree, and each subclone has a CCF and a relative mutation rate.

    There are two special nodes:
    - 0 represents the tumor before any chromosome number alterations.
    - 1 is the least common ancestor of all current tumor cells.
    """

    num_subclones: int
    """The number of subclones in the simulated phylogeny."""
    parents: Dict[int, Optional[int]]
    """A dictionary mapping each subclone to its parent node."""
    ccfs: Dict[int, float]
    """A dictionary mapping each subclone to its CCF"""

    def __init__(self, num_subclones: int):
        """Class to represent simulated tumor phylogeny

        :param num_subclones: desired number of subclones
        :attribute parents: dictionary mapping clones (keys) to their parent clone (values), with 1: None representing truncal branch
        :attribute ccfs: dictionary mapping clones to their CCF
        """
        self.num_subclones = num_subclones
        self.parents, self.ccfs = self.make_phylogeny()

    def post_order_iter(self) -> Iterator[int]:
        """
        Returns the postorder iterator over the nodes of the tree.
        A parent node is guaranteed to be iterated over after all the nodes in the subtree rooted by it."""
        # Topologically sorting the nodes.
        rem_children: Dict[int | None, int] = {node: 0 for node in self.parents}
        rem_children[None] = self.num_subclones + 1_000_000

        for parent in self.parents.values():
            if parent is not None:
                rem_children[parent] += 1

        stack = [n for n, v in rem_children.items() if n is not None and v == 0]
        i = 0
        while i < len(stack):
            child = stack[i]
            par = self.parents[child]
            rem_children[par] -= 1
            if rem_children[par] == 0 and par is not None:
                stack.append(par)
            i += 1
            yield child

    def pre_order_iter(self) -> Iterator[int]:
        """
        Returns the preorder iterator over the nodes of the tree.
        A parent node is guaranteed to be iterated over before any node in the subtree rooted by it."""
        # Topologically sorting the nodes.
        rem_children: Dict[int | None, int] = {node: 0 for node in self.parents}
        rem_children[None] = self.num_subclones + 1_000_000

        for parent in self.parents.values():
            if parent is not None:
                rem_children[parent] += 1

        stack = [n for n, v in rem_children.items() if n is not None and v == 0]
        i = 0
        while i < len(stack):
            child = stack[i]
            par = self.parents[child]
            rem_children[par] -= 1
            if rem_children[par] == 0 and par is not None:
                stack.append(par)
            i += 1
        stack = stack[::-1]
        return iter(stack)

    def make_weights(self) -> Dict[int, float]:
        """
        Samples random weights for the Phylogeny objects.
        """

        # The weight of each node is a uniformly distributed variable.
        # Not the most accurate of assumptions.
        weights = {}
        for p in self.pre_order_iter():
            weights[p] = random.random()
        weights[1] = 1

        tot = sum(weights.values())
        for i in weights:
            weights[i] /= tot

        return weights

    def make_phylogeny(self):
        """Greedy algorithm to assign children clones in correct phylogeny based on random CCFs

        :return: (dict, dict) representing the parent and ccfs dictionaries"""
        ccfs = sorted(np.random.rand(self.num_subclones), reverse=True)
        ccfs = {cluster + 2: ccf for cluster, ccf in enumerate(ccfs)}
        parent_dict: Dict[int, Optional[int]] = {1: 0, 0: None}

        unassigned = deque(list(ccfs.keys()))
        parent_queue = deque([1])
        ccfs[0] = 1
        ccfs[1] = 1
        while len(unassigned) > 0:
            parent = parent_queue.popleft()
            ccf_remaining = ccfs[parent]
            for c in unassigned.copy():
                if ccfs[c] <= ccf_remaining:
                    this_cluster = c
                    unassigned.remove(c)
                    parent_queue.append(this_cluster)
                    parent_dict[this_cluster] = parent
                    ccf_remaining -= ccfs[this_cluster]

        return parent_dict, ccfs

    def get_lineage(self, node: Optional[int]) -> List[int]:
        """
        Returns the lineage for the specified clone.
        The lineage contains the clone.

        :param node: index of desired clone
        :return: list[int] representing the clones in the lineage"""
        cluster_list = []

        while node is not None:
            cluster_list.append(node)
            node = self.parents[node]

        return cluster_list

    def __repr__(self) -> str:
        return f"Phylogeny({self.ccfs=}, {self.parents=})"


def simulate_coverage_and_depth(
    cnv_pickle: io.BufferedIOBase,
    coverage_file: PathLike | pd.DataFrame,
    vcf_file: PathLike | pd.DataFrame,
    read_depths: PathLike | pd.DataFrame,
    purity: float,
    output_coverage_fn: PathLike,
    output_hets_fn: PathLike,
    normal_coverage=None,
    normal_depths=None,
    do_parallel=True,
):
    cnv_object: CNV_Profile = pickle.load(cnv_pickle)
    cnv_object.save_coverage_file(
        output_coverage_fn, purity, coverage_file, do_parallel=do_parallel
    )
    cnv_object.save_hets_file(
        output_hets_fn, vcf_file, read_depths, purity, do_parallel=do_parallel
    )

    if normal_coverage and normal_depths:
        cov_fn = os.path.splitext(output_coverage_fn)
        hets_fn = os.path.splitext(output_hets_fn)
        cnv_object.save_coverage_file(
            f"{cov_fn[0]}_normal{cov_fn[1]}", 0, normal_coverage
        )
        cnv_object.save_hets_file(
            f"{hets_fn[0]}_normal{hets_fn[1]}", vcf_file, normal_depths, 0
        )
    else:
        print("No Normal CNV profile calculated")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Get coverage and VCF outputs based on a simulated CNV profile"
    )
    parser.add_argument(
        "cnv_pickle", help="CNV_Profile object as a pickle file", type=Path
    )
    parser.add_argument(
        "coverage_file",
        help="Bed file giving coverage over bins or intervals",
        type=Path,
    )
    parser.add_argument(
        "vcf_file", help="VCF file for this (simulated) participant", type=Path
    )
    parser.add_argument(
        "read_depths", help="Depths at SNPs given in VCF file, as a bed file"
    )
    parser.add_argument("purity", help="Desired tumor purity")
    parser.add_argument(
        "-oc", "--output_coverage", default="./simulated_coverage.txt"
    )
    parser.add_argument("-oh", "--output_hets", default="./simulated_hets.txt")
    parser.add_argument(
        "--normal_coverage",
        help="Bed file with bin/interval coverage for Normal Sample",
    )
    parser.add_argument(
        "--normal_depths",
        help="Depths at SNPs for normal sample, as a bed file",
    )

    args = parser.parse_args()

    simulate_coverage_and_depth(
        open(args.cnv_pickle, "rb"),
        args.coverage_file,
        args.vcf_file,
        args.read_depths,
        args.purity,
        args.output_coverage,
        args.output_hets,
        args.normal_coverage,
        args.normal_depths,
    )


if __name__ == "__main__":
    main()

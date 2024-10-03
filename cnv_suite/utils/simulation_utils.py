import enum
import re
from typing import Any, Dict, overload
import numpy as np
import pandas as pd

from utils import PathLike


class Haplotype(enum.Enum):
    MATERNAL = enum.auto()
    PATERNAL = enum.auto()


def get_alt_count(m_prop, p_prop, m_present, p_present, coverage, correct_phase):
    """Returns number of alternate reads generated from a binomial.

    If both alleles have mutation (homozygous mutation), returns the given coverage. If neither allele has mutation,
    returns 0. Also adjusts for phasing by the boolean correct_phase."""
    if not correct_phase:
        m_prop, p_prop = p_prop, m_prop
        m_present, p_present = p_present, m_present

    if m_present and p_present:
        return coverage  # add noise? as in: np.random.binomial(coverage, 0.999)
    elif not m_present and not p_present:
        return 0  # noise?
    elif m_present:
        return np.random.binomial(coverage, m_prop)
    else:
        return np.random.binomial(coverage, p_prop)


def get_average_ploidy(pat_ploidy: float, mat_ploidy: float, purity: float) -> float:
    """Get the average ploidy defined by the paternal/maternal tumor CN and the purity."""
    return (pat_ploidy + mat_ploidy) * purity + 2 * (1 - purity)


def single_allele_ploidy(allele, start, end):
    """Get the ploidy for this allele tree over this interval [start, end)."""
    intervals = allele.envelop(start, end) | allele[start] | allele[end - 1]
    if len(intervals) == 1:
        return intervals.pop().data.cn_change
    else:
        interval_totals = [(min(i.end, end) - max(i.begin, start)) * i.data.cn_change for i in intervals]
        return sum(interval_totals) / (end - start)


def get_contigs_from_header(vcf_fn):
    contig_dict = {}
    with open(vcf_fn, "r") as vcf:
        pre_contig = True
        post_contig = False
        while not post_contig:
            line = vcf.readline()
            re_groups: re.Match[str] = re.search(r"##(?P<id>\w+)=(?P<value>.*)", line)  # type: ignore
            if re_groups.group("id") == "contig":
                pre_contig = False
                contig_groups = re.search(r"<ID=(?P<name>[chrXY\d]+),length=(?P<len>\d+)>", re_groups.group("value"))  # type: ignore
                contig_dict[contig_groups.group("name")] = int(contig_groups.group("len"))  # type: ignore
            elif not pre_contig:
                post_contig = True

    return contig_dict


@overload
def switch_contigs(input_data: pd.DataFrame) -> pd.DataFrame: ...


@overload
def switch_contigs(input_data: Dict[str, Any]) -> Dict[str, Any]: ...


def switch_contigs(input_data: pd.DataFrame | Dict[str, Any]):
    """Return the input data with 'chr' removed from contigs and X/Y changed to 23/24.

    :param input_data: dict or pd.DataFrame with contig as keys or column
    :returns: dict or pd.DataFrame with altered contig names"""
    if isinstance(input_data, pd.DataFrame):
        contig_column_names = ["Chr", "Chromosome", "Chrom", "Contig"]  # defines possible column names
        # accounts for all lower/upper-case
        contig_column_names = (
            contig_column_names + [s.lower() for s in contig_column_names] + [s.upper() for s in contig_column_names]
        )
        contig_column_names = contig_column_names + [s + "s" for s in contig_column_names]  # pluralizes column names
        column_idx = np.where([c in contig_column_names for c in input_data.columns])[0][0]  # find contig column
        column_label = input_data.columns[column_idx]

        input_data[column_label] = input_data[column_label].apply(
            lambda x: re.search(r"(?<=chr)[\dXY]+|^[\dXY]+", x).group()  # type: ignore
        )
        input_data.replace(to_replace={column_label: {"X": "23", "Y": "24"}}, inplace=True)
        # should already be sorted
        # input_data.sort_values([column_label, 'start'], key=natsort.natsort_keygen(), inplace=True)
        return input_data
    elif isinstance(input_data, dict):
        input_data = {re.search(r"(?<=chr)[\dXY]+|^[\dXY]+", key).group(): loc for key, loc in input_data.items()}  # type: ignore
        if "X" in input_data.keys():
            input_data["23"] = input_data["X"]
            input_data.pop("X")
        if "Y" in input_data.keys():
            input_data["24"] = input_data["Y"]
            input_data.pop("Y")
        return input_data
    else:
        raise ValueError(f"Only dictionaries and pandas DataFrames supported. Not {type(input_data)}.")


def dump_tsv(df: pd.DataFrame, path: PathLike, header=True):
    """
    Dumps the given dataframe as a TSV file.
    """
    df.to_csv(path, sep="\t", index=False, header=header)

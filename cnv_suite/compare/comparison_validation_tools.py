import os
from typing import Optional, Union
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import math
import plotly.express as px

from ..utils import get_segment_interval_trees


def num_segments(file_name: Union[str, os.PathLike, None] = None, seg_df: Optional[pd.DataFrame] = None):
    assert file_name or seg_df is not None

    if file_name is not None:
        return sum(1 for i in open(file_name, "rb")) - 1
    else:
        assert seg_df is not None
        return seg_df.shape[0]


def compare_length_distribution(file_1=None, file_2=None, seg_df_1=None, seg_df_2=None, sample_names=None):
    assert (file_1 or seg_df_1 is not None) and (file_2 or seg_df_2 is not None)

    if file_1:
        seg_df_1 = pd.read_csv(file_1, sep="\t")
    if file_2:
        seg_df_2 = pd.read_csv(file_2, sep="\t")
    assert seg_df_1 is not None
    assert seg_df_2 is not None
    if sample_names is None:
        sample_names = ["Profile 1", "Profile 2"]

    seg_df_1["length"] = seg_df_1["End.bp"] - seg_df_1["Start.bp"]
    seg_df_2["length"] = seg_df_2["End.bp"] - seg_df_2["Start.bp"]

    mannwhitney = ss.mannwhitneyu(seg_df_1["length"], seg_df_2["length"])

    fig, ax = plt.subplots(1, 2)
    round_to = 10**7
    bins = np.linspace(
        0,
        round_to
        * math.ceil((np.max(np.concatenate([seg_df_1["length"].values, seg_df_2["length"].values]))) / round_to),
        50,
    )
    ax[0].hist(seg_df_1["length"], bins, alpha=0.5, label="x", density=True)
    ax[0].hist(seg_df_2["length"], bins, alpha=0.5, label="y", density=True)
    ax[0].legend(sample_names, loc="upper right")
    ax[0].set_xlabel("Segment Length (bp)")
    ax[0].set_ylabel("Density of Counts")
    ax[0].set_title("Segment Lengths", fontsize=10)

    # add plot that shows segment length, normalized to chromosome (or arm?)
    chr_size = pd.concat(
        [seg_df_1.groupby("Chromosome")["End.bp"].max(), seg_df_2.groupby("Chromosome")["End.bp"].max()], axis=1
    )
    chr_size = chr_size.max(axis=1).to_dict()
    seg_df_1["length_normalized"] = seg_df_1.apply(lambda x: x["length"] / chr_size[x["Chromosome"]], axis=1)
    seg_df_2["length_normalized"] = seg_df_2.apply(lambda x: x["length"] / chr_size[x["Chromosome"]], axis=1)
    bins = np.linspace(0, 1, 50)
    ax[1].hist(seg_df_1["length_normalized"], bins, alpha=0.5, label="x", density=True)
    ax[1].hist(seg_df_2["length_normalized"], bins, alpha=0.5, label="y", density=True)
    ax[1].legend(["First profile", "Second profile"], loc="upper right")
    ax[1].set_xlabel("Normalized Segment Length")
    ax[1].set_title("Lengths Normalized by Contig Length", fontsize=10)

    fig.suptitle("Comparing Segment Length Distributions", fontsize=16)

    return mannwhitney.pvalue, fig


def breakpoint_distance(
    file_control: Union[str, os.PathLike, None] = None,
    file_case: Union[str, os.PathLike, None] = None,
    seg_df_control: Optional[pd.DataFrame] = None,
    seg_df_case: Optional[pd.DataFrame] = None,
):
    """Plots the distance to the nearest breakpoints with the difference in mu's for two profiles.

    Plots combined distance to the breakpoints left and right of each intersection segment (after merging the two profiles) and assigns to one of four categories based on size and placement of original segments (relative to merged segment)

    :param file_control: filename and path for control seg file, optional (must give either filename of DataFrame)
    :param file_case: filename and path for case seg file, optional (must give either filename of DataFrame)
    :param seg_df_control: pandas.DataFrame for control seg file, optional (must give either filename of DataFrame)
    :param seg_df_case: pandas.DataFrame for control seg file, optional (must give either filename of DataFrame)
    :return: (plotly Figure, pandas.Series with breakpoint category counts)
    """
    assert (file_control or seg_df_control is not None) and (file_case or seg_df_case is not None)

    if file_control:
        seg_df_control = pd.read_csv(file_control, sep="\t")
    if file_case:
        seg_df_case = pd.read_csv(file_case, sep="\t")

    assert seg_df_case is not None
    assert seg_df_control is not None

    if "Sample_ID" not in seg_df_control:
        seg_df_control["Sample_ID"] = "control_profile"
    if "Sample_ID" not in seg_df_case:
        seg_df_case["Sample_ID"] = "case_profile"

    seg_df_control.dropna(subset=["mu.minor", "mu.major"], inplace=True)
    seg_df_case.dropna(subset=["mu.minor", "mu.major"], inplace=True)

    seg_df_control = seg_df_control.reset_index(drop=True)
    seg_df_case = seg_df_case.reset_index(drop=True)

    seg_df_control["start"], seg_df_control["end"] = seg_df_control["Start.bp"], seg_df_control["End.bp"]
    seg_df_case["start"], seg_df_case["end"] = seg_df_case["Start.bp"], seg_df_case["End.bp"]

    contig_trees = get_segment_interval_trees(pd.concat([seg_df_control, seg_df_case]))
    diff_df = pd.concat(
        [
            get_differences_from_intervals(
                tree,
                i + 1,
                sample_names=[seg_df_control.loc[0, "Sample_ID"], seg_df_case.loc[0, "Sample_ID"]],
                breakpoint_data=True,
            )
            for i, tree in enumerate(contig_trees)
        ]
    )
    diff_df_stacked = (
        diff_df.set_index(["chrom", "start", "end", "bp_size", "bp_type"])[["mu_major_diff", "mu_minor_diff"]]
        .stack()
        .reset_index(name="mu_diff")
        .rename(columns={"level_5": "homolog"})
    )

    fig = px.scatter(
        diff_df_stacked,
        x="bp_size",
        y="mu_diff",
        color="homolog",
        facet_col="bp_type",
        hover_data=["chrom", "start", "end"],
    )

    return fig, diff_df.groupby("bp_type")["bp_size"].describe()


def mu_sigma_difference(file_1=None, file_2=None, seg_df_1=None, seg_df_2=None, mu_lim=None, sigma_lim=None):
    assert (file_1 or seg_df_1 is not None) and (file_2 or seg_df_2 is not None)

    if file_1:
        seg_df_1 = pd.read_csv(file_1, sep="\t")
    if file_2:
        seg_df_2 = pd.read_csv(file_2, sep="\t")

    assert seg_df_1 is not None
    assert seg_df_2 is not None

    if "Sample_ID" not in seg_df_1:
        seg_df_1["Sample_ID"] = "profile_1"
    if "Sample_ID" not in seg_df_2:
        seg_df_2["Sample_ID"] = "profile_2"

    # should I take the intersection of the segments??
    # I think so, because the larger segment intersections will be prioritized (in terms of their lengths)
    contig_trees = get_segment_interval_trees(pd.concat([seg_df_1, seg_df_2]))

    # get differences, with non-overlapping segments removed (this can be visualized better using acr_compare tool)
    diff_df = pd.concat(
        [
            get_differences_from_intervals(
                tree, i + 1, sample_names=[seg_df_1.loc[0, "Sample_ID"], seg_df_2.loc[0, "Sample_ID"]]
            )
            for i, tree in enumerate(contig_trees)
        ]
    )

    fig, ax = plt.subplots(1, 1)
    # code taken from Claudia's cgaprojects_ibm_tAML_analysis
    ax.axvline(0, c="k", linewidth=0.5)
    ax.axhline(0, c="k", linewidth=0.5)

    sigma_diffs = np.concatenate([diff_df["sigma_minor_diff"], diff_df["sigma_major_diff"]])
    mu_diffs = np.concatenate([diff_df["mu_minor_diff"], diff_df["mu_major_diff"]])
    lengths = np.concatenate([diff_df["length"], diff_df["length"]])
    pcm = ax.scatter(sigma_diffs, mu_diffs, c=["blue"] * diff_df.shape[0] + ["red"] * diff_df.shape[0], alpha=0.2)

    max_length = np.nanmax(lengths)
    pcm.set_sizes(lengths / max_length * 100 + 3)

    max_mu_diff = np.nanmax(np.abs(mu_diffs))
    max_sigma_diff = np.nanmax(np.abs(sigma_diffs))

    if mu_lim is not None and sigma_lim is not None:
        ax.set_ylim(-mu_lim, mu_lim)
        ax.set_xlim(-sigma_lim, sigma_lim)
    else:
        ax.set_ylim(-max_mu_diff, max_mu_diff)
        ax.set_xlim(-max_sigma_diff if np.amin(sigma_diffs) < 0 else 0, max_sigma_diff)
    ax.set_ylabel("ACR (mu) Difference")
    ax.set_xlabel("Variance (sigma) Difference")
    ax.set_title("Comparison of CNV Profiles")

    # legend
    plt.legend(*pcm.legend_elements("sizes", num=5, func=lambda x: (x - 3) * max_length / 100, fmt="{x:.1e}bp"), loc=3)
    fig.show()

    return fig, ax


def get_differences_from_intervals(contig_tree, contig_name, sample_names=None, breakpoint_data=False):
    if sample_names is None:
        sample_names = ["profile_1", "profile_2"]
    length_arr = []
    data = []

    for interval in contig_tree:
        if len(interval.data) == 1:
            continue
        elif len(interval.data) == 2:
            mu_minor_diff = interval.data[sample_names[1]].mu_minor - interval.data[sample_names[0]].mu_minor
            mu_major_diff = interval.data[sample_names[1]].mu_major - interval.data[sample_names[0]].mu_major
            sigma_minor_diff = interval.data[sample_names[1]].sigma_minor - interval.data[sample_names[0]].sigma_minor
            sigma_major_diff = interval.data[sample_names[1]].sigma_major - interval.data[sample_names[0]].sigma_major
            interval_data = [
                contig_name,
                interval.begin,
                interval.end,
                mu_minor_diff,
                mu_major_diff,
                sigma_minor_diff,
                sigma_major_diff,
            ]

            if breakpoint_data:
                start_dist_2 = interval.begin - interval.data[sample_names[1]].start
                start_dist_1 = interval.begin - interval.data[sample_names[0]].start
                end_dist_2 = interval.data[sample_names[1]].end - interval.end
                end_dist_1 = interval.data[sample_names[0]].end - interval.end
                bp_size = sum([start_dist_1, start_dist_2, end_dist_1, end_dist_2])

                if start_dist_1 == 0 and end_dist_1 == 0:
                    bp_type = "smaller_1"
                elif start_dist_2 == 0 and end_dist_2 == 0:
                    bp_type = "smaller_2"
                elif start_dist_2 == 0 and end_dist_1 == 0:
                    bp_type = "2_shifted_right"
                elif start_dist_1 == 0 and end_dist_2 == 0:
                    bp_type = "2_shifted_left"
                elif start_dist_1 == start_dist_2 == end_dist_1 == end_dist_2 == 0:
                    bp_type = "perfect_match"
                else:
                    bp_type = "unknown"
                interval_data = interval_data + [bp_type, bp_size]
        else:
            raise ValueError(f"Interval should have one or two samples, not {len(interval.data)}")

        interval_data.append(interval.end - interval.begin)
        data.append(interval_data)

    columns = ["chrom", "start", "end", "mu_minor_diff", "mu_major_diff", "sigma_minor_diff", "sigma_major_diff"]
    if breakpoint_data:
        columns = columns + ["bp_type", "bp_size"]

    return pd.DataFrame(data, columns=columns + ["length"])

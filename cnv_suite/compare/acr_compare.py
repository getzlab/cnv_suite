from collections import defaultdict
from dataclasses import dataclass
import glob
import os
import re
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy


try:
    from statistics import NormalDist
except (ModuleNotFoundError, ImportError):
    from scipy.stats import norm as NormalDist
from math import log
import sys
from scipy.optimize import minimize
from utils import BASE_PATH, PathLike
import random

STAT_COLUMNS = ["mu.minor", "sigma.minor", "mu.major", "sigma.major"]


def acr_compare(
    file_1: pd.DataFrame | PathLike | None = None,
    file_2: pd.DataFrame | PathLike | None = None,
    fit_params=False,
):
    """
    Compare the allelic copy ratio segments between the two seg files.

    :param file_1: name and path of file_1 (default to user input)
    :param file_2: name and path of file_2 (default to user input)
    :return: weighted average of all compared ACR segment normal distributions
    """

    # Getting the inputs.
    # If the inputs are None, we read a path from user input.
    # If they are paths, we read the CSV from the path.
    # Otherwise, they are Dataframes, and we can just use them.

    if file_1 is None:
        file_1 = input("Input first file name: ")
    if isinstance(file_1, PathLike):
        seg1 = pd.read_csv(file_1, sep="\t")
    else:
        seg1 = file_1

    if file_2 is None:
        file_2 = input("Input second file name: ")
    if isinstance(file_2, PathLike):
        seg2 = pd.read_csv(file_2, sep="\t")
    else:
        seg2 = file_2

    # format dataframes (throwing out rows with nan values for mu/sigma)
    seg1 = seg1.dropna(subset=STAT_COLUMNS).reset_index(drop=True)
    try:
        seg2 = seg2.dropna(subset=STAT_COLUMNS).reset_index(drop=True)
    except:
        print(list(seg2), STAT_COLUMNS)
        raise

    # take union of segments to allocate bins
    bins = get_union(seg1, seg2)

    if fit_params:
        minimization_result = minimize(
            overlap_min_helper,
            x0=np.array([1, 0.5]),
            args=bins,
            method="Powell",
            bounds=[(0.00001, 2000), (0, 1)],
            options={"xtol": 0.000001, "ftol": 0.00001},
        )

        optimal_scale_factor = minimization_result.x[0]
        optimal_purity = minimization_result.x[1]
        params = minimization_result.x
    else:
        params = None
        optimal_scale_factor, optimal_purity = 0, 0
    overlap_score, maj_ov, min_ov = get_avg_overlap(bins, params)
    bins["major_overlap"] = maj_ov
    bins["minor_overlap"] = min_ov

    if fit_params:
        bins["mu.major_1"] = bins.apply(
            lambda x: optimal_scale_factor
            * (optimal_purity * (x["mu.major_1"] - 1) + 1),
            axis=1,
        )
        bins["mu.minor_1"] = bins.apply(
            lambda x: optimal_scale_factor
            * (optimal_purity * (x["mu.minor_1"] - 1) + 1),
            axis=1,
        )
        bins["sigma.major_1"] *= optimal_scale_factor
        bins["sigma.minor_1"] *= optimal_scale_factor

    non_overlap_length = int(bins["length_1_unique"].sum()) + int(
        bins["length_2_unique"].sum()
    )
    overlap_length = int(bins["length_overlap"].sum())

    return (
        overlap_score,
        optimal_scale_factor,
        optimal_purity,
        non_overlap_length,
        overlap_length,
        bins,
    )


def overlap_min_helper(params, bins):
    """
    Return just the inverse overlap score for optimization minimize function.
    """
    overlap_score, _, _ = get_avg_overlap(bins, params)

    return 1 / overlap_score


def get_avg_overlap(bins, params=None):
    """
    Calculate the overlap score for the major and minor alleles and get the average.

    :param ratio: ratio between copy number profile 1 and 2
    :param bins: bins dataframe
    :return: overlap score, major overlap score series, and minor overlap score series
    """
    # call calc_overlap on each "bin", for both alleles
    # can I assume allele marker (major vs. minor) is same for both inputs? todo
    major_overlap = bins.apply(
        lambda x: calc_overlap(
            x["mu.major_1"],
            x["sigma.major_1"],
            x["mu.major_2"],
            x["sigma.major_2"],
            x["unique"],
            params=params,
        ),
        axis=1,
    )
    minor_overlap = bins.apply(
        lambda x: calc_overlap(
            x["mu.minor_1"],
            x["sigma.minor_1"],
            x["mu.minor_2"],
            x["sigma.minor_2"],
            x["unique"],
            params=params,
        ),
        axis=1,
    )

    # get average overlap score
    overlap_score = np.average(
        np.concatenate((major_overlap, minor_overlap)),
        weights=np.concatenate([bins["length"], bins["length"]]),
    )
    return overlap_score, major_overlap, minor_overlap


def calc_overlap(
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
    unique: bool,
    params: np.ndarray | None = np.array([1.0, 1.0]),
):
    """
    Calculate the overlap between two normal distributions, defined by given statistics.


    :param mu1: mean of distribution 1
    :param sigma1: std dev of distribution 1
    :param mu2: mean of distribution 2
    :param sigma2: std dev of distribution 2
    :param unique: boolean, if bin represents a non-overlapping, unique segment
    :param ratio: ratio between first and second distribution
    :return: overlap between two distributions as value -> [0, 1]
    """
    # check if bin is non-overlapping
    if unique:
        return 0
    if params is None:
        params = np.array(
            [
                1.0,
                1,
            ]
        )
    scale_factor, purity = params
    mu1 = scale_factor * (purity * (mu1 - 1) + 1)
    sigma1 *= scale_factor

    # check if distributions are equal
    if mu1 == mu2 and sigma1 == sigma2:
        return 1

    # run builtin method in statistics.NormalDist
    # returns OVL; should I give option for Bhattacharyya? todo
    if "statistics" in sys.modules:
        try:
            overlap = NormalDist(mu1, sigma1).overlap(NormalDist(mu2, sigma2))  # type: ignore
        except AttributeError:
            pass
        else:
            return overlap

    # calculate intersection(s) of the two distributions
    x_intersect = calc_pdf_intersect(mu1, sigma1, mu2, sigma2)

    if len(x_intersect) == 1:  # sigma1 == sigma2
        area = NormalDist(mu1, sigma1).cdf(x_intersect[0])
        if area < 0.5:
            return (
                area * 2
            )  # doubled -> pdf1_area = pdf2_area when sigma1 = sigma2
        else:  # take other side of cdf
            return (1 - area) * 2

    # calculate overlap cdf
    mid_section1 = NormalDist(mu1, sigma1).cdf(max(x_intersect)) - NormalDist(
        mu1, sigma1
    ).cdf(min(x_intersect))
    mid_section2 = NormalDist(mu2, sigma2).cdf(max(x_intersect)) - NormalDist(
        mu2, sigma2
    ).cdf(min(x_intersect))

    # compute sum of overlap sections based on which middle section is larger
    if mid_section1 < mid_section2:
        sum_overlap = 1 + mid_section1 - mid_section2
    else:
        sum_overlap = 1 + mid_section2 - mid_section1

    return sum_overlap


def calc_pdf_intersect(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> npt.NDArray[np.complexfloating] | npt.NDArray[np.floating]:
    """
    Calculate intersection(s) of two normal distributions.

    Returns one value (in a list) if sigmas are equal; otherwise returns two values.

    :param mu1: mean of distribution 1
    :param sigma1: std dev of distribution 1
    :param mu2: mean of distribution 2
    :param sigma2: std dev of distribution 2
    :return: list of intersection values
    """
    if sigma1 == sigma2:
        return np.array([(mu1 + mu2) / 2])

    # calculate roots of log[pdf_1] = log[pdf_2]
    a = 0.5 * (1 / sigma2**2 - 1 / sigma1**2)
    b = mu1 / sigma1**2 - mu2 / sigma2**2
    c = 0.5 * (mu2**2 / sigma2**2 - mu1**2 / sigma1**2) + log(sigma2 / sigma1)

    return np.roots([a, b, c])


####################


def get_union(seg1_df: pd.DataFrame, seg2_df: pd.DataFrame) -> pd.DataFrame:
    full_bins = _union_one_sided(seg1_df, seg2_df)
    bins_2 = _union_one_sided(seg2_df, seg1_df)
    # take only unique values from bins_2
    unique_2_df = bins_2.loc[bins_2["unique"] == True]

    # swap stats columns (and 1/2 length columns)
    column_list = []
    for c in unique_2_df.columns.values:
        if "_1" in c:
            c = c.replace("1", "2")
        else:
            c = c.replace("2", "1")
        column_list.append(c)
    unique_2_df.columns = column_list

    bins = pd.concat([full_bins, unique_2_df], ignore_index=True)

    return bins


def _union_one_sided(
    seg1_df: pd.DataFrame, seg2_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Gets the union of the segment partitions between the two files.

    :param seg1_df: Dataframe of the first seg file
    :param seg2_df: Dataframe of the second seg file
    :return: union of two seg files as a Bin dataframe
    """

    bin_list = []
    for chrom in np.arange(
        1, 23
    ):  # assumes segments in each chromosome (except for XY)
        segments1 = seg1_df.loc[seg1_df["Chromosome"] == chrom].reset_index(
            drop=True
        )
        segments2 = seg2_df.loc[seg2_df["Chromosome"] == chrom].reset_index(
            drop=True
        )

        pointer2 = 0  # starting at beginning of seg2
        for i, (start, end) in enumerate(
            zip(segments1["Start.bp"], segments1["End.bp"])
        ):
            # create bins for this segment1, updating pointer2 to save computation
            bin_list, pointer2 = create_bins(
                start,
                end,
                segments2,
                pointer2,
                segments1.loc[i][STAT_COLUMNS],
                chrom,
                bin_list=bin_list,
            )

    bin_df = pd.DataFrame(bin_list)
    return bin_df


def create_bins(
    start, end, segments2, pointer2, segments1_stats, chrom, bin_list=None
):
    """
    Creates bins over this segment defined by start to end, as compared to segments2 df.

    Adds a bin for each overlap section and for all sections that are unique to segment1 (as defined by start, end).
    Unique segments2 sections must be found by reversing the inputs.

    :param start: starting base pair (of seg1)
    :param end: ending base pair (of seg1)
    :param segments2: dataframe for seg2
    :param pointer2: current index of seg2 to save computation
    :param segments1_stats: statistics of seg1 to copy to Bin
    :param chrom: this chromosome
    :param bin_list: list of bins, for recursion
    :return: final list of Bin dicts
    """
    if bin_list is None:
        bin_list = []

    # exit statement for end of segment2
    if pointer2 >= len(segments2):
        bin_list.append(append_bin(start, end, segments1_stats, None, chrom))
        return bin_list, pointer2

    start2 = segments2.loc[pointer2]["Start.bp"]
    end2 = segments2.loc[pointer2]["End.bp"]

    if start < start2:
        if end <= start2:
            bin_list.append(
                append_bin(start, end, segments1_stats, None, chrom)
            )  # unique segment
            return bin_list, pointer2
        else:
            bin_list.append(
                append_bin(start, start2, segments1_stats, None, chrom)
            )  # unique segment

            if end < end2:
                bin_list.append(
                    append_bin(
                        start2,
                        end,
                        segments1_stats,
                        segments2.loc[pointer2][STAT_COLUMNS],
                        chrom,
                    )
                )
                return bin_list, pointer2
            else:
                bin_list.append(
                    append_bin(
                        start2,
                        end2,
                        segments1_stats,
                        segments2.loc[pointer2][STAT_COLUMNS],
                        chrom,
                    )
                )
                return create_bins(
                    end2,
                    end,
                    segments2,
                    pointer2 + 1,
                    segments1_stats,
                    chrom,
                    bin_list=bin_list,
                )
    else:
        if end <= end2:
            bin_list.append(
                append_bin(
                    start,
                    end,
                    segments1_stats,
                    segments2.loc[pointer2][STAT_COLUMNS],
                    chrom,
                )
            )
            return bin_list, pointer2
        else:
            if start < end2:
                bin_list.append(
                    append_bin(
                        start,
                        end2,
                        segments1_stats,
                        segments2.loc[pointer2][STAT_COLUMNS],
                        chrom,
                    )
                )
                return create_bins(
                    end2,
                    end,
                    segments2,
                    pointer2 + 1,
                    segments1_stats,
                    chrom,
                    bin_list=bin_list,
                )
            else:
                return create_bins(
                    start,
                    end,
                    segments2,
                    pointer2 + 1,
                    segments1_stats,
                    chrom,
                    bin_list=bin_list,
                )


def append_bin(start, end, stats1, stats2, chromosome):
    unique = stats1 is None or stats2 is None
    length = end - start
    length_overlap = length_1_unique = length_2_unique = 0

    if not unique:
        length_overlap = length
    elif stats2 is None:
        length_1_unique = length
    elif stats1 is None:
        length_2_unique = length

    bin_dict = {
        "chromosome": chromosome,
        "Start.bp": start,
        "End.bp": end,
        "unique": unique,
        "length": length,
        "length_overlap": length_overlap,
        "length_1_unique": length_1_unique,
        "length_2_unique": length_2_unique,
    }

    for key in STAT_COLUMNS:  # todo keep in multiindex
        if stats1 is not None:
            val = stats1[key]
        else:
            val = None
        bin_dict[f"{key}_1"] = val

        if stats2 is not None:
            val = stats2[key]
        else:
            val = None
        bin_dict[f"{key}_2"] = val

    return bin_dict


@dataclass(frozen=True, order=True)
class Segment:
    start: int
    end: int
    minor: float
    major: float


def compute_aad(target: pd.DataFrame, pred: pd.DataFrame) -> float:
    """
    Computes the AAD as described in the HapASeg paper:
    The naive AAD is the weighted average of the absolute difference between the copy number,
    and the AAD is the minimum over all linear transformations of `pred` of the AAD.
    """

    weights = []
    src = []
    tar = []

    tar_chr = list(target).index("Chromosome")
    tar_start = list(target).index("Start.bp")
    tar_end = list(target).index("End.bp")
    tar_minor = list(target).index("mu.minor")
    tar_major = list(target).index("mu.major")

    src_chr = list(pred).index("Chromosome")
    src_start = list(pred).index("Start.bp")
    src_end = list(pred).index("End.bp")
    src_minor = list(pred).index("mu.minor")
    src_major = list(pred).index("mu.major")

    src_by_chr: defaultdict[int, list[Segment]] = defaultdict(list)
    tar_by_chr: defaultdict[int, list[Segment]] = defaultdict(list)

    for row in pred.to_numpy():
        src_by_chr[row[src_chr]].append(
            Segment(
                row[src_start], row[src_end], row[src_minor], row[src_major]
            )
        )
    for row in target.to_numpy():
        tar_by_chr[row[tar_chr]].append(
            Segment(
                row[tar_start], row[tar_end], row[tar_minor], row[tar_major]
            )
        )

    for chr in src_by_chr.keys() & tar_by_chr.keys():
        src_segments = src_by_chr[chr]
        tar_segments = tar_by_chr[chr]
        src_segments.sort()
        tar_segments.sort()

        src_idx = 0
        tar_idx = 0

        while src_idx < len(src_segments) and tar_idx < len(tar_segments):
            if src_segments[src_idx].end < tar_segments[tar_idx].start:
                src_idx += 1
                continue
            if tar_segments[tar_idx].end < src_segments[src_idx].start:
                tar_idx += 1
                continue
            # There is an intersection!
            start = max(
                src_segments[src_idx].start, tar_segments[tar_idx].start
            )
            end = min(src_segments[src_idx].end, tar_segments[tar_idx].end)
            weights.append((end - start))
            src.append(src_segments[src_idx].minor)
            tar.append(tar_segments[tar_idx].minor)
            weights.append((end - start))
            src.append(src_segments[src_idx].major)
            tar.append(tar_segments[tar_idx].major)

            if src_segments[src_idx].end < tar_segments[tar_idx].end:
                src_idx += 1
            else:
                tar_idx += 1

    weights = np.array(weights)
    weights /= np.sum(weights)
    src = np.array(src)
    tar = np.array(tar)

    params = minimize(
        lambda p: np.sum(weights * np.abs(src * p[0] + p[1] - tar)),
        np.array([1, 0]),
    ).x
    return float(np.sum(weights * np.abs(src * params[0] + params[1] - tar)))


def compute_output_aad(
    src_folder: PathLike, solution_folder: PathLike
) -> dict[float, list[float]]:
    """
    Computes the AAD of all solved instances.
    """
    assert os.path.exists(src_folder)
    assert os.path.exists(solution_folder)
    print(src_folder, solution_folder)

    simulated_problems = glob.glob(
        f"{src_folder}/**/sim_profile.tsv", recursive=True
    )

    res: dict[float, list[float]] = {}

    pur_re = re.compile(r"pur_(\d+)")

    for problem_path in simulated_problems:
        rel_path = "\\".join(
            problem_path.replace(str(src_folder), "").split("\\")[:-1]
        )
        solution_path = f"{solution_folder}{rel_path}\\ploidities.tsv"
        if not os.path.exists(solution_path):
            continue
        print(solution_path)

        pred_df = pd.read_csv(problem_path, sep="\t")
        sol_df = pd.read_csv(solution_path, sep="\t")

        aad = compute_aad(sol_df, pred_df)
        pur_str = re.findall(pur_re, problem_path)[0]
        pur = int(pur_str) * 10 ** (1 - len(pur_str))

        res.setdefault(pur, []).append(aad)
    print(res)
    return res


def compute_output_purity(
    src_folder: PathLike, solution_folder: PathLike
) -> dict[float, list[float]]:
    """
    Computes the AAD of all solved instances.
    """
    assert os.path.exists(src_folder)
    assert os.path.exists(solution_folder)
    print(src_folder, solution_folder)

    simulated_problems = glob.glob(
        f"{src_folder}/**/sim_profile.tsv", recursive=True
    )

    res: dict[float, list[float]] = {}

    pur_re = re.compile(r"pur_(\d+)")

    for problem_path in simulated_problems:
        rel_path = "\\".join(
            problem_path.replace(str(src_folder), "").split("\\")[:-1]
        )
        solution_path = f"{solution_folder}{rel_path}\\solutions.tsv"
        if not os.path.exists(solution_path):
            continue
        print(solution_path)

        sol_df = pd.read_csv(solution_path, sep="\t")

        best_sol, best_ll = 0, -np.inf

        for sol, ll in sol_df.to_numpy():
            if ll > best_ll:
                (best_sol, best_ll) = sol, ll
        pur_str = re.findall(pur_re, problem_path)[0]
        pur = int(pur_str) * 10 ** (1 - len(pur_str))

        res.setdefault(pur, []).append(best_sol)

    return res


def plot_aad(ax: Axes, aads: dict[float, list[float]]):
    """
    Plots the AADs similarly to the HapASeg graph.

    Parameters:
    * `ax`: The `matplotlib` axis to draw on.
    * `aads`: A dictionary from the purity to the AADs achieved for it.
    * `offset`: [Optional]: An offset from the purity. Used to make it easier to compare several AAD results.
    """
    xs = []
    ys = []

    # vxs = []
    # vys = []

    for pur, vals in aads.items():
        # vxs.append(pur)
        # vys.append(list(vals))
        for val in vals:
            xs.append(pur)
            ys.append(val)

    ax.scatter(
        xs,
        ys,
    )

    # ax.violinplot(vys, positions=vxs, widths=1.0 / len(xs), showextrema=False)

    ax.set_xlabel("purity")
    ax.set_ylabel("AAD")


def plot_purity(ax: Axes, purities: dict[float, list[float]]):
    ax.set_xlabel("Simulated purity")
    ax.set_ylabel("Simulated purity")

    xs = []
    ys = []
    for pur, vals in purities.items():
        for val in vals:
            xs.append(pur)
            ys.append(val)

    ax.scatter(xs, ys)


def main():
    # p1 = pd.read_csv(BASE_PATH / "out/mass/ch1022gl/pur_02/s_0/sim_profile.tsv", sep="\t")
    # p1["sigma.minor"] = 1e-3
    # p1["sigma.major"] = 1e-3
    # p2 = pd.read_csv(BASE_PATH / "out/mass/ch1022gl/pur_02/s_0/test.tsv", sep="\t")
    # p2["sigma.minor"] = 1e-3
    # p2["sigma.major"] = 1e-3
    # overlap_score, optimal_scale_factor, optimal_purity, non_overlap_length, overlap_length, bins = acr_compare(p1, p2)

    # print(f"{overlap_score=}")
    # print(f"{optimal_scale_factor=}")
    # print(f"{optimal_purity=}")
    # print(f"{non_overlap_length=}")
    # print(f"{overlap_length=}")
    # print(f"{bins=}")

    _, (ax1, ax2) = plt.subplots(1, 2)  # type: ignore

    ax1.semilogy()

    input_addr = r"C:\Users\rsolan\Documents\hap-rs\data\mass"
    output_addr = r"C:\Users\rsolan\Documents\hap-rs\out\mass5_1"

    # plot_aad(
    #     ax1,
    #     compute_output_aad(
    #         r"C:\Users\rsolan\Documents\hap-rs\data\mass\ch1022gl",
    #         r"C:\Users\rsolan\Documents\hap-rs\out\mass4_01",
    #     ),
    #     offset=-0.02,
    # )
    plot_aad(
        ax1,
        compute_output_aad(
            input_addr,
            output_addr,
        ),
    )

    # plot_purity(
    #     ax2,
    #     compute_output_purity(
    #         r"C:\Users\rsolan\Documents\hap-rs\data\mass\ch1022gl",
    #         r"C:\Users\rsolan\Documents\hap-rs\out\mass4_01",
    #     ),
    # )
    plot_purity(
        ax2,
        compute_output_purity(
            input_addr,
            output_addr,
        ),
    )

    plt.show()


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
try:
    from statistics import NormalDist
except ModuleNotFoundError:
    from scipy.stats import norm as NormalDist
from math import log
import sys
from scipy.optimize import minimize

STAT_COLUMNS = ['mu.minor', 'sigma.minor', 'mu.major', 'sigma.major']

# Given original interval I0 = [s0 e0] and new interval I1 = [s1 e1],
# maps coordinates c0 in I0 to c1 in I1.

def interval_remap(c0, s0, e0, s1, e1):
    return (c0 - s0)/(e0 - s0)*(e1 - s1) + s1

def acr_compare(file_1=None, file_2=None, fit_params=False):
    """
    Compare the allelic copy ratio segments between the two seg files.

    :param file_1: name and path of file_1 (default to user input)
    :param file_2: name and path of file_2 (default to user input)
    :return: weighted average of all compared ACR segment normal distributions
    """
    # read in two seg files
    if not file_1 or not file_2:
        file_1 = input('Input first file name: ')
        file_2 = input('Input second file name: ')
    seg1 = pd.read_csv(file_1, sep='\t')
    seg2 = pd.read_csv(file_2, sep='\t')

    # format dataframes (throwing out rows with nan values for mu/sigma)
    seg1 = seg1.dropna(subset=STAT_COLUMNS).reset_index(drop=True)
    seg2 = seg2.dropna(subset=STAT_COLUMNS).reset_index(drop=True)

    # take union of segments to allocate bins
    bins = get_union(seg1, seg2)
    
    # compute upper bound and lower bound of the interval
    shared_bins = bins.loc[~bins.unique]
    lb = shared_bins['mu.minor_1'].min()
    ub = shared_bins['mu.major_1'].max()
    if fit_params:
        # pick good starting point for optimiztion using grid search
        ub_guess = shared_bins['mu.major_2'].max()
        x0_grid = MAD_grid_search(bins, lb, ub, ub_guess)
        minimization_result = minimize(MAD_int_helper, x0=x0_grid,
                                   args=(bins, lb, ub), method='Powell',
                                   options={'xtol': 0.0001, 'ftol': 0.0001})
    
        optimal_lb = minimization_result.x[0]
        optimal_ub = minimization_result.x[1]
        params = minimization_result.x
    else:
        params = None
        optimal_lb, optimal_ub = 0,0

    overlap_score, maj_ad, min_ad = get_MAD_interval(bins, lb, ub, params)
    bins["major_AD"] = maj_ad
    bins["minor_AD"] = min_ad

    if fit_params:
        bins["mu.major_1"] = bins.apply(lambda x: interval_remap(x["mu.major_1"], lb, ub, optimal_lb, optimal_ub), axis=1)
        bins["mu.minor_1"] = bins.apply(lambda x: interval_remap(x["mu.minor_1"], lb, ub, optimal_lb, optimal_ub), axis=1)
        if "sigma.major_1" in bins.columns and "sigma.minor_1" in bins.columns:
            bins["sigma.major_1"] *= (optimal_ub - optimal_lb) / (ub - lb)
            bins["sigma.minor_1"] *= (optimal_ub - optimal_lb) / (ub - lb)

    non_overlap_length = int(bins['length_1_unique'].sum()) + int(bins['length_2_unique'].sum())
    overlap_length = int(bins['length_overlap'].sum())

    return overlap_score, optimal_lb, optimal_ub, non_overlap_length, overlap_length, bins

def MAD_grid_search(bins, lb, ub, ub_guess):
    """
    Preform crude gridsaerch to find good starting point for optmization
    optimizer is very sensitive to starting point
    """
    print("performing grid search to find good starting values for optimizer...")
    print("WARNING: grid search range only valid for benchmarking samples")

    low_grid = np.r_[-250:500:10]
    high_grid = np.r_[0:2 * ub_guess:10]
    grid_res = np.ones((len(low_grid), len(high_grid))) * np.inf
    
    for i, l in enumerate(low_grid):
        for j, h in enumerate(high_grid):
            if l < h:
                score, _, _ = get_MAD_interval(bins, lb, ub, params=np.array([l, h]))
                grid_res[i,j]= score

    opt_ind = np.where(grid_res == grid_res.min())
    return np.r_[low_grid[opt_ind[0][0]], high_grid[opt_ind[1][0]]]

def MAD_int_helper(params, bins, lb, ub):
    """
    Return just the inverse overlap score for optimization minimize function.
    """
    MAD, _, _ = get_MAD_interval(bins, lb, ub, params)
    return MAD

def get_MAD_interval(bins, lb, ub, params=None):
    """
    Calculate the mean absolute difference for the major and minor alleles and get the average weighted by length
    if new upper bound and lower bound params are passed the MAD is computed on the transformed values

    :param params: np array of shape (1,2) containing scale factor and purity
    :param bins: bins dataframe
    :return: MAD, major MAD series, and minor MAD series
    """

    if params is None:
        params=np.array([lb, ub])

    new_lb, new_ub = params
    maj_mu1 = interval_remap(bins['mu.major_1'].values, lb, ub, new_lb, new_ub)
    min_mu1 = interval_remap(bins['mu.minor_1'].values, lb, ub, new_lb, new_ub)
    
    major_overlap = bins['mu.major_2'].subtract(maj_mu1).abs().fillna(0)
    minor_overlap = bins['mu.minor_2'].subtract(min_mu1).abs().fillna(0)

    # get average overlap score
    overlap_score = np.average(np.concatenate((major_overlap, minor_overlap)),
                               weights=np.concatenate([bins['length'], bins['length']]))
    return overlap_score, major_overlap, minor_overlap

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
    major_overlap = bins.apply(lambda x: calc_overlap(x["mu.major_1"], x["sigma.major_1"],
                                                              x["mu.major_2"], x["sigma.major_2"],
                                                              x['unique'], params=params), axis=1)
    minor_overlap = bins.apply(lambda x: calc_overlap(x["mu.minor_1"], x["sigma.minor_1"],
                                                              x["mu.minor_2"], x["sigma.minor_2"],
                                                              x['unique'], params=params), axis=1)

    # get average overlap score
    overlap_score = np.average(np.concatenate((major_overlap, minor_overlap)),
                               weights=np.concatenate([bins['length'], bins['length']]))
    return overlap_score, major_overlap, minor_overlap


def calc_overlap(mu1, sigma1, mu2, sigma2, unique, params=np.array([1.,1.])):
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
        params=np.array([1., 1,])
    scale_factor, purity = params
    mu1 = scale_factor * (purity * (mu1 - 1) + 1)
    sigma1 *= scale_factor

    # check if distributions are equal
    if mu1 == mu2 and sigma1 == sigma2:
        return 1

    # run builtin method in statistics.NormalDist
    # returns OVL; should I give option for Bhattacharyya? todo
    if 'statistics' in sys.modules:
        return NormalDist(mu1, sigma1).overlap(NormalDist(mu2, sigma2))

    # calculate intersection(s) of the two distributions
    x_intersect = calc_pdf_intersect(mu1, sigma1, mu2, sigma2)

    if len(x_intersect) == 1:  # sigma1 == sigma2
        area = NormalDist(mu1, sigma1).cdf(x_intersect[0])
        if area < 0.5:
            return area * 2  # doubled -> pdf1_area = pdf2_area when sigma1 = sigma2
        else:  # take other side of cdf
            return (1 - area) * 2

    # calculate overlap cdf
    mid_section1 = NormalDist(mu1, sigma1).cdf(max(x_intersect)) - \
                   NormalDist(mu1, sigma1).cdf(min(x_intersect))
    mid_section2 = NormalDist(mu2, sigma2).cdf(max(x_intersect)) - \
                   NormalDist(mu2, sigma2).cdf(min(x_intersect))

    # compute sum of overlap sections based on which middle section is larger
    if mid_section1 < mid_section2:
        sum_overlap = 1 + mid_section1 - mid_section2
    else:
        sum_overlap = 1 + mid_section2 - mid_section1

    return sum_overlap


def calc_pdf_intersect(mu1, sigma1, mu2, sigma2):
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
        return [(mu1 + mu2) / 2]

    # calculate roots of log[pdf_1] = log[pdf_2]
    a = 0.5 * (1/sigma2**2 - 1/sigma1**2)
    b = mu1 / sigma1**2 - mu2 / sigma2**2
    c = 0.5 * (mu2**2 / sigma2**2 - mu1**2 / sigma1**2) + log(sigma2 / sigma1)

    return np.roots([a, b, c])


####################

def get_union(seg1_df, seg2_df):
    full_bins = _union_one_sided(seg1_df, seg2_df)
    bins_2 = _union_one_sided(seg2_df, seg1_df)

    # take only unique values from bins_2
    unique_2_df = bins_2.loc[bins_2['unique'] == True]

    # swap stats columns (and 1/2 length columns)
    column_list = []
    for c in unique_2_df.columns.values:
        if '_1' in c:
            c = c.replace('1', '2')
        else:
            c = c.replace('2', '1')
        column_list.append(c)
    unique_2_df.columns = column_list

    bins = pd.concat([full_bins, unique_2_df], ignore_index=True)

    return bins


def _union_one_sided(seg1_df: pd.DataFrame, seg2_df: pd.DataFrame):
    """
    Gets the union of the segment partitions between the two files.

    :param seg1_df: Dataframe of the first seg file
    :param seg2_df: Dataframe of the second seg file
    :return: union of two seg files as a Bin dataframe
    """
    bin_list = []
    for chrom in np.arange(1, 23):  # assumes segments in each chromosome (except for XY)
        segments1 = seg1_df.loc[seg1_df['Chromosome'] == chrom].reset_index(drop=True)
        segments2 = seg2_df.loc[seg2_df['Chromosome'] == chrom].reset_index(drop=True)

        pointer2 = 0  # starting at beginning of seg2
        for i, (start, end) in enumerate(zip(segments1['Start.bp'],
                                             segments1['End.bp'])):
            # create bins for this segment1, updating pointer2 to save computation
            bin_list, pointer2 = create_bins(start, end, segments2, pointer2, segments1.loc[i][STAT_COLUMNS],
                                             chrom, bin_list=bin_list)

    bin_df = pd.DataFrame(bin_list)
    return bin_df


def create_bins(start, end, segments2, pointer2, segments1_stats, chrom, bin_list=None):
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

    start2 = segments2.loc[pointer2]['Start.bp']
    end2 = segments2.loc[pointer2]['End.bp']

    if start < start2:
        if end <= start2:
            bin_list.append(append_bin(start, end, segments1_stats, None, chrom))  # unique segment
            return bin_list, pointer2
        else:
            bin_list.append(append_bin(start, start2, segments1_stats, None, chrom))  # unique segment

            if end < end2:
                bin_list.append(append_bin(start2, end, segments1_stats, segments2.loc[pointer2][STAT_COLUMNS], chrom))
                return bin_list, pointer2
            else:
                bin_list.append(append_bin(start2, end2, segments1_stats, segments2.loc[pointer2][STAT_COLUMNS], chrom))
                return create_bins(end2, end, segments2, pointer2 + 1, segments1_stats, chrom, bin_list=bin_list)
    else:
        if end <= end2:
            bin_list.append(append_bin(start, end, segments1_stats, segments2.loc[pointer2][STAT_COLUMNS], chrom))
            return bin_list, pointer2
        else:
            if start < end2:
                bin_list.append(append_bin(start, end2, segments1_stats, segments2.loc[pointer2][STAT_COLUMNS], chrom))
                return create_bins(end2, end, segments2, pointer2 + 1, segments1_stats, chrom, bin_list=bin_list)
            else:
                return create_bins(start, end, segments2, pointer2 + 1, segments1_stats, chrom, bin_list=bin_list)


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

    bin_dict = {'Chromosome': chromosome,
                'Start.bp': start,
                'End.bp': end,
                'unique': unique,
                'length': length,
                'length_overlap': length_overlap,
                'length_1_unique': length_1_unique,
                'length_2_unique': length_2_unique}

    for key in STAT_COLUMNS:  # todo keep in multiindex
        if stats1 is not None:
            val = stats1[key]
        else:
            val = None
        bin_dict[f'{key}_1'] = val

        if stats2 is not None:
            val = stats2[key]
        else:
            val = None
        bin_dict[f'{key}_2'] = val

    return bin_dict


# TESTING
# if __name__ == '__main__':
#     score, non_o_len, o_len, df = acr_compare(file_1='./REBC-ACBH-rebc.acs.seg', file_2='./REBC-ACBH-wolf.acs.seg')

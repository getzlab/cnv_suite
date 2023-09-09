import pandas as pd
import numpy as np
from intervaltree import IntervalTree
from natsort import natsorted
from collections import namedtuple


def get_segment_interval_trees(seg_dfs, seg_cluster_df=None, cluster_colname='Cluster_assignment'):
    """Make a tree for the segments of this participant given by these seg files

    The seg_cluster_df is optional; if provided, segments will also be marked by cluster assignment.

    :param seg_dfs: seg files given as a pd.DataFrame
            If either no sample_id column is given, will assume segments belong to single sample, with name SAMPLE. All columns other than [Sample_ID, Start.bp, End.bp] will be stored in data tuple for each segment interval.
    :param seg_cluster_df: cluster assignment for segments, given as a pd.DataFrame
    :param cluster_colname: Column name for the cluster in the seg_cluster_df; defaults to Cluster_assignment
    :return: list of IntervalTree givings segment data for each chromosome
    """
    seg_dfs = seg_dfs.astype({'Start.bp': int, 'End.bp': int})
    if 'Sample_ID' not in seg_dfs:
        seg_dfs['Sample_ID'] = 'SAMPLE'

    # remove 0 length segments (not sure how these even exist) - produce errors in IntervalTree
    seg_dfs = seg_dfs[seg_dfs['length'] > 0]

    # must remove periods in columns used to create namedtuple (replace with underscores)
    essential_data_columns = {s: s.replace('.', '_') for s in
                              seg_dfs.columns.drop(['Sample_ID', 'Chromosome', 'Start.bp', 'End.bp']).values}
    Data_Tuple = namedtuple("Data_Tuple", [*essential_data_columns.values(), cluster_colname],
                            rename=True, defaults=['0'])  # if cluster isn't provided, defaults to str(0)

    contig_trees = []
    for contig in natsorted(pd.unique(seg_dfs['Chromosome'])):

        contig_seg_df = seg_dfs.loc[seg_dfs['Chromosome'] == contig]

        single_tree = IntervalTree.from_tuples((start, end, {sample: Data_Tuple(*data)}) for
                                               start, end, sample, data in
                                               zip(contig_seg_df['Start.bp'],
                                                   contig_seg_df['End.bp'],
                                                   contig_seg_df['Sample_ID'],
                                                   contig_seg_df[essential_data_columns.keys()].to_numpy()))

        single_tree.split_overlaps()
        single_tree.merge_equals(data_reducer=lambda current, new: dict(**current, **new))
        # don't remove segments that appear in fewer than all samples - deal with this when plotting

        if not seg_cluster_df:
            contig_trees.append(single_tree)
        else:
            # make tree for this chromosome from phylogic_seg_cluster file
            this_chrom_cluster = seg_cluster_df[seg_cluster_df['Chromosome'] == contig]
            cluster_tree = IntervalTree.from_tuples([(s, e, d) for s, e, d in zip(this_chrom_cluster['Start.bp'],
                                                                                  this_chrom_cluster['End.bp'],
                                                                                  this_chrom_cluster[cluster_colname])])
            # need to test to make sure only one cluster given for each segment
            tree_with_clusters = []
            for interval_obj in single_tree:
                cluster_tree_segs = cluster_tree.overlap(interval_obj.begin, interval_obj.end)
                if len(cluster_tree_segs) > 1:
                    raise ValueError(f'MORE THAN ONE CLUSTER in interval {interval_obj.begin} - {interval_obj.end}')
                elif not cluster_tree_segs:   # empty set
                    single_cluster = 0
                else:
                    single_cluster = cluster_tree_segs.pop().data
                # append cluster onto the data list for each sample in this interval
                # _replace method returns new namedtuple, as they are immutable
                data = {sample: old_tuple._replace(**{cluster_colname: single_cluster})
                        for sample, old_tuple in interval_obj.data.items()}
                tree_with_clusters.append((interval_obj.begin, interval_obj.end, data))

            contig_trees.append(IntervalTree.from_tuples(tree_with_clusters))

    return contig_trees


def calc_absolute_cn(mu_minor, mu_major, sigma, c0, cn_diff, zero_min=True):
    """Calculate ABSOLUTE Copy Number values, given the c0 value and difference between CN 0 and 1.

    Note: Values should be given as numpy arrays to use as vectorized function. However, applying on floats will give correct results as well.

    :param mu_minor: mu minor value(s) as array or float
    :param mu_major: mu major value(s) as array or float
    :param sigma: sigma minor value(s) as array or float
    :param c0: Value of Copy Number 0, i.e. copy number ratio for clonally deleted loci
    :param cn_diff: Difference between CN 1 and CN 0, i.e. CN ratio corresponding to haploid gain/loss
    :param zero_min: boolean whether to replace negative CN values with 0; default True
    :return: (absolute mu minor, absolute mu major, absolute sigma minor, absolute sigma major)
    """
    abs_mu_minor = (mu_minor - c0) / cn_diff
    abs_mu_major = (mu_major - c0) / cn_diff
    abs_sigma = sigma / cn_diff

    if zero_min:  # todo what about sigma?
        abs_mu_minor = np.where(abs_mu_minor < 0, 0, abs_mu_minor)
        abs_mu_major = np.where(abs_mu_major < 0, 0, abs_mu_major)

    return abs_mu_minor, abs_mu_major, abs_sigma


def calc_cn_levels(purity, ploidy, avg_cn=1):
    """Calculate CN zero line and difference between CN levels based on given purity, ploidy and average.

    :param purity: sample tumor purity
    :param ploidy: sample tumor ploidy
    :param avg_cn: average CN value across genome, default 1
    :return: (CN_zero_value, CN_delta_value)
    """
    avg_ploidy = purity * ploidy + 2 * (1 - purity)
    cn_delta = avg_cn * 2 * purity / avg_ploidy
    cn_zero = avg_cn * 2 * (1 - purity) / avg_ploidy
    return cn_zero, cn_delta


def calc_avg_cn(seg_df,
                chr_col='Chromosome',
                len_col='length',
                total_cn_col='tau',
                allele_col='mu.minor',
                remove_null=True):
    """Calculate average Copy Number value across genome

    :param seg_df: pandas.DataFrame segment profile
    :param chr_col: name of chromosome column, default = 'Chromosome'
    :param len_col: name of length column, default = 'length'
    :param total_cn_col: name of total CN column, default = 'tau'
    :param allele_col: name of minor allele CN column, default = 'mu.minor'
    :param remove_null: boolean, True (default) if NA segments should be removed (segments with 0 probes)
    :return: average ACR value
    """
    # remove sex chromosomes
    df = seg_df[~(seg_df[chr_col].isin(['X', 'Y', '23', '24', 23, 24]))].copy()
    if remove_null:  # remove null allelic segments
        df = df[~(df[allele_col].isnull())].copy()
    avg_acr = (df[total_cn_col] * df[len_col]).sum() / (2 * df[len_col].sum())

    return avg_acr


def return_seg_data_at_loci(seg_trees, sample, contig, pos):
    """Access segment data at given loci, handling missing data.

    If segment doesn't exist at loci or sample doesn't have data at loci, returns None

    :param seg_trees: dict of IntervalTrees with segment data
    :param sample: sample name
    :param contig: chromosome
    :param pos: loci position
    :return: dict with data (or None if segment doesn't exist)

    :raise: ValueError if contig is not int or castable to int. NO X/Y chromosomes!
    """
    try:
        contig = int(contig)
    except ValueError as e:
        raise ValueError('Contig must be an int (or castable to int). Consider using switch_contigs if contigs include X/Y') from e
    else:
        if contig <= len(seg_trees):
            segment = seg_trees[contig - 1][pos]
            if len(segment) == 0:  # no segment present for any sample
                return None
            data = segment.pop().data  # only one segment should be in tree (after split and merge overlaps)
            if sample not in data.keys():  # no segment present for this sample
                return None
            return data[sample]._asdict()


def apply_segment_data_to_df(df, seg_trees):
    """Annotate dataframe with segment data

    :param df: pandas.DataFrame with loci information (with at least 'Chromosome' and 'Start_position' columns)
    :param seg_trees: dict of IntervalTrees with segment data
    :return: pandas.DataFrame with segment data appended to given df
    """
    df_copy = df.copy()
    if 'Sample_ID' not in df_copy:
        df_copy['Sample_ID'] = 'SAMPLE'

    data = [return_seg_data_at_loci(seg_trees, s, c, p) for s, c, p in zip(df_copy['Sample_ID'],
                                                                           df_copy['Chromosome'],
                                                                           df_copy['Start_position'])]
    # get columns from one of the dicts in data
    col_names = ['No_Segment_Data']
    for d in data:
        if isinstance(d, dict):
            col_names = d.keys()
            break
        else:
            continue

    # make dict of NaN's using those columns
    # turn into df
    data_df = pd.DataFrame([{n: np.NaN for n in col_names} if pd.isnull(d) else d for d in data])

    return pd.concat([df_copy.reset_index(drop=True), data_df], axis=1)

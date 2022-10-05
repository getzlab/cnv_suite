import re
import numpy as np
import pandas as pd


def get_alt_count(m_prop, p_prop, m_present, p_present, coverage, correct_phase):
    """Returns number of alternate reads generated from a binomial.

    If both alleles have mutation (homozygous mutation), returns the given coverage. If neither allele has mutation,
    returns 0. Also adjusts for phasing by the boolean correct_phase."""
    if not correct_phase:
        m_present, p_present = p_present, m_present

    if m_present and p_present:
        return coverage   # add noise? as in: np.random.binomial(coverage, 0.999)
    elif not m_present and not p_present:
        return 0  # noise?
    elif m_present:
        return np.random.binomial(coverage, m_prop)
    else:
        return np.random.binomial(coverage, p_prop)


def get_average_ploidy(pat_ploidy, mat_ploidy, purity):
    """Get the average ploidy defined by the paternal/maternal tumor CN and the purity."""
    return (pat_ploidy + mat_ploidy) * purity + 2 * (1 - purity)


def single_allele_ploidy(allele, start, end):
    """Get the ploidy for this allele tree over this interval [start, end)."""
    intervals = allele.envelop(start, end) | allele[start] | allele[end - 1]
    if len(intervals) == 1:
        return intervals.pop().data[3]
    else:
        interval_totals = [(min(i.end, end) - max(i.begin, start)) * i.data[3] for i in intervals]
        return sum(interval_totals) / (end - start)


def get_contigs_from_header(vcf_fn):
    contig_dict = {}
    with open(vcf_fn, "r") as vcf:
        pre_contig = True
        post_contig = False
        while not post_contig:
            line = vcf.readline()
            re_groups = re.search("##(?P<id>\w+)=(?P<value>.*)", line)
            if re_groups.group('id') == 'contig':
                pre_contig = False
                contig_groups = re.search("<ID=(?P<name>[chrXY\d]+),length=(?P<len>\d+)>", re_groups.group('value'))
                contig_dict[contig_groups.group('name')] = int(contig_groups.group('len'))
            elif not pre_contig:
                post_contig = True

    return contig_dict


def switch_contigs(input_data):
    """Return the input data with 'chr' removed from contigs and X/Y changed to 23/24.

    :param input_data: dict or pd.DataFrame with contig as keys or column
    :returns: dict or pd.DataFrame with altered contig names"""
    if type(input_data) == pd.DataFrame:
        contig_column_names = ['Chr', 'Chromosome', 'Chrom', 'Contig']  # defines possible column names
        # accounts for all lower/upper-case
        contig_column_names = contig_column_names + [s.lower() for s in contig_column_names] + [s.upper() for s in
                                                                                                contig_column_names]
        contig_column_names = contig_column_names + [s + 's' for s in contig_column_names]  # pluralizes column names
        column_idx = np.where([c in contig_column_names for c in input_data.columns])[0][0]  # find contig column
        column_label = input_data.columns[column_idx]

        input_data[column_label] = input_data[column_label].apply(
            lambda x: re.search('(?<=chr)[\dXY]+|^[\dXY]+', x).group())
        input_data.replace(to_replace={column_label: {'X': '23', 'Y': '24'}},  inplace=True)
        # should already be sorted
        # input_data.sort_values([column_label, 'start'], key=natsort.natsort_keygen(), inplace=True)
        return input_data
    elif type(input_data) == dict:
        input_data = {re.search('(?<=chr)[\dXY]+|^[\dXY]+', key).group(): loc for key, loc in input_data.items()}
        if 'X' in input_data.keys():
            input_data['23'] = input_data['X']
            input_data.pop('X')
        if 'Y' in input_data.keys():
            input_data['24'] = input_data['Y']
            input_data.pop('Y')
        return input_data
    else:
        raise ValueError(f'Only dictionaries and pandas DataFrames supported. Not {type(input_data)}.')

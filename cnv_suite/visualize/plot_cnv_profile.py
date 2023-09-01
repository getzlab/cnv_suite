#!/bin/bash/env python3

import numpy as np
import matplotlib.pyplot as plt
import plotly.subplots
from matplotlib import patches
from natsort import natsorted
import plotly.graph_objects as go
import pandas as pd

from cnv_suite import calc_cn_levels, calc_absolute_cn, calc_avg_cn


def plot_acr_static(seg_df, ax, csize,
                    segment_colors=None, sigmas=None, min_seg_lw=None, y_upper_lim=None):
    """Create static Allelic Copy Ratio plot for given segment profile.

    :param seg_df: pandas.DataFrame with segment profile (allelic CN mu and sigmas)
    :param ax: matplotlib Figure axes
    :param csize: dict with chromosome sizes, as {contig_name: size}
    :param segment_colors: color specification for segments. One of [black, difference (default), cluster, or blue_red (any other input)].
    :param sigmas: boolean, True (default) if segments should have heights determined by sigma values
    :param min_seg_lw: Segment line_width (for all segments if sigmas=False or minimum if sigmas=True); default=2
    :param y_upper_lim: yaxis upper limit; default=2
    :return: None (modifies given figure.axes)
    """
    if not min_seg_lw:
        min_seg_lw = 2
    if not y_upper_lim:
        y_upper_lim = 2

    seg_df, chr_order, chrom_start = prepare_df(seg_df, csize, suffix='.bp')
    add_background(ax, chr_order, csize, height=max(7, y_upper_lim+1))

    # determine segment colors based on input
    if segment_colors == 'difference' or segment_colors is None:
        seg_df['color_bottom'], seg_df['color_top'] = calc_color(seg_df, 'mu_major', 'mu_minor')
    elif segment_colors == 'black':
        seg_df['color_bottom'] = '#000000'
        seg_df['color_top'] = '#000000'
    elif segment_colors == 'cluster':
        phylogic_color_dict = get_phylogic_color_scale()
        seg_df['color_bottom'] = seg_df['cluster_assignment'].map(phylogic_color_dict)
        seg_df['color_top'] = seg_df['color_bottom']
    else:
        seg_df['color_bottom'] = '#2C38A8'  # blue
        seg_df['color_top'] = '#E6393F'  # red

    # draw segments as lines with default line width
    ax.hlines(seg_df['mu_minor'].values, seg_df['genome_start'], seg_df['genome_end'],
              color=seg_df['color_bottom'], lw=min_seg_lw)
    ax.hlines(seg_df['mu_major'].values, seg_df['genome_start'], seg_df['genome_end'],
              color=seg_df['color_top'], lw=min_seg_lw)

    # if sigmas are desired, draw over segments
    if sigmas or sigmas is None:
        if 'sigma_major' not in seg_df or 'sigma_minor' not in seg_df:
            print(f'Displaying sigmas is desired, but no sigma columns (i.e. "sigma_major") exist.')
        else:
            for _, x in seg_df.iterrows():
                ax.add_patch(patches.Rectangle(
                    (x['genome_start'], x['mu_major'] - x['sigma_major']),
                    x['genome_end'] - x['genome_start'], 2 * x['sigma_major'],
                    color=x['color_top'],
                    alpha=1,
                    linewidth=0
                ))
                ax.add_patch(patches.Rectangle(
                    (x['genome_start'], x['mu_minor'] - x['sigma_minor']),
                    x['genome_end'] - x['genome_start'], 2 * x['sigma_minor'],
                    color=x['color_bottom'],
                    alpha=1,
                    linewidth=0
                ))

    # layout (can be overridden)
    ax.set_xticks(np.asarray(list(chrom_start.values())[:-1]) + np.asarray(list(csize.values())) / 2)
    ax.tick_params(axis='x', bottom=False)
    ax.set_xticklabels(chr_order, fontsize=10)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(12)
    ax.set_xlim(0, chrom_start['Z'])

    ax.set_yticks(list(range(y_upper_lim + 1)))
    ax.set_yticklabels([str(i) for i in range(y_upper_lim + 1)], fontsize=12)
    ax.set_ylim(-0.05, y_upper_lim + 0.05)
    plt.setp(ax.spines.values(), visible=False)
    ax.spines['left'].set(lw=1, position=('outward', 10), bounds=(0, y_upper_lim), visible=True)
    plt.xlabel("Chromosome")
    plt.ylabel("Allelic Copy Number")


def plot_acr_subplots(fig_list, title, fig_names, csize, height_per_sample=350, **kwargs):
    """Add each Figure in list to plotly subplots.

    Additional kwargs are passed to plotly's make_subplots method.

    :param height_per_sample: plot height for each sample plot (scales based on size of fig_list); default 350
    :param csize: dict with chromosome sizes, as {contig_name: size}
    :param fig_list: List of plotly.graph_objects.Figure, one for each row
    :param title: Title of plot
    :param fig_names: Title for each subplot (one for each row)
    """
    fig = plotly.subplots.make_subplots(rows=len(fig_list), cols=1,
                                        shared_xaxes=True, subplot_titles=fig_names,
                                        vertical_spacing=0.15/len(fig_list), **kwargs)

    for i, sub_figure in enumerate(fig_list):
        fig.add_traces(sub_figure.data, rows=i + 1, cols=1)
        fig.update_yaxes(fig_list[i].layout.yaxis, row=i+1, col=1)

        # Add chromosome background back in
        add_background(fig, csize.keys(), csize, plotly_row=i+1, plotly_col=1)

    # update height based on subplot number
    fig.update_layout(height=len(fig_list)*height_per_sample + 50,
                      title_text=title,
                      title_font_size=26,
                      plot_bgcolor='white')
    fig.update_xaxes(fig_list[0].layout.xaxis)

    # show x-axis title on only bottom plot
    fig.update_xaxes(title_text='', selector=(lambda x: not x['showticklabels']))

    return fig


def plot_acr_interactive(seg_df, csize,
                         segment_colors='Difference', sigmas=True,
                         purity=None, ploidy=None,
                         min_seg_lw=0.015, y_upper_lim=2):
    """Create interactive plotly Allelic Copy Ratio plot for given segment profile.

    :param seg_df: pandas.DataFrame with segment profile (allelic CN mu and sigmas)
    :param csize: dict with chromosome sizes, as {contig_name: size}
    :param segment_colors: color specification for segments. One of [Black, Difference (default), Cluster, or Blue/Red].
    :param sigmas: boolean, True (default) if segments should have heights determined by sigma values
    :param min_seg_lw: Segment line_width (for all segments if sigmas=False or minimum if sigmas=True); default=0.015
    :param y_upper_lim: yaxis upper limit; default=2
    :return: (plotly.graph_objects.Figure, pd.DataFrame, int, int)
             (ACR Figure, modified Segment df, trace start, trace end)
    """
    # fig should have background set to white
    seg_df, chr_order, chrom_start = prepare_df(seg_df, csize, suffix='.bp')
    if sigmas and 'sigma_major' not in seg_df:
        print(f'Displaying sigmas is desired, but no sigma columns (i.e. {"sigma_major"}) exist.')
        sigmas = False

    fig = go.Figure()
    add_background(fig, chr_order, csize)

    # calculate Phylogic cluster colors
    if 'cluster_assignment' in seg_df.columns:
        phylogic_color_dict = get_phylogic_color_scale()
        seg_df['color_bottom_cluster'] = seg_df['cluster_assignment'].map(phylogic_color_dict)
        seg_df['color_top_cluster'] = seg_df['color_bottom_cluster']

    # calculate red/blue gradient for difference between segments
    seg_df['color_bottom_diff'], seg_df['color_top_diff'] = calc_color(seg_df, 'mu_major', 'mu_minor')

    # calculate absolute segment values (scaled by purity/ploidy)
    if purity and ploidy:
        c_0, c_delta = calc_cn_levels(purity, ploidy, avg_cn=calc_avg_cn(seg_df, allele_col='mu_minor', total_cn_col='tau'))
        sigma = np.zeros(seg_df.shape[0]) if 'sigma_major' not in seg_df else seg_df['sigma_major']
        seg_df['mu_major_adj'], seg_df['mu_minor_adj'], seg_df['sigma_adj'] = calc_absolute_cn(
            seg_df['mu_major'], seg_df['mu_minor'], sigma, c_0, c_delta)
        seg_df['color_bottom_diff_adj'], seg_df['color_top_diff_adj'] = calc_color(seg_df, 'mu_major_adj', 'mu_minor_adj')

    trace_num = len(fig.data)
    seg_df.apply(lambda x: make_cnv_scatter(x, fig, lw=min_seg_lw, sigmas=sigmas), axis=1)
    trace_end = len(fig.data)

    # update segment colors and absolute CN values, if desired
    update_cnv_color_absolute(fig, seg_df, absolute=False, color=segment_colors,
                              start_trace=trace_num, end_trace=trace_end)

    # modify layout
    fig.update_xaxes(showgrid=False,
                     zeroline=False,
                     tickvals=np.asarray(list(chrom_start.values())[:-1]) + np.asarray(list(csize.values())) / 2,
                     ticktext=chr_order,
                     tickfont_size=10,
                     tickangle=0,
                     range=[0, chrom_start['Z']])
    fig.update_xaxes(title_text="Chromosome")
    fig.update_yaxes(showgrid=False,
                     zeroline=False,
                     tickvals=list(range(int(np.floor(y_upper_lim)) + 1)),
                     ticktext=[str(i) for i in range(int(np.floor(y_upper_lim)) + 1)],
                     tickfont_size=12,
                     ticks="outside",
                     range=[-0.05, y_upper_lim + 0.05],
                     title_text="Allelic Copy Number")

    ################
    fig.update_layout(plot_bgcolor='white')

    return fig, seg_df, trace_num, trace_end


def make_cnv_scatter(series, fig, lw=0.015, sigmas=False):
    """Add segment to figure as Scatter traces

    :param series: pandas.Series for single segment
    :param fig: plotly.Figure
    :param lw: Segment line_width (for all segments if sigmas=False or minimum if sigmas=True); default=0.015
    :param sigmas: boolean, True if segments should have heights determined by sigma values; default = False
    :return: None (modifies given Figure)
    """
    start = series['genome_start']
    end = series['genome_end']
    mu_maj = series['mu_major']
    mu_min = series['mu_minor']
    sigma = series['sigma_major'] if sigmas else 0
    length = end - start
    n_probes = series['n_probes'] if 'n_probes' in series else np.NaN
    color_maj = '#E6393F'  # red
    color_min = '#2C38A8'  # blue
    cluster = series['cluster_assignment'] if 'cluster_assignment' in series else np.NaN

    fig.add_trace(go.Scatter(x=[start, start, end, end],
                  y=[mu_min + sigma, mu_min - sigma, mu_min - sigma, mu_min + sigma],
                  fill='toself', fillcolor=color_min, mode='none',
                  hoverinfo='none', name='cnv_sigma',
                  showlegend=False, visible=sigmas))
    fig.add_trace(go.Scatter(x=[start, start, end, end],
                  y=[mu_maj + sigma, mu_maj - sigma, mu_maj - sigma, mu_maj + sigma],
                  fill='toself', fillcolor=color_maj, mode='none',
                  hoverinfo='none', name='cnv_sigma',
                  showlegend=False, visible=sigmas))
    fig.add_trace(go.Scatter(x=[start, start, end, end],
                  y=[mu_min + lw, mu_min - lw, mu_min - lw, mu_min + lw],
                  fill='toself', fillcolor=color_min, mode='none',
                  hoveron='fills', name='cnv',
                  text=f'chr{series["Chromosome"]}:{series["Start.bp"]}-{series["End.bp"]}; '
                       f'CN Minor: {mu_min:.2f} +-{sigma:.4f}; '  # todo make original, so no updating needed
                       f'Cluster: {cluster}; '
                       f'Length: {length:.2e} ({n_probes} probes)',
                  showlegend=False))
    fig.add_trace(go.Scatter(x=[start, start, end, end],
                  y=[mu_maj + lw, mu_maj - lw, mu_maj - lw, mu_maj + lw],
                  fill='toself', fillcolor=color_maj, mode='none',
                  hoveron='fills', name='cnv',
                  text=f'chr{series["Chromosome"]}:{series["Start.bp"]}-{series["End.bp"]}; '
                       f'CN Major: {mu_maj:.2f} +-{sigma:.4f}; '  # todo make original so no updating needed
                       f'Cluster: {cluster}; '
                       f'Length: {length:.2e} ({n_probes} probes)',
                  showlegend=False))


def update_cnv_color_absolute(fig, seg_df, absolute, color, start_trace, end_trace):
    """Helper function to update both the CN values (raw or ABSOLUTE) and segment colors.

    :param fig: plotly.graph_objects.Figure to be updated
    :param seg_df: pd.DataFrame with segment data; should include relevant columns (mu_major_adj, color_bottom_diff, etc.)
    :param absolute: True if segments should be displayed as ABSOLUTE adjusted
    :param color: color specification for segments. One of [Black, Difference, Cluster, Blue/Red].
    :param start_trace: which trace to begin with in Figure
    :param end_trace: which trace to end with in Figure
    """

    if absolute and 'mu_major_adj' in seg_df:
        major_cn = seg_df['mu_major_adj']
        minor_cn = seg_df['mu_minor_adj']
        sigma = seg_df['sigma_adj']
    else:
        major_cn = seg_df['mu_major']
        minor_cn = seg_df['mu_minor']
        sigma = seg_df['sigma_major']

    if color == 'Difference' and absolute and 'color_bottom_diff_adj' in seg_df:
        minor_color, major_color = seg_df['color_bottom_diff_adj'], seg_df['color_top_diff_adj']
    elif color == 'Difference':
        minor_color, major_color = seg_df['color_bottom_diff'], seg_df['color_top_diff']
    elif color == 'Cluster' and 'color_bottom_cluster' in seg_df:
        minor_color, major_color = seg_df['color_bottom_cluster'], seg_df['color_top_cluster']
    elif color == 'Black':
        minor_color = major_color = ['#000000'] * seg_df.shape[0]  # black
    # elif color == 'Clonal/Subclonal':  # todo either get from CN levels or clusters?
    #     pass
    else:
        minor_color = ['#2C38A8'] * seg_df.shape[0]  # blue
        major_color = ['#E6393F'] * seg_df.shape[0]  # red

    update_cnv_scatter_cn(fig, major_cn, minor_cn, sigma, start_trace, end_trace)
    update_cnv_scatter_color(fig, minor_color, major_color, start_trace, end_trace)


def update_cnv_scatter_cn(fig, major, minor, sigma, start_trace, end_trace, lw=0.015):
    """Updates y values for CNV traces from start_trace to end_trace given by major/minor lists (optionally change line width also).

    Critical that original traces were added in order: minor (sigma), major (sigma), minor, major as performed in make_cnv_scatter method
    :param fig: plotly.Figure
    :param major: array-like, new major allele mu values
    :param minor: array-like, new minor allele mu values
    :param sigma: array-like, new sigma values
    :param start_trace: which trace to begin with in Figure
    :param end_trace: which trace to end with in Figure
    :param lw: Segment line_width (for all segments if sigmas=False or minimum if sigmas=True); default=0.015
    :return: None
    """
    assert end_trace - start_trace == len(major) * 4 and len(major) == len(minor) == len(sigma)

    for i, (minor_val, major_val, sigma) in enumerate(zip(minor, major, sigma)):
        fig.data[start_trace + 4 * i]['y'] = [minor_val + sigma, minor_val - sigma, minor_val - sigma, minor_val + sigma]
        fig.data[start_trace + 4 * i + 1]['y'] = [major_val + sigma, major_val - sigma, major_val - sigma, major_val + sigma]
        fig.data[start_trace + 4 * i + 2]['y'] = [minor_val + lw, minor_val - lw, minor_val - lw, minor_val + lw]
        fig.data[start_trace + 4 * i + 3]['y'] = [major_val + lw, major_val - lw, major_val - lw, major_val + lw]


def update_cnv_scatter_color(fig, color_minor, color_major, start_trace, end_trace):
    """Updates colors of the segments, based on given arrays and trace numbers.
    
    :param fig: plotly.Figure
    :param color_minor: array-like, new minor allele color values
    :param color_major: array-like, new major allele color values
    :param start_trace: which trace to begin with in Figure
    :param end_trace: which trace to end with in Figure
    :return: None
    """
    assert end_trace - start_trace == len(color_minor) * 4 == len(color_major) * 4

    for i, (minor_val, major_val) in enumerate(zip(color_minor, color_major)):
        fig.data[start_trace + 4 * i]['fillcolor'] = minor_val
        fig.data[start_trace + 4 * i + 1]['fillcolor'] = major_val
        fig.data[start_trace + 4 * i + 2]['fillcolor'] = minor_val
        fig.data[start_trace + 4 * i + 3]['fillcolor'] = major_val


def update_cnv_scatter_sigma_toggle(fig, sigmas):
    """Changes the visibility of the sigmas.
    
    :param fig: plotly.Figure
    :param sigmas: boolean, True if segments should have heights determined by sigma values
    :return: None
    """
    fig.update_traces(dict(visible=sigmas), selector={'name': 'cnv_sigma'})


def add_background(ax, chr_order, csize, height=100, plotly_row=None, plotly_col=None):
    """Add background alternating gray/white bars to demarcate chromosomes.
    
    :param ax: matplotlib axes or plotly.Figure
    :param chr_order: contig names in order as list
    :param csize: dict with chromosome sizes, as {contig_name: size}
    :param height: height of bars, default=100
    :param plotly_row: 1-indexed row index to plot background
    :param plotly_col: 1-indexed col index to plot background
    :return: None
    """
    base_start = 0
    chrom_ticks = []
    patch_color = 'white'

    is_plotly_figure = isinstance(ax, go.Figure) and not ((plotly_row is None) or (plotly_col is None))
    for chrom in chr_order:
        if type(ax) == go.Figure:
            if is_plotly_figure:
                ax.add_shape(x0=base_start, x1=base_start + csize[chrom], y0=0, y1=1, fillcolor=patch_color,
                             opacity=0.1, layer='below', line_width=0, yref= 'y domain', row=plotly_row, col=plotly_col)
            else:
                ax.add_shape(x0=base_start, x1=base_start + csize[chrom], y0=0, y1=1, fillcolor=patch_color,
                             opacity=0.1, layer='below', line_width=0, yref= 'y domain')
        else:
            p = patches.Rectangle((base_start, -0.2), csize[chrom], height, fill=True, facecolor=patch_color,
                                  edgecolor=None, alpha=.1)  # Background

            if is_plotly_figure:
                ax.add_patch(p, row=plotly_row, col=plotly_col)
            else:
                ax.add_patch(p)
        patch_color = 'gray' if patch_color == 'white' else 'white'
        chrom_ticks.append(base_start + csize[chrom] / 2)
        base_start += csize[chrom]


def calc_color(seg_df, major, minor):
    """Calculate major/minor allele colors based on the difference between them
    
    :param seg_df: pandas.DataFrame with segment data
    :param major: column name for mu_major
    :param minor: column name for mu_minor
    :return: (array of minor allele colors, array of major allele colors)
    """
    from matplotlib import colors
    cmap = colors.LinearSegmentedColormap.from_list("", ["blue", "purple", "red"])
    color_bottom = seg_df.apply(lambda x: colors.rgb2hex(cmap(
        int(np.floor(max(0, (0.5 - 0.5 * scale_diff(x[major] - x[minor])) * 255))))),
                                          axis=1)
    color_top = seg_df.apply(lambda x: colors.rgb2hex(cmap(
        int(np.floor(min(255, (0.5 + 0.5 * scale_diff(x[major] - x[minor])) * 255))))),
                                       axis=1)
    return color_bottom, color_top


def scale_diff(mu_diff):
    """Transforms distance between alleles for use in color formula    
    :param mu_diff: Distance between alleles
    :return: Transformed value
    """
    return (7*mu_diff**2) / (7*mu_diff**2 + 10)


def prepare_df(df, csize, suffix='.bp'):
    """Preparation of dataframe for use (adding genome_position), collecting column names, and chromosome starting positions
    
    :param df: pandas.DataFrame segment profile
    :param csize: dict with chromosome sizes, as {contig_name: size}
    :param suffix: suffix on "Start" and "End" position columns
    :return: (modified_segment_df, chromosome_order_list, chromsome_start_dict, column_names_dict)
    """
    # discover columns
    if 'mu.major' in df.columns:
        col_names = dict(
            mu_major = 'mu.major',
            mu_minor = 'mu.minor',
            sigma_major = 'sigma.major',
            sigma_minor = 'sigma.minor',
            tau = 'tau'
        )
    elif 'hscr.a2' in df.columns:
        col_names = dict(
            mu_major = 'hscr.a2',
            mu_minor = 'hscr.a1',
            sigma_major = 'seg_sigma',  # = tau sigma (not allelic sigma), generally slightly lower
            sigma_minor = 'seg_sigma',  # = tau sigma
            tau = 'tau'
        )
        df['tau'] = df['total_copy_ratio'] * 2
    # todo add major and minor case (and no sigma?)
    else:
        col_names = None

    chr_order = natsorted(list(csize.keys()))
    chrom_start = {chrom: start for (chrom, start) in
                   zip(np.append(chr_order, 'Z'), np.cumsum([0] + [csize[a] for a in chr_order]))}

    df['Chromosome'] = df['Chromosome'].astype(str)
    df = df[df['Chromosome'].isin(chr_order)]

    df[f'Start{suffix}'] = df[f'Start{suffix}'].astype(int)
    df[f'End{suffix}'] = df[f'End{suffix}'].astype(int)
    df['genome_start'] = df.apply(lambda x: chrom_start[str(x['Chromosome'])] + x[f'Start{suffix}'], axis=1)
    df['genome_end'] = df.apply(lambda x: chrom_start[str(x['Chromosome'])] + x[f'End{suffix}'], axis=1)

    for key, val in col_names.items():
        try:
            df[key] = df[val]
        except KeyError:
            print(f'{val} does not exist in df columns')

    return df, chr_order, chrom_start


def get_hex_string(c):
    return '#{:02X}{:02X}{:02X}'.format(*c)


def get_phylogic_color_scale():
    """Generate dictionary defining phylogic cluster colors."""
    phylogic_color_list = [[166, 17, 129],
                           [39, 140, 24],
                           [103, 200, 243],
                           [248, 139, 16],
                           [16, 49, 41],
                           [93, 119, 254],
                           [152, 22, 26],
                           [104, 236, 172],
                           [249, 142, 135],
                           [55, 18, 48],
                           [83, 82, 22],
                           [247, 36, 36],
                           [0, 79, 114],
                           [243, 65, 132],
                           [60, 185, 179],
                           [185, 177, 243],
                           [139, 34, 67],
                           [178, 41, 186],
                           [58, 146, 231],
                           [130, 159, 21],
                           [161, 91, 243],
                           [131, 61, 17],
                           [248, 75, 81],
                           [32, 75, 32],
                           [45, 109, 116],
                           [255, 169, 199],
                           [55, 179, 113],
                           [34, 42, 3],
                           [56, 121, 166],
                           [172, 60, 15],
                           [115, 76, 204],
                           [21, 61, 73],
                           [67, 21, 74],  # Additional colors, uglier and bad
                           [123, 88, 112],
                           [87, 106, 46],
                           [37, 66, 58],
                           [132, 79, 62],
                           [71, 58, 32],
                           [59, 104, 114],
                           [46, 107, 90],
                           [84, 68, 73],
                           [90, 97, 124],
                           [121, 66, 76],
                           [104, 93, 48],
                           [49, 67, 82],
                           [71, 95, 65],
                           [127, 85, 44],  # even more additional colors, gray
                           [88, 79, 92],
                           [220, 212, 194],
                           [35, 34, 36],
                           [200, 220, 224],
                           [73, 81, 69],
                           [224, 199, 206],
                           [120, 127, 113],
                           [142, 148, 166],
                           [153, 167, 156],
                           [162, 139, 145],
                           [0, 0, 0]]  # black
    colors_dict = {str(i): get_hex_string(c) for i, c in enumerate(phylogic_color_list)}
    return colors_dict


def save_static_plot(seg_df, output_fn, csize=None,
                     segment_colors=None, sigmas=None, min_seg_lw=None, y_upper_lim=None):
    if not csize:
        # get max positions for each chromosome
        csize = seg_df.groupby('Chromosome')['End.bp'].max().to_dict()

    fig, ax = plt.subplots()
    plot_acr_static(seg_df, ax=ax, csize=csize, segment_colors=segment_colors,
                    sigmas=sigmas, min_seg_lw=min_seg_lw, y_upper_lim=y_upper_lim)

    fig.savefig(output_fn)


def main():
    # parse args
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Save static CN Profile plot for this segment dataframe')
    parser.add_argument("segment_fn", help='CN Segment Profile filename')
    parser.add_argument("output_fn", help='Output path for saved figure')
    parser.add_argument("--csize_file", help='tsv file containing chromosome sizes')
    parser.add_argument("-sc", "--segment_colors", choices=['black', 'difference', 'cluster', 'blue_red'],
                        help='Method for determining segment colors')
    parser.add_argument("--hide_sigmas", dest='sigmas', action='store_false', help='Do not display segment sigmas')
    parser.add_argument("--min_seg_lw",
                        help='Segment line_width (all segments if sigmas are hidden else minimum); default=2')
    parser.add_argument("--y_upper", help='yaxis upper limit; default=2')

    args = parser.parse_args()

    if args.csize_file and os.path.exists(args.csize_file):
        _, ext = os.path.splitext(args.csize_file)
        if ext == '.bed':
            columns = ['chr', 'start', 'len']  # three columns if bed file
        else:
            columns = ['chr', 'len']
        csize = pd.read_csv(args.csize_file, sep='\t', header=None, names=columns).set_index(
            'chr', inplace=True).to_dict()['len']
    else:
        csize = None

    segment_df = pd.read_csv(args.segment_fn, sep='\t', header=0)

    save_static_plot(segment_df, args.output_fn, csize=csize, segment_colors=args.segment_colors,
                     sigmas=args.sigmas, min_seg_lw=args.min_seg_lw, y_upper_lim=args.y_upper)


if __name__ == '__main__':
    main()

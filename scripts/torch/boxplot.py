from typing import Dict, List

import numpy as np
import csv

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes


# function for setting the colors of the box plots pairs
def setBoxColors(bp, colors: List[str] = None, num_data=3):
    if colors is None:
        colors = ['b', 'r', 'k']
    setp(bp['boxes'][0], color=colors[0])
    setp(bp['caps'][0], color=colors[0])
    setp(bp['caps'][1], color=colors[0])
    setp(bp['whiskers'][0], color=colors[0])
    setp(bp['whiskers'][1], color=colors[0])
    setp(bp['medians'][0], color=colors[0])

    if num_data > 1:
        setp(bp['boxes'][1], color=colors[1])
        setp(bp['caps'][2], color=colors[1])
        setp(bp['caps'][3], color=colors[1])
        setp(bp['whiskers'][2], color=colors[1])
        setp(bp['whiskers'][3], color=colors[1])
        setp(bp['medians'][1], color=colors[1])

    if num_data > 2:
        setp(bp['boxes'][2], color=colors[2])
        setp(bp['caps'][4], color=colors[2])
        setp(bp['caps'][5], color=colors[2])
        setp(bp['whiskers'][4], color=colors[2])
        setp(bp['whiskers'][5], color=colors[2])
        setp(bp['medians'][2], color=colors[2])


def boxplot_scores(data_list, tags, legends, ax, title: str, pre=None, on_top=False):  #: List[Dict[str: np.ndarray]]
    # colors = ['maroon', 'orangered', 'darkgoldenrod']  # red shade
    colors = ['navy', 'steelblue', 'seagreen'] #  blue shae
    h_line_color = 'dimgrey'  # 'tab:cyan'
    ax.set_facecolor('whitesmoke')

    pos = 0
    ticks_pos = []
    n = len(data_list) + 1
    total_ticks = len(tags) * n
    ax.axvline(x=pos, ls='-.', color='k', lw='0.25')
    for i, tag in enumerate(tags):
        # first boxplot pair
        data = []
        for d in data_list:
            data.append(d[tag])
        bp = ax.boxplot(data, positions=[pos+i for i in range(1, n)], widths=0.6, showfliers=False)
        setBoxColors(bp, colors, num_data=n-1)
        ticks_pos.append(pos + n/2)
        # if i != len(tags) - 1:
        ax.axvline(x=pos + n, ls='-.', color='k', lw='0.25')
        pos += n

    if not on_top:
        # draw temporary red and blue lines and use them to create a legend
        h_array = [plot([1, 1], ls='-', color=color)[0] for color in colors]
        h_array.append(plot([1, 1], ls='--', color=h_line_color)[0])

        legend(h_array, legends, loc='lower right')
        [h.set_visible(False) for h in h_array]

    if on_top:
        ax.set_xticks([])
    else:
        ax.set_xticks(ticks_pos)
        ax.set_xticklabels(tags)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

    if pre is not None:
        pos = 0
        x0, x1 = ax.get_xbound()
        w = x1 - x0
        margin = -0.1
        for i, tag in enumerate(tags):
            ax.axhline(y=pre[tags[i]], ls='--', color=h_line_color,
                       xmin=(pos + margin - x0) / w, xmax=(pos - margin + n - x0) / w)
            pos += n
    ax.set_ylabel(title, weight='bold')

    # savefig('boxcompare.svg', bbox_inches=fig.bbox_inches)
    # show()


def draw_boxplot(filenames: List[str], ax, title: str, pre_reg_value=None, on_top=False):
    data1, tags1 = read_csv_result_vert(filenames[0])
    data2, _ = read_csv_result_vert(filenames[1])
    data3, _ = read_csv_result_vert(filenames[2])

    # plot the box plot using the data list and the tags
    boxplot_scores([data1, data2], tags1,
                   legends=('baseline (SSD)', 'learnt'),
                   pre=pre_reg_value, ax=ax, title=title, on_top=on_top)


def read_csv_result_vert(filename):
    with open(filename, newline='') as csvfile:
        info = csv.reader(csvfile, delimiter=',', quotechar='|')
        data3 = {}
        tags = []
        for i, row in enumerate(info):
            tag = row[0].replace('_', ' ')
            if 'background' in tag:
                continue
            tags.append(tag)
            data3[tag] = np.asarray(row[1:]).astype(np.float)
    return data3, tags


def read_csv_result(filename):
    with open(filename, newline='') as csvfile:
        info = csv.reader(csvfile, delimiter=',', quotechar='|')
        tags1 = info.__next__()
        data1 = []
        for i, row in enumerate(info):
            row = [r.replace('"', '') for r in row]  # remove the \" char from saving to csv
            data1.append(np.asarray(row).astype(np.float))
        data1 = np.asarray(data1)
        data1_dict = {}
        for i, tag in enumerate(tags1):
            data1_dict[tag] = data1[:, i]
        data1 = data1_dict
    return data1, tags1


if __name__ == "__main__":
    # read csv file

    fig = plt.figure(constrained_layout=True, figsize=(10.8, 8.4))
    gs = GridSpec(2, 1, figure=fig, hspace=0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    filenames = ['../../results/baseline_ASD.csv', '../../results/learned_ASD.csv', '../../results/vxm_asd.csv']
    asd_pre_registration = {'brain stem': 1.85,
                            'left accumbens': 1.20, 'right accumbens': 1.13,
                            'left amygdala': 2.18, 'right amygdala': 1.44,
                            'left caudate': 1.37, 'right caudate': 1.44,
                            'left hippocampus': 1.45, 'right hippocampus': 1.60,
                            'left pallidum': 1.56, 'right pallidum': 1.12,
                            'left putamen': 1.30, 'right putamen': 1.02,
                            'left thalamus': 0.90, 'right thalamus': 0.67}
    draw_boxplot(filenames, pre_reg_value=None, ax=ax1, title='average surface distance (mm)', on_top=True)

    filenames = ['../../results/baseline_DSC.csv', '../../results/learned_DSC.csv', '../../results/vxm_dice.csv']
    dsc_pre_registration = {'brain stem': 0.815,
                            'left accumbens': 0.593, 'right accumbens': 0.653,
                            'left amygdala': 0.335, 'right amygdala': 0.644,
                            'left caudate': 0.705, 'right caudate': 0.813,
                            'left hippocampus': 0.708, 'right hippocampus': 0.665,
                            'left pallidum': 0.673, 'right pallidum': 0.794,
                            'left putamen': 0.772, 'right putamen': 0.812,
                            'left thalamus': 0.896, 'right thalamus': 0.920}
    draw_boxplot(filenames, pre_reg_value=None, ax=ax2, title='Dice score')


    fig.savefig('../../results/boxplot.pdf')
    fig.show()

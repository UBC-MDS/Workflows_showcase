"""
Author: Zeliha Ural Merpez

Created Date: Nov 25, 2020

Purpose: To print the given dataframe in a nice format, that is easy to save

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def render_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    
    """[Taken from ref: https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure]
       [Prints given dataframe in a nice format, that is easy to save]
    Parameters
    ----------
    data : [data frame]
        [data frame]
    col_width : float, optional
        [column width], by default 3.0
    row_height : float, optional
        [row height], by default 0.625
    font_size : int, optional
        [font size], by default 14
    header_color : str, optional
        [header color], by default '#40466e'
    row_colors : list, optional
        [row color], by default ['#f1f1f2', 'w']
    edge_color : str, optional
        [edge color], by default 'w'
    bbox : list, optional
        [bbox ], by default [0, 0, 1, 1]
    header_columns : int, optional
        [header columns], by default 0
    ax : [type], optional
        [plotting table, by default None

    Returns
    -------
    [object]
        [figure]
    """
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax
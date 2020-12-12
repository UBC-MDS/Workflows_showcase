"""
Author: Craig McLaughlin

Date: Dec 9, 2020

This is a utility library script for generic project utils.
"""

import os
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import altair as alt
from vega_datasets import data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pylab import savefig

def validate_inputs(input_file_path, output_dir_path):
    """
    Validates input argument paths.
    Parameters:
    -----------
    input_file_path : str
        input path to be verified

    output_file_path : str
        output path to be verified

    Returns:
    -----------
    None
    """
    if not os.path.isfile(input_file_path):
        print(f"Cannot locate input file: {input_file_path}")
        sys.exit()

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    if not os.path.exists(output_dir_path + "/figures"):
        os.makedirs(output_dir_path + "/figures")
    if not os.path.exists(output_dir_path + "/tables"):
        os.makedirs(output_dir_path + "/tables")
    if not os.path.exists(output_dir_path + "/models"):
        os.makedirs(output_dir_path + "/models")
    assert os.path.exists(output_dir_path), f"Invalid output path: {output_dir_path}"
    assert os.path.exists(output_dir_path + "/figures"), f"Invalid output path: {output_dir_path}/figures"
    assert os.path.exists(output_dir_path + "/tables"), f"Invalid output path: {output_dir_path}/tables"
    assert os.path.exists(output_dir_path + "/models"), f"Invalid output path: {output_dir_path}/models"


def read_input_file(input_file_path, verbose):
    """
    Reads input file path and reads cleaned data.
    Parameters:
    -----------
    input_file_path : str
        input path to be verified

    Returns:
    -----------
    pandas.DataFrame
        if path is valid and verified
    """
    try:
        input_df = pd.read_csv(input_file_path)
        if verbose: print('Input filename path is valid.')
    except:
        print(input_file_path + ': Input filename path is not valid. Please check!')
        sys.exit()

    # TODO possibly move this to a config or test script to remove magic values
    combined_columns = ['name',
                        'id',
                        'align',
                        'eye',
                        'hair',
                        'sex',
                        'gsm',
                        'appearances',
                        'first_appearance',
                        'year',
                        'publisher',
                        'is_common',
                        'name_len',
                        'has_last_name',
                        'appear_per_yr']

    if not all([item in input_df.columns for item in combined_columns]):
        print(input_file_path + " should contain these columns: " + str(combined_columns))
        sys.exit()

    return input_df


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

def save_img(data_frame, output_dir, file_name, filename_prefix = ""):
        data_frame.to_pickle(output_dir + "/tables/" + filename_prefix + file_name + ".pkl")
        fig_1, ax_1 = render_table(data_frame, header_columns=0, col_width=5)
        fig_1.savefig(output_dir + "/figures/" + filename_prefix + file_name)

def save_img_large(data_frame, output_dir, file_name, filename_prefix = ""):
        data_frame.to_pickle(output_dir + "/tables/" + filename_prefix + file_name + ".pkl")
        fig_1, ax_1 = render_table(data_frame, header_columns=0, col_width=9)
        fig_1.savefig(output_dir + "/figures/" + filename_prefix + file_name)

def save_matrix(matrix, output_folder, file_name, filename_prefix = ""):
    matrix.figure_.savefig(output_folder + "/figures/" + filename_prefix + file_name)
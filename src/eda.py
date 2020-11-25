"""
This script reads combined data and generates images and tables to be used in further analysis.

Usage: eda.py -i=<input> -o=<output> [-v]

Options:
-i <input>, --input <input>     Local raw data csv file directory
-o <output>, --output <output>  Local output filename and path for preprocessed csv
[-v]                            Report verbose output of dataset retrieval process
"""

# example to run: python src/eda.py -i "data/processed/clean_characters.csv" -o "img"

import sys
import os
import numpy as np
import pandas as pd
#import altair as alt
import matplotlib.pyplot as plt
#import seaborn as sns
from pathlib import Path
from docopt import docopt
from pylab import savefig
#from selenium import webdriver
import matplotlib.pyplot as plt
print("start")
args = docopt(__doc__)
#from webdriver_manager.chrome import ChromeDriverManager

#from altair import pipe, limit_rows, to_values
#from pandas_profiling.report.presentation.core import Table, Container

#t = lambda data: pipe(data, limit_rows(max_rows=20000), to_values)
#alt.data_transformers.register('custom', t)
#alt.data_transformers.enable('custom')

IMAGE_FOLDER = 'images'
#DATA_FOLDER = 'data'

def read_input_file(input_file_path):
    """
    Validates input file path.
    Parameters:
    -----------
    input_file_path : str
        input path to be verified
        
    Returns:
    -----------
    pandas.DataFrame
        if path is valid and verified
    """
    if not os.path.isfile(input_file_path):
        print("Input file does not exist.")
        sys.exit()

    try:
        data_frame = pd.read_csv(input_file_path, index_col=0)
        print('Path is valid')
    except:
        print(input_file_path + 'Path is not valid. Please check ')
        sys.exit()

    combined_columns = ['name', 'id', 'align', 'eye', 'hair', 'sex', 'gsm','appearances', 'first_appearance', 'year', 'publisher']

    if not all([item in data_frame.columns for item in combined_columns]):
        print(input_file_path + " should contain these columns: " + str(combined_columns))
        sys.exit()

    print('Returning data frame')
    return data_frame

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    """[Taken from ref: https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure]
       [Prints given data in a nice format, that is easy to save]
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

def generate_dataset_overview(data_frame, output_folder, file_name):
    """
    Generates an overview of the dataset.
    Also saves resulting table as file in given output folder.
    Parameters:
    -----------
    data_frame : pandas.DataFrame
        input path to be verified
    output_folder : str
        output folder path to save the chart
    file_name : str
        file name for generated chart image
        
    Returns:
    -----------
    str
        saved file path
    """
    data_overview = [
            {"Dataset": "Number of features", "Value": len(data_frame.columns)},
            {"Dataset": "Number of characters", "Value": len(data_frame)},
            {"Dataset": "Number of Missing cells", "Value": (data_frame.isnull()).sum().sum()},
            {"Dataset": "Percentage of Missing cells", "Value": round((data_frame.isnull()).sum().sum()/data_frame.size*100, 2)}
        ]
    overview_frame = pd.DataFrame(data_overview)
    fig_1, ax_1 = render_mpl_table(overview_frame, header_columns=0, col_width=5)

    return fig_1.savefig(output_folder +"/"+ file_name)
def generate_feature_overview(data_frame, output_folder, file_name):
    """
    Generates an overview of the features in dataset.
    Also saves resulting table as file in given output folder.
    Parameters:
    -----------
    data_frame : pandas.DataFrame
        input path to be verified
    output_folder : str
        output folder path to save the chart
    file_name : str
        file name for generated chart image
        
    Returns:
    -----------
    object
        saved .png file
    """
    dict2 = dict()
    dict3=dict()

    for col in data_frame.columns:
        dict2[col]=len(data_frame)-data_frame[col].isnull().sum()
        dict3[col]=len(list(data_frame[col].unique()))
    data_frame2=pd.DataFrame([dict3, dict2]).T.reset_index()
    data_frame2.columns=["Features","Dictinct Class", "Non-Null Count"]
    data_frame2["Missing Percentage"]=round((len(data_frame) - data_frame2["Non-Null Count"])/len(data_frame)*100,2)

    fig_2, ax_2 = render_mpl_table(data_frame2, header_columns=0, col_width=3)

    return fig_2.savefig(output_folder +"/"+ file_name)



def maint(input_file_path, output_folder_path):
    print(input_file_path)
    data_frame = read_input_file(input_file_path)
    print("generating")
    generate_dataset_overview(data_frame, output_folder_path, "Table_01")
    generate_feature_overview(data_frame, output_folder_path, "Table_02")
    print("done")


if __name__ == "__main__":
    print("101")
    
    input_file = args["--input"]
    output_dir = args["--output"]
    verbose = args["-v"]
    print("102")
    maint(input_file, output_dir)
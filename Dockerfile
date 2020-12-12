# Docker file for the Workflows
# Author: Zeliha Ural Merpez
# Nov.-Dec. 2020

FROM continuumio/miniconda3

# Install system dependencies. 
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        graphviz

# Install python dependencies via conda
RUN conda install -y -c conda-forge \
    altair \
    altair_saver \
    docopt \
    jupyterlab \
    nltk \
    numpy \
    pandas \
    scikit-learn \
    pandas-profiling \
    vega \
    vega_datasets \
    lxml 

RUN conda -update -y --all

# Install altair saver dependencies
RUN conda install -y -c conda-forge/label/gcc7 selenium && \
    conda install -y -c conda-forge python-chromedriver-binary && \
    conda install -y -c conda-forge matplotlib=3.3.*

# Install pip
RUN pip install -U webdriver-manager \
    psutil \
    xgboost \
    lightgbm \ 
    catboost \ 
    git+git://github.com/mgelbart/plot-classifier.git

# Docker file for the Workflows
# Author: Zeliha Ural Merpez
# Nov.-Dec. 2020

FROM continuumio/miniconda3

# Install system dependencies. 
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        graphviz

# We need wget to set up the PPA and xvfb to have a virtual screen and unzip to install the Chromedriver
RUN apt-get install -y wget xvfb unzip

# Set up the Chrome PPA
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list

# Update the package list and install chrome
RUN apt-get update -y
RUN apt-get install -y google-chrome-stable

# Set up Chromedriver Environment variables
ENV CHROMEDRIVER_VERSION 2.19
ENV CHROMEDRIVER_DIR /chromedriver
RUN mkdir $CHROMEDRIVER_DIR

# Download and install Chromedriver
RUN wget -q --continue -P $CHROMEDRIVER_DIR "http://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip"
RUN unzip $CHROMEDRIVER_DIR/chromedriver* -d $CHROMEDRIVER_DIR

# Put Chromedriver into the PATH
ENV PATH $CHROMEDRIVER_DIR:$PATH

# Install python dependencies via conda
RUN conda install -y -c conda-forge \
    altair=4.1.* \
    altair_saver=0.5.* \
    docopt=0.6.* \
    jupyterlab=2.2.* \
    nltk=3.4.* \
    numpy=1.19.* \
    pandas=1.1.* \
    scikit-learn \
    pandas-profiling \
    vega \
    vega_datasets \
    lxml 

RUN conda update -y --all

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

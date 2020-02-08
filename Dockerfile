# Docker file for DSCI_522_GROUP_312
# Eithar Elbasheer, Sam Solomon, George Thio
# January 2020

# Use continuumio/anaconda3 as base image
FROM continuumio/anaconda3 

# Install Chromium
RUN apt-get update && \
    apt install -y chromium && \
    apt-get install -y libnss3 && \
    apt-get install unzip

# Install Chromedriver
RUN wget -q "https://chromedriver.storage.googleapis.com/79.0.3945.36/chromedriver_linux64.zip" -O /tmp/chromedriver.zip && \
    unzip /tmp/chromedriver.zip -d /usr/bin/ && \
    rm /tmp/chromedriver.zip && chown root:root /usr/bin/chromedriver && chmod +x /usr/bin/chromedriver

# Install R and packages
RUN conda install -c r r-tidyverse && \
    conda install -c r r-tidyr && \
    conda install -c anaconda requests&& \
    conda install -c r r-caret && \
    conda install -c conda-forge r-checkmate && \
    conda install -c r r-testthat && \
    conda install -c conda-forge r-readr && \
    conda config --add channels conda-forge && \
    conda install r-readr

# Install Altair
RUN conda install -y -c conda-forge altair && \
    conda install -y vega_datasets && conda install -y selenium

# Install docopt Python package
RUN conda install -c anaconda docopt==0.6.2

# Install docopt R package
RUN conda install -y -c conda-forge r-docopt==0.6.1

# Put Anaconda Python in PATH
ENV PATH="/opt/conda/bin:${PATH}"
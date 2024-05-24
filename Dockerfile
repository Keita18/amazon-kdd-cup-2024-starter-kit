FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    USER_NAME=aicrowd \
    HOME_DIR=/home/aicrowd \
    CONDA_DIR=/home/aicrowd/.conda \
    PATH=/home/aicrowd/.conda/bin:${PATH} \
    SHELL=/bin/bash

# Install system dependencies and clean up in one layer
COPY apt.txt /tmp/apt.txt
RUN apt -qq update && apt -qq install -y --no-install-recommends `cat /tmp/apt.txt | tr -d '\r'` locales wget build-essential \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/cache/apt/* /var/lib/apt/lists/* \
    && apt clean

# Set up user
RUN groupadd -g 1001 aicrowd && \
    useradd -m -s /bin/bash -u 1001 -g aicrowd -G sudo aicrowd

USER ${USER_NAME}
WORKDIR ${HOME_DIR}

# Install Miniconda and Python packages. You can change the python version by using another Miniconda. 
RUN wget -nv -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh \
    && bash miniconda.sh -b -p ${CONDA_DIR} \
    && . ${CONDA_DIR}/etc/profile.d/conda.sh \
    && conda install cmake -y \
    && conda clean -y -a \
    && rm -rf miniconda.sh

COPY --chown=1001:1001 requirements.txt ${HOME_DIR}/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
COPY --chown=1001:1001 requirements_eval.txt ${HOME_DIR}/requirements_eval.txt
RUN pip install -r requirements_eval.txt --no-cache-dir

## Add your custom commands below

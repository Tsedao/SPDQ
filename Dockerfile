# FROM ubuntu:18.04
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y htop python3-dev wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n rl python=3.7


# COPY ./model/ home/model/
# COPY ./configs/ home/configs/
# COPY ./environment/ home/environment/
# COPY ./utils/ home/utils/
# COPY ./requirements.txt home/
# COPY ./train.py home/
COPY . home/

RUN /bin/bash -c "cd home/ \
    && source activate rl \
    && conda install cudatoolkit=10.0 \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt"

WORKDIR home/

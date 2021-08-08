# FROM ubuntu:18.04
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y htop python3-dev wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n rl python=3.7


COPY ./model/ usr/src/model/
COPY ./configs/ usr/src/configs/
COPY ./environment/ usr/src/environment/
COPY ./utils/ usr/src/utils/
COPY ./requirements.txt usr/src/
COPY ./train.py usr/src/

RUN /bin/bash -c "cd usr/src/ \
    && source activate rl \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt"

WORKDIR /usr/src/

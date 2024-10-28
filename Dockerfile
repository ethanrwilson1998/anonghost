FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV TZ=Europe/Prague
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa -y && \
 apt-get update && apt-get install -y \
    wget \
    vim \
    python3.8 \
    python3.8-distutils \
    ffmpeg \
    libsm6 \
    libxext6 && \
apt-get clean && \
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && apt-get -y install gcc

RUN wget https://bootstrap.pypa.io/get-pip.py

RUN python3.8 get-pip.py

WORKDIR /ghost
COPY . .

RUN pip3 install --upgrade pip setuptools wheel && pip3 install -r requirements-docker.txt && pip3 install onnxruntime_gpu-1.9.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
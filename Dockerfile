ARG BASE_IMAGE=python:3.10-slim-buster
FROM $BASE_IMAGE as base
WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app"

##
# Install any runtime depenencies here
ENV RUNTIME_DEPENDENCIES="ffmpeg"

RUN apt-get update \
    && apt-get install -y software-properties-common wget gpg \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y libcudnn8 libcudnn8-dev libcublas-12-0 \
    && apt-get install -y $RUNTIME_DEPENDENCIES \
    && rm -rf /var/lib/apt/lists/*


ENV BUILD_DEPENDENCIES="build-essential"

COPY requirements.txt /app/requirements.txt


# INSTALL AUDIOWAVEFORM dependencies
RUN apt-get update && apt-get install -y git make cmake gcc g++ libmad0-dev \
  libid3tag0-dev libsndfile1-dev libgd-dev libboost-filesystem-dev \
  libboost-program-options-dev \
  libboost-regex-dev

# Install any build dpendencies depenencies here
RUN wget https://github.com/bbc/audiowaveform/releases/download/1.8.1/audiowaveform_1.8.1-1-10_amd64.deb
RUN dpkg -i audiowaveform_1.8.1-1-10_amd64.deb \
 && apt-get -f install -y
# RUN add-apt-repository -y ppa:chris-needham/ppa && apt-get update
RUN apt-get update \
    && apt-get install -y $BUILD_DEPENDENCIES \
    # && pip install --no-cache-dir -r requirements.txt \
&& apt-get remove -y $BUILD_DEPENDENCIES \
&& apt-get auto-remove -y \
&& rm -rf /var/lib/apt/lists/*

FROM base
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
WORKDIR /src
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

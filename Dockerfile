# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04

ARG WANDB_API_KEY
ARG WANDB_TAGS

ENV PATH="${PATH}:/home/docker/.local/bin"

# Set timezone
ENV TZ=Canada/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Configure Ray Tune results directory
ENV TUNE_RESULT_DIR="/home-local2/adtup.extra.nobkp/ray_results"

# Configure Weights & Biases
ENV WANDB_API_KEY=${WANDB_API_KEY}
ENV WANDB_DIR="/home-local2/adtup.extra.nobkp/wandb"
ENV WANDB_CACHE_DIR="/home-local2/adtup.extra.nobkp/.cache/wandb"
ENV WANDB_TAGS=${WANDB_TAGS}

RUN apt-get update && \
    apt-get install -y sudo

# Copy files to a new 'work' directory (replace X's with your UID and GID)
RUN mkdir -p /home/docker/workspace/
COPY --chown=XXXXX:XXX ./ /home/docker/workspace/

# Install Dependencies
RUN sudo apt-get update && \
    sudo apt-get install -y openssh-client git python3.9 python3-pip
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh pip3 install -r /home/docker/workspace/requirements.txt

# Add (and switch to) new 'docker' user (replace X's with your UID and GID)
RUN groupadd -g XXX usergroup
RUN adduser --disabled-password --gecos '' --uid XXXXX --gid XXX docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER docker

# Transfer ownership of home directory
RUN sudo chown -hR docker /home/docker

# Create mount point for large file storage drive
RUN sudo mkdir -p /home-local2/adtup.extra.nobkp && \
    sudo chown -R docker /home-local2/adtup.extra.nobkp

# Set the working directory
WORKDIR /home/docker/workspace/

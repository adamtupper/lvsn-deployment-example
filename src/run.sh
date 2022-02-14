#!/bin/bash

cd src

# Run experiment
python3 run.py \
    --cpus-per-trial 4 \
    --project-name resnet18-cifar10-baselines  # Used for Weights & Biases
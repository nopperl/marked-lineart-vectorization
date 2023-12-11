#!/bin/sh
set -euo pipefail

mkdir -p data/download
mkdir -p data/clean
mkdir -p data/processed
wget https://cragl.cs.gmu.edu/sketchbench/Benchmark_Dataset.zip --no-check-certificate -P data/download
unzip data/download/Benchmark_Dataset.zip -d data/download
mkdir -p data/clean/sketchbench
cp data/download/Benchmark_Dataset/GT/*norm_cleaned.svg data/clean/sketchbench

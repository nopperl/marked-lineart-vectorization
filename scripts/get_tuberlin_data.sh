#!/bin/sh
set -euo pipefail

mkdir -p data/download
mkdir -p data/clean
wget https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip --no-check-certificate -P data/download
unzip data/download/sketches_svg.zip -d data/clean/tuberlin
rm data/clean/tuberlin/svg/filelist*

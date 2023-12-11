#!/bin/sh
set -eu

gt_dir=outputs/ground-truth/svg
mkdir -p $gt_dir 

cp -r data/processed/tonari-black-tonari-blue-tonari-red-tonari-lime-sketchbench-black-tuberlin-black-512-0.512/test/images $gt_dir/512-0.512
cp -r data/processed/sketchbench/tonari-black-tonari-blue-tonari-red-tonari-lime-sketchbench-black-tuberlin-black-512-0.512/test/images/* $gt_dir/512-0.512
cp -r data/processed/tonari-black-tonari-blue-tonari-red-tonari-lime-sketchbench-black-tuberlin-black-1024-1.024/test/images $gt_dir/1024-1.024
cp -r data/processed/sketchbench/tonari-black-tonari-blue-tonari-red-tonari-lime-sketchbench-black-tuberlin-black-1024-1.024/test/images/* $gt_dir/1024-1.024
mkdir -p $gt_dir/binarized
cp -r $gt_dir/512-0.512 $gt_dir/binarized
cp -r $gt_dir/1024-1.024 $gt_dir/binarized
find $gt_dir/binarized -type f -name '*.png' -exec scripts/binarize.sh {} \;

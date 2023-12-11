#!/bin/sh
set -eu

model=${1:-model.onnx}
stroke_width="$2"
svg_output_dir="$3"
runtime_output_dir="$4"
filepath="$5"
thr="${6:-0}"
filename=$(basename "$filepath")
mkdir -p "$svg_output_dir"
mkdir -p "$runtime_output_dir"
/usr/bin/time -o "$runtime_output_dir"/"${filename%.png}.txt" scripts/onnx_inference.py "$model" --output "$svg_output_dir" -b $thr --stroke_width "$stroke_width" --input_images "$filepath"

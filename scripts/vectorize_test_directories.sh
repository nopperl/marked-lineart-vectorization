#!/bin/sh
set -eu

model=${1:-model.onnx}
gt_dir=outputs/ground-truth
output_dir=outputs/marked
find $gt_dir/512-0.512 -type f -name '*.png' -exec scripts/test_lineart.sh $model 0.512 $output_dir/svg/512-0.512 $output_dir/runtime/512-0.512 {} \;
find $gt_dir/512-0.512 -type f -name '*.png' -exec scripts/test_lineart.sh $model 0.512 $output_dir/svg/binarized/512-0.512 $output_dir/runtime/binarized/512-0.512 {} .4 \;
find $gt_dir/1024-1.024 -type f -name '*.png' -exec scripts/test_lineart.sh $model 1.024 $output_dir/svg/1024-1.024 $output_dir/runtime/1024-1.024 {} \;
find $gt_dir/1024-1.024 -type f -name '*.png' -exec scripts/test_lineart.sh $model 1.024 $output_dir/svg/binarized/1024-1.024 $output_dir/runtime/binarized/1024-1.024 {} .4 \;

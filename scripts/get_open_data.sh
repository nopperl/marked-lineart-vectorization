#!/bin/sh
set -euo pipefail

scripts/get_tuberlin_data.sh
scripts/get_sketchbench_data.sh

scripts/sanitize-svg.sh data/clean/tuberlin
scripts/sanitize-svg.sh data/clean/sketchbench

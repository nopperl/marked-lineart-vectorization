#!/bin/sh
set -euo pipefail

find "$1" -type f -name '*.svg' -exec sh -c 'cairosvg "$0" -o "${0%.svg}".png -b white' {} \;
find "$1" -type f -name '*.svg' -exec scripts/to_flat_cubic.py {} \;

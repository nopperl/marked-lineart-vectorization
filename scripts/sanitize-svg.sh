#!/bin/sh
set -euo pipefail

find $1 -type f -name *.svg -exec bin/inkscape --actions "select-all:groups; SelectionUnGroup; select-all:groups; SelectionUnGroup; export-filename: {}; export-plain-svg; export-do;" {} \;

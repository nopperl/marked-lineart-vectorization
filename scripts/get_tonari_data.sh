#!/bin/sh
set -euo pipefail

mkdir -p data/clean/tonari
find /home/dlmain/deep-anime/horizon/files/douga/ -type f -name '*.svg' -exec cp {} data/clean/tonari \;


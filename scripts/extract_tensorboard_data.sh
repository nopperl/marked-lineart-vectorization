#!/bin/sh
set -euo pipefail

find logs -type d -name 'tensorboard' | parallel python scripts/extract_tensorboard_data.py --log_dir '{}'

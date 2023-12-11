#!/bin/sh
set -eu

if [ -f "$1" ]; then
	convert -threshold 90% "$1" "$1"
elif [ -d "$1" ]; then
	find "$1" -type f -name '*.png' -exec convert -threshold 90% {} {} \;
fi

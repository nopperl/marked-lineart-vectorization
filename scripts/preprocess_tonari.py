#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from os import makedirs
from os.path import basename, normpath, join, splitext

from cairosvg import svg2png
from svgelements import Color
from svgpathtools import CubicBezier, svg2paths2, wsvg, QuadraticBezier, Line, Path

from scripts.preprocess_clean_svg import convert_svg_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="data/clean/tonari")
    parser.add_argument("-o", "--output_dir_root", default="data/processed")
    parser.add_argument("-m", "--max_size", type=int ,default=512)
    parser.add_argument("-f", "--fallback_stroke_width", type=float, default=0.512)
    parser.add_argument("-s", "--strokes", nargs="+", default=["black", "blue", "red", "lime"])
    parser.add_argument("-b", "--black", action="store_true", help="whether to convert all strokes to black")
    parser.add_argument("--hierarchy", action="store_true", default=False)
    args = parser.parse_args()
    for stroke in args.strokes: 
        convert_svg_data(input_dir=args.input_dir, output_dir_root=args.output_dir_root, max_size=args.max_size, fallback_stroke_width=args.fallback_stroke_width, stroke=stroke, hierarchy=args.hierarchy, filter_stroke=True, output_stroke="black" if args.black else stroke)

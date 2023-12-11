#!/usr/bin/env python
from argparse import ArgumentParser
from os import makedirs
from os.path import basename, join

from svgpathtools import svg2paths2, wsvg


def visualize_order(svg_file, output_dir="report/figures"):
    makedirs(output_dir, exist_ok=True)
    output_filename = join(output_dir, basename(svg_file))
    paths, path_attributes, svg_attributes = svg2paths2(svg_file)
    no_curves = len(path_attributes)
    colors = ["red", "green", "blue", "brown", "black", "magenta"]  # + orange, yellow
    for i in range(no_curves):
        path_attributes[i].update({"stroke": colors[i % len(colors)]})
    wsvg(paths, attributes=path_attributes, svg_attributes=svg_attributes, filename=output_filename)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--svg_file", default="data/clean/sketchbench/Art_freeform_AG_05_Branislav Mirkovic_norm_cleaned.svg")
    args = parser.parse_args()
    visualize_order(args.svg_file)

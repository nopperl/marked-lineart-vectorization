#!/usr/bin/env python
from argparse import ArgumentParser
from glob import glob
from os import makedirs
from os.path import basename, join, splitext

from svgpathtools import svg2paths2, wsvg


def resolve_path(path, recursive=True):
    if type(path) in [Line, QuadraticBezier, CubicBezier]:
        yield path
    elif type(path) is Path:
        if not recursive:
            if len(path._segments) == 1:
                segment = next(iter(path))
                yield segment
        else:
            for segment in iter(path):
                yield segment
    else:
        yield path


def resolve_all_paths(paths, recursive=True):
    return list(chain(*(resolve_path(path, recursive=recursive) for path in paths)))


def merge_tonari_segments(black_svg, output=None):
    segment_svgs = glob(black_svg.replace("black", "*"))
    paths = []
    path_attributes = []
    for segment_svg in segment_svgs:
        if "full" in basename(segment_svg):
            continue
        stroke = segment_svg.split("-")[-1].split("_")[0]
        segment_paths, segment_path_attributes, svg_attributes = svg2paths2(segment_svg)
        paths += segment_paths
        for attribute in segment_path_attributes:
            attribute["stroke"] = stroke
        path_attributes += segment_path_attributes
    output_filepath = black_svg.replace("black", "full")
    if output is not None:
        if isdir(output):
            makedirs(output, exist_ok=True)
            output_filepath = join(output, output_filepath)
        elif type(output) is str and output.endswith(".svg"):
            output_filepath = output
    print(output_filepath)
    wsvg(paths=paths, attributes=path_attributes, svg_attributes=svg_attributes, filename=output_filepath)


def main():
    parser = ArgumentParser(description="Inference script for the marked curve reconstruction model in ONNX format.")
    parser.add_argument("-i", "--input_images", nargs="*", metavar="FILE", help="one or multiple paths to raster images that should be vectorized.")
    parser.add_argument("-d", "--input_dir", metavar="DIR", help="path to a directory of raster images that should be vectorized.")
    parser.add_argument("-o", "--output", default=None, help="optional output directory or file")
    args = parser.parse_args()
        
    input_images = args.input_images
    if args.input_dir is not None and isdir(args.input_dir):
        input_images.extend(glob(args.input_dir + "tonari-black*.svg"))
    for input_image in input_images:
        merge_tonari_segments(input_image, args.output)

if __name__ == "__main__":
    main()

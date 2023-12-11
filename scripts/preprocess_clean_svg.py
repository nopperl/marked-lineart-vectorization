#!/usr/bin/env python3
from argparse import ArgumentParser
from os import makedirs
from os.path import basename, normpath, join, splitext
from glob import glob

from cairosvg import svg2png
from svgelements import Color
from svgpathtools import CubicBezier, svg2paths2, wsvg, QuadraticBezier, Line, Path


def convert_svg_data(input_dir, output_dir_root, max_size, fallback_stroke_width, stroke, hierarchy, filter_stroke, output_stroke=None):
    filenames = glob(join(input_dir, "**/*.svg"), recursive=True)
    output_dir = join(output_dir_root, f"{basename(normpath(input_dir))}-{stroke}-{max_size}-{fallback_stroke_width}")
    stroke = Color(stroke)
    if output_stroke is None:
        output_stroke = stroke
    else:
        output_stroke = Color(output_stroke)
    makedirs(output_dir, exist_ok=True)
    for filename in filenames:
        print(filename)
        paths, path_attributes, svg_attributes = svg2paths2(filename)
        original_width = float(svg_attributes["viewBox"].split()[2]) if "viewBox" in svg_attributes else float(svg_attributes["width"].replace("px", ""))
        original_height = float(svg_attributes["viewBox"].split()[3]) if "viewBox" in svg_attributes else float(svg_attributes["height"].replace("px", ""))
        if original_width > original_height:
            width_scale = max_size / original_width
            height_scale = width_scale
        else:
            height_scale = max_size / original_height
            width_scale = height_scale
        curves = []
        output_attributes = []
        stroke_width = None
        for i, path in enumerate(paths):
            attributes = path_attributes[i]
            if filter_stroke and ("stroke" not in attributes or Color(attributes["stroke"]) != stroke):
               continue
            if stroke_width is None and "stroke-width" in attributes:
                stroke_width = float(attributes["stroke-width"]) * width_scale
            new_curves = (curve.scaled(sx=width_scale, sy=height_scale) for curve in path if type(curve) in [Line, QuadraticBezier, CubicBezier])
            if hierarchy:
                path_curves = list(new_curves)
                if len(path_curves) > 0:
                    curves.append(Path(*path_curves))
            else:
                curves.extend(new_curves)
        if len(curves) < 20: # discard images with small amount of curves
            continue
        if stroke_width is None:
            stroke_width = fallback_stroke_width
        output_attributes = [{"stroke": output_stroke, "stroke-width": stroke_width, "stroke-linecap": "round", "fill": "none"}] * len(curves)
        svg_attributes = {"width": f"{original_width * width_scale}", "height": f"{original_height * height_scale}px"}
        name = join(output_dir, splitext(basename(filename))[0])
        svg_filename = name + ".svg"
        png_file = name + ".png"
        wsvg(curves, attributes=output_attributes, svg_attributes=svg_attributes, filename=svg_filename)
        with open(svg_filename, "r") as file:
            svg2png(file_obj=file, write_to=png_file, background_color="white")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dirs", nargs="+", default=["data/clean/sketchbench", "data/clean/tuberlin"])
    parser.add_argument("-o", "--output_dir_root", default="data/processed")
    parser.add_argument("-m", "--max_size", type=int, default=512)
    parser.add_argument("-f", "--fallback_stroke_width", type=float, default=0.512)
    parser.add_argument("-s", "--stroke", default="black")
    parser.add_argument("--hierarchy", action="store_true", default=False)
    parser.add_argument("--filter_stroke", action="store_true", default=False)
    args = parser.parse_args()
    for input_dir in args.input_dirs: 
        convert_svg_data(input_dir=input_dir, output_dir_root=args.output_dir_root, max_size=args.max_size, fallback_stroke_width=args.fallback_stroke_width, stroke=args.stroke, hierarchy=args.hierarchy, filter_stroke=args.filter_stroke)

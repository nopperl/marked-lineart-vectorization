#!/usr/bin/env python3
from random import randint

from cairosvg import svg2png
from svgpathtools import CubicBezier, wsvg

MIN_CURVES = 5
MAX_CURVES = 15
NUM_IMAGES = 500
OUTPUT_DIR = "data/bezier-multiple9/test"
IMG_WIDTH = 256
IMG_HEIGHT = 256

for im in range(NUM_IMAGES):
    num_curves = randint(MIN_CURVES, MAX_CURVES)
    curves = []
    attributes = []
    for c in range(num_curves):
        points = [complex(randint(-IMG_WIDTH*.5, IMG_WIDTH * 1.5), randint(-IMG_HEIGHT*.5, IMG_HEIGHT * 1.5)) for _ in range(3)]
        points.insert(0, points[0])
        curves.append(CubicBezier(*points))
        attributes.append({"stroke": "#000000", "stroke-width": "3.0", "fill": "none"})
    # render and save image
    svg_attributes = {"width": f"{IMG_WIDTH}px", "height": f"{IMG_HEIGHT}px"}
    name = OUTPUT_DIR + "/" + str(im).zfill(len(str(NUM_IMAGES)))
    svg_filename = name + ".svg"
    png_file = name + ".png"
    wsvg(curves, attributes=attributes, svg_attributes=svg_attributes, filename=svg_filename)
    with open(svg_filename, "r") as file:
        svg2png(file_obj=file, write_to=png_file, background_color="white")

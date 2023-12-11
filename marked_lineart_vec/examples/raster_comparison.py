#!/usr/bin/env python
from itertools import chain

from cairosvg import svg2png
from svgpathtools import svg2paths2
import torch
from torchvision.utils import save_image

from marked_lineart_vec.util import all_paths_as_cubic
from marked_lineart_vec.render import raster_bezier

in_filename = "data/processed/sketchbench-black-512-0.512/Art_freeform_AG_03_Branislav Mirkovic_norm_cleaned.svg"
out_basename = "report/figures/Art_freeform_AG_03_Branislav Mirkovic_norm_cleaned"

out_filename = out_basename + "_cairosvg.png"
svg2png(url=in_filename, write_to=out_filename)

out_filename = out_basename + "_diffvg.png"
paths_pre, path_attributes, svg_attributes = svg2paths2(in_filename)
width = round(float(svg_attributes["width"].replace("px", "")))
height = round(float(svg_attributes["height"].replace("px", "")))
paths = all_paths_as_cubic(paths_pre, recursive=True)
points = list(chain(*(path.bpoints() for path in paths)))
points = torch.tensor([(point.real, point.imag) for point in points]).float()
points = points.view(-1, 4, 2)
points[:, :, 0] /= width
points[:, :, 1] /= height
image, _ = raster_bezier(points, image_width=width, image_height=height, stroke_width=0.512)
save_image(image.view(3, height, width), out_filename)

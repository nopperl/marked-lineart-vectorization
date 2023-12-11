#!/usr/bin/env python
from itertools import chain

from cairosvg import svg2pdf
import numpy as np
from svgpathtools import svg2paths2, wsvg
import torch

from marked_lineart_vec.datasets import transform_paths
from marked_lineart_vec.util import all_paths_as_cubic, tensor_to_bezier

in_filename = "data/processed/tonari-black-512-0.512/007AD_DOU_27.svg"
out_basename = "report/figures/007AD_DOU_27"

paths_pre, path_attributes, svg_attributes = svg2paths2(in_filename)
width = float(svg_attributes["viewBox"].split()[2]) if "viewBox" in svg_attributes else float(svg_attributes["width"].replace("px", ""))
height = float(svg_attributes["viewBox"].split()[3]) if "viewBox" in svg_attributes else float(svg_attributes["height"].replace("px", ""))
paths = all_paths_as_cubic(paths_pre, recursive=True)

out_filename = out_basename + "_dropout.svg"
paths_dropout = transform_paths(paths, width, height, dropout=True, reverse=False, flip=False, p=1)
wsvg(paths_dropout, attributes=path_attributes, svg_attributes=svg_attributes, filename=out_filename)
svg2pdf(url=out_filename, write_to=out_filename.replace(".svg", ".pdf")) 

out_filename = out_basename + "_reverse.svg"
paths_reverse = transform_paths(paths, width, height, dropout=False, reverse=True, flip=False, p=1)
wsvg(paths_reverse, attributes=path_attributes, svg_attributes=svg_attributes, filename=out_filename)
svg2pdf(url=out_filename, write_to=out_filename.replace(".svg", ".pdf")) 

out_filename = out_basename + "_flip.svg"
paths_flip = transform_paths(paths, width, height, dropout=False, reverse=False, flip=True, p=1)
wsvg(paths_flip, attributes=path_attributes, svg_attributes=svg_attributes, filename=out_filename)
svg2pdf(url=out_filename, write_to=out_filename.replace(".svg", ".pdf")) 

out_filename = out_basename + "_rotate.svg"
points = list(chain(*(path.bpoints() for path in paths)))
points = torch.tensor([(point.real, point.imag) for point in points]).float()
points = points.view(-1, 4, 2)
swap = points[:, :, 0].clone()
points[:, :, 0] = points[:, :, 1]
points[:, :, 1] = swap
paths_rotate = []
for c in points:
    paths_rotate.append(tensor_to_bezier(c))
wsvg(paths_rotate, dimensions=(height, width), stroke_widths=[0.512] * len(paths_rotate), filename=out_filename)
svg2pdf(url=out_filename, write_to=out_filename.replace(".svg", ".pdf"))
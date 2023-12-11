#!/usr/bin/env python
import argparse
from os import makedirs
from os.path import join

import torch
from yaml import safe_load

from marked_lineart_vec.datasets import GeneratedRasterVectorDataset

parser = argparse.ArgumentParser(description='Generic train script for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/iterative.yaml')
parser.add_argument("--output", "-o", default="data/fixed")
parser.add_argument("--samples", "-s", default=100)
parser.add_argument("--batch_size", "-b", default=16)

args = parser.parse_args()

with open(args.filename, 'r') as yaml_file:
    config = safe_load(yaml_file)

params = config["exp_params"]
torch.manual_seed(config['logging_params']['manual_seed'])
dataset = GeneratedRasterVectorDataset(num_images=args.samples,
                                                   min_paths=params["min_paths"],
                                                   max_paths=params["max_paths"],
                                                   img_width=params["img_size"],
                                                   img_height=params["img_size"],
                                                   num_points=params["points"],
                                                   stroke_width=params["radius"],
                                       batch_size=args.batch_size)
for nr in range(args.samples):
    (raster_img, remaining, remaining_points, canvas_img, canvas_points, stop), _ = next(iter(dataset))
    makedirs(join(args.output, str(nr)), exist_ok=True)
    torch.save(raster_img, join(args.output, str(nr), "raster_img.pt"))
    torch.save(remaining, join(args.output, str(nr), "remaining.pt"))
    torch.save(remaining_points, join(args.output, str(nr), "remaining_points.pt"))
    torch.save(canvas_img, join(args.output, str(nr), "canvas_img.pt"))
    torch.save(canvas_points, join(args.output, str(nr), "canvas_points.pt"))
    torch.save(stop, join(args.output, str(nr), "stop.pt"))

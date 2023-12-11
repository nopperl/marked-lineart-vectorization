#!/usr/bin/env python3
from argparse import ArgumentParser
from os.path import isdir

import numpy as np
import torch
import torchvision.utils as vutils
from torch.backends import cudnn
from torchvision.transforms.functional import rgb_to_grayscale

from marked_lineart_vec.datasets import RasterVectorDataset
from marked_lineart_vec.experiment import VAEExperiment
from marked_lineart_vec.render import raster_bezier_batch
from marked_lineart_vec.util import place_point_on_img, sample_black_pixel, save_as_svg, diff_remaining_img

parser = ArgumentParser(description='Test script for the marked reconstruction model')
parser.add_argument("--model", "-m", metavar="FIlE", help="path to the *ckpt file", required=True)
parser.add_argument("--data", "-d", metavar="DIR", help="path to a data directory if a different dataset from the one specified in the model should be used")
parser.add_argument("--radius", "-r", type=float, help="radius if it should be different from the one specified in the model")
parser.add_argument("--truth", "-t", action="store_true")
parser.add_argument("--seed", "-s", default=1234)

args = parser.parse_args()
model_path = args.model
use_ground_truth_canvas = args.truth

experiment = VAEExperiment.load_from_checkpoint(model_path, strict=False, map_location="cpu")
model = experiment.model
model.to("cpu")
model.eval()
params = experiment.params
torch.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.deterministic = True

if args.data and isdir(args.data):
    params["data_path"] = args.data

if args.radius:
    params["radius"] = args.radius

with torch.no_grad():
    print(params)
    dataset = RasterVectorDataset(num_points=params["points"], stroke_width=params["radius"], return_canvas=False, root=params['data_path'].replace("train", "test"), transform=experiment.data_transforms())
    dataset_loader = iter(dataset)
    (raster_img, remaining_points), labels = next(dataset_loader)
    num_paths = remaining_points.shape[0]
    if params.get("force_grayscale"):
        raster_img = rgb_to_grayscale(raster_img)
    if params.get("binarize"):
        raster_img[raster_img < params.get("binarization_threshold", 1)] = 0.
    vutils.save_image(raster_img.data, f"test.png")
    width = raster_img.shape[1]
    height = raster_img.shape[2]
    recons_points = None
    recons_img = torch.ones_like(raster_img)
    for nr_path in range(int(num_paths)):
        remaining_img = diff_remaining_img(raster_img, recons_img)
        try:
            mark = sample_black_pixel(torch.clamp(remaining_img, min=0))
        except:
            break
        canvas_img = remaining_img.clone()
        mark_real = mark.clone()
        mark_real[0] *= width
        mark_real[1] *= height
        canvas_img = place_point_on_img(canvas_img, mark_real).unsqueeze(0)
        points = model.generate(canvas_img, mark=mark.unsqueeze(0))
        points = points.unsqueeze(1)
        if recons_points is None:
            recons_points = points.to(points.device)
        else:
            recons_points = torch.cat((recons_points, points), dim=1).to(points.device)
        recons_img, _ = raster_bezier_batch(recons_points, image_width=params["img_size"], image_height=params["img_size"], stroke_width=params["radius"], mode="hard")
        recons_img = recons_img.squeeze(0)
        vutils.save_image(canvas_img.data, f"test-{nr_path + 1}-canvas.png")
        vutils.save_image(recons_img.data, f"test-{nr_path + 1}.png")

vutils.save_image(recons_img.data, "test-final.png")
recons_points = recons_points.squeeze(0)
recons_points[:, :, 0] *= width
recons_points[:, :, 1] *= height
save_as_svg(recons_points, "test-final.svg", width, height, stroke_width=params["radius"])

#!/usr/bin/env python3
from argparse import ArgumentParser

import torch
import torchvision.utils as vutils
from torch.nn.functional import mse_loss
from torchvision.transforms.functional import rgb_to_grayscale

from marked_lineart_vec.experiment import VAEExperiment
from marked_lineart_vec.datasets import GeneratedRasterVectorDataset
from marked_lineart_vec.render import raster_bezier_batch

parser = ArgumentParser(description='Test script for the iterative model')
parser.add_argument("--model", "-m", metavar="FIlE", help="path to the *ckpt file")
parser.add_argument("--truth", "-t", action="store_true")

args = parser.parse_args()
model_path = args.model
use_ground_truth_canvas = args.truth

experiment = VAEExperiment.load_from_checkpoint(model_path)
model = experiment.model
model.eval()
params = experiment.params

with torch.no_grad():
    dataset = GeneratedRasterVectorDataset(num_images=1,
                                           min_paths=1,
                                           max_paths=params["max_paths"],
                                           img_width=params["img_size"],
                                           img_height=params["img_size"],
                                           num_points=params["points"],
                                           stroke_width=params["radius"],
                                           batch_size=1,
                                           canvas_blur=False,
                                           canvas_noise=False)
    (raster_img, remaining_imgs, remaining_points, canvas_img, canvas_points, stop), labels = next(iter(dataset))
    vutils.save_image(raster_img.data, f"test.png")
    num_paths = remaining_points.shape[1]
    vutils.save_image(canvas_img.data, "test-0.png")
    recons_points = canvas_points.clone()
    if params.get("force_grayscale"):
        raster_img = rgb_to_grayscale(raster_img)
    for nr_path in range(num_paths):
        if params.get("force_grayscale"):
            rgb_to_grayscale(canvas_img)
        points = model.generate(raster_img, canvas=canvas_img, canvas_points=canvas_points)
        points = points.unsqueeze(1)
        recons_points = torch.cat((recons_points, points), dim=1).to(points.device)
        recons_img, _ = raster_bezier_batch(recons_points, image_width=params["img_size"], image_height=params["img_size"], stroke_width=params["radius"], mode="soft")
        if use_ground_truth_canvas:
            curve_idx = torch.argmin(mse_loss(points, remaining_points, reduction="none").mean(dim=[2, 3]))
            correct_curve = remaining_points[:, curve_idx].unsqueeze(1)
            remaining_points = remaining_points[:, [i for i in range(remaining_points.shape[1]) if i != curve_idx]]
            canvas_points = torch.cat((canvas_points, correct_curve), dim=1).to(points.device)
            canvas_img, _ = raster_bezier_batch(canvas_points, image_width=params["img_size"], image_height=params["img_size"], stroke_width=params["radius"], mode="soft")
        else:
            canvas_points = recons_points
            canvas_img = recons_img
        vutils.save_image(canvas_img.data, f"test-{nr_path + 1}-canvas.png")
        vutils.save_image(recons_img.data, f"test-{nr_path + 1}.png")

vutils.save_image(recons_img.data, "test-final.png")

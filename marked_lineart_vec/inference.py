#!/usr/bin/env python3
from argparse import ArgumentParser
from os import listdir, makedirs
from os.path import basename, isdir, join, splitext
from typing import Union

import numpy as np
from PIL import Image
import torch
from torchvision.io import read_image
from torchvision.utils import save_image
from torch.backends import cudnn
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor

from marked_lineart_vec.experiment import VAEExperiment
from marked_lineart_vec.models.marked_reconstruction import MarkedReconstructionModel
from marked_lineart_vec.render import raster_bezier_batch
from marked_lineart_vec.util import center_on_point, diff_remaining_img, place_point_on_img, reverse_center_on_point, sample_black_pixel, save_as_svg, switch_major


def set_seed(seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True


def setup_model(model_path, device="cuda"):
    experiment = VAEExperiment.load_from_checkpoint(model_path, strict=False, map_location=device)
    model = experiment.model
    model = model.to(device)
    model.eval()
    transforms = experiment.data_transforms()
    return model, transforms


def vectorize_image(input_image, model: Union[str, MarkedReconstructionModel], output=None, transforms=None, threshold_ratio=0.1, stroke_width=0.512, binarization_threshold=0, force_grayscale=False):
    if model is str:
        model, experiment_transforms = setup_model(model)
        if transforms is None:
            transforms = experiment_transforms
    if transforms is None:
        transforms = Compose([ToTensor()])
    device = next(model.parameters()).device
    with torch.no_grad():
        pil_image = Image.open(input_image).convert("RGB")
        original_width = pil_image.width
        original_height = pil_image.height
        if force_grayscale:
            transforms = Compose([transforms, Grayscale(num_output_channels=3)])
        raster_img = transforms(pil_image).to(device)
        if binarization_threshold > 0:
            raster_img[raster_img < binarization_threshold] = 0.
        width = raster_img.shape[2]
        height = raster_img.shape[1]
        curve_pixels = (raster_img < .5).sum()
        threshold = curve_pixels * threshold_ratio 
        print(f"Reconstruction candidate pixels: {curve_pixels}")
        print(f"Reconstruction threshold: {threshold.cpu().int()}")
        recons_points = None
        recons_img = torch.ones_like(raster_img)
        remaining_img = raster_img.clone()
        while (remaining_img < .5).sum() > threshold:
            remaining_img = diff_remaining_img(raster_img, recons_img)
            try:
                mark = sample_black_pixel(torch.clamp(remaining_img, min=0))
            except ValueError:
                break
            centered_img = remaining_img.clone()
            mark_real = mark.clone()
            mark_real[0] *= width
            mark_real[1] *= height
            centered_img = place_point_on_img(centered_img, mark_real)
            centered_img = center_on_point(centered_img, mark)
            points = model.generate(centered_img.unsqueeze(0))
            reverse_center_on_point(points, mark.unsqueeze(0))
            points = points.unsqueeze(1)
            if recons_points is None:
                recons_points = points.to(points.device)
            else:
                recons_points = torch.cat((recons_points, points), dim=1).to(points.device)
            recons_img, _ = raster_bezier_batch(recons_points, image_width=width, image_height=height, stroke_width=stroke_width, mode="hard")
            recons_img = recons_img.squeeze(0)

    output_filepath = splitext(basename(input_image))[0] + ".svg"
    if output is not None:
        if isdir(output):
            makedirs(output, exist_ok=True)
            output_filepath = join(output, output_filepath)
        elif type(output) is str and output.endswith(".svg"):
            output_filepath = output
    recons_points = recons_points.squeeze(0)
    recons_points[:, :, 0] *= width
    recons_points[:, :, 1] *= height
    save_as_svg(recons_points, output_filepath, original_width, original_height, stroke_width=stroke_width)


def main():
    parser = ArgumentParser(description='Test script for the marked reconstruction model')
    parser.add_argument("model", metavar="FIlE", help="path to the *ckpt file")
    parser.add_argument("-i", "--input_images", nargs="*", metavar="FILE", help="one or multiple paths to raster images that should be vectorized.")
    parser.add_argument("-d", "--input_dir", metavar="DIR", help="path to a directory of raster images that should be vectorized.")
    parser.add_argument("-o", "--output", help="optional output directory or file")
    parser.add_argument("--threshold_ratio", "-t", default=0.1, type=float, help="The ratio of black pixels which need to be reconstructed before the algorithm terminates")
    parser.add_argument("--stroke_width", "-r", default=0.512, type=float, help="stroke width if it should be different from the one specified in the model")
    parser.add_argument("--seed", "-s", default=1234)
    parser.add_argument("--device", default="cuda", help="Which device to use. For available values, refer to https://pytorch.org/docs/stable/tensor_attributes.html#torch.device")
    parser.add_argument("-b", "--binarization_threshold", default=0., type=float, help="Set to (0,1) to binarize the image.")
    parser.add_argument("-u", "--use_default_transforms", action="store_true", help="whether to use the default transforms instead of the ones defined in the VAEExperiment object")
    
    args = parser.parse_args()

    set_seed(args.seed)
    model, transforms = setup_model(args.model, args.device)
    if args.use_default_transforms:
        transforms = None
    if args.input_images is not None:
        input_images = args.input_images
    elif args.input_dir is not None and isdir(args.input_dir):
        input_images = [join(args.input_dir, f) for f in listdir(args.input_dir)]
    else:
        print("-i or -d need to be passed")
        exit(1)
    for input_image in input_images:
        vectorize_image(input_image, model, output=args.output, transforms=transforms, threshold_ratio=args.threshold_ratio, stroke_width=args.stroke_width, binarization_threshold=args.binarization_threshold, force_grayscale=False)


if __name__ == "__main__":
    main()

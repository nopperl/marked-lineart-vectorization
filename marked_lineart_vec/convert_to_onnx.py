#!/usr/bin/env python3
from argparse import ArgumentParser
from os.path import join, split

import torch

from marked_lineart_vec.experiment import VAEExperiment

parser = ArgumentParser(description="Convert the marked curve reconstruction model to ONNX.")
parser.add_argument("model", metavar="FIlE", help="path to the *ckpt file")

args = parser.parse_args()
model_path = args.model
onnx_filepath = join(split(model_path)[0], "model.onnx")
print("ONNX version will be saved to " + onnx_filepath)

with torch.no_grad():
    experiment = VAEExperiment.load_from_checkpoint(model_path, strict=False, map_location="cuda")
    model = experiment.model
    model = model.cuda()
    model.eval()
    params = experiment.params
    dummy_image = torch.randn(params["val_batch_size"], 3, params["img_size"], params["img_size"], device="cuda")
    input_names = ["marked_raster_image"]
    output_names = ["cubic_bezier_curve"]
    dynamic_axes = {input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}}
    torch.onnx.export(model, (dummy_image, None, torch.tensor(True)), onnx_filepath, verbose=True, input_names=input_names, output_names=output_names, opset_version=11, dynamic_axes=dynamic_axes)

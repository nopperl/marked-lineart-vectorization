#!/usr/bin/env python3
from argparse import ArgumentParser
from io import BytesIO
from os import listdir, makedirs
from os.path import basename, isdir, join, splitext
from random import randint
from typing import Union

from cairosvg import svg2png
import numpy as np
from imageio.v3 import imread, imwrite
from skimage.transform import rescale
from svgpathtools import CubicBezier, Line, QuadraticBezier, disvg, wsvg

import onnx
import onnxruntime as ort


def raster_bezier_hard(all_points, image_width=128, image_height=128, stroke_width=2., colors=None, white_background=True, mark=None):
    if colors is None:
        colors = [[0., 0., 0., 1.]] * len(all_points)
    elif colors is list and colors[0] is not list:
        colors = [colors] * len(all_points)
    else:
        colors = np.array(colors)
        colors[:, :3] *= 255
    colors = ["rgb(" + ",".join(map(str, color[:3])) + ")" for color in colors]
    background_color = "white" if white_background else None
    all_points = all_points + 0
    all_points[:, :, 0] *= image_width
    all_points[:, :, 1] *= image_height
    bezier_curves = [numpy_to_bezier(points) for points in all_points]
    attributes = [{"stroke": colors[i], "stroke-width": str(stroke_width), "fill": "none"} for i in range(len(bezier_curves))]
    if mark is not None:
        mark = mark + 0
        mark[0] *= image_width
        mark[1] *= image_height
        mark_points = np.vstack([mark - stroke_width, mark + stroke_width])
        mark_path = numpy_to_bezier(mark_points)
        mark_attr = {"stroke": "blue", "stroke-width": str(stroke_width * 2), "fill": "blue"}
        bezier_curves.append(mark_path)
        attributes.append(mark_attr)
    svg_attributes = {"width": f"{image_width}px", "height": f"{image_height}px"}
    svg_string = disvg(bezier_curves, attributes=attributes, svg_attributes=svg_attributes, paths2Drawing=True).tostring()
    png_string = svg2png(bytestring=svg_string, background_color=background_color)
    image = imread(BytesIO(png_string), extension=".png")
    output = image.astype("float32")
    output /= 255
    output = np.moveaxis(output, 2, 0)
    return output, all_points

def diff_remaining_img(raster_img: np.ndarray, recons_img: np.ndarray):
    remaining_img = raster_img.copy()
    tmp_remaining_img = remaining_img.copy()
    tmp_remaining_img[tmp_remaining_img < 1] = 0.
    recons_img[recons_img < 1] = 0.
    same_mask = (tmp_remaining_img == recons_img).copy()
    remaining_img[same_mask] = 1
    return remaining_img


def place_point_on_img(image, point):
    if np.any(point == point.astype(int)):
        point_idx_start = point.astype(int)
        point_idx_end = point.astype(int) + 1
    else:
        point_idx_start = np.floor(point).astype(int)
        point_idx_end = np.ceil(point).astype(int)
    if image.shape[0] == 3:
        image[0, point_idx_start[1]:point_idx_end[1], point_idx_start[0]:point_idx_end[0]] = 0
        image[1, point_idx_start[1]:point_idx_end[1], point_idx_start[0]:point_idx_end[0]] = 0
        image[2, point_idx_start[1]:point_idx_end[1], point_idx_start[0]:point_idx_end[0]] = 1
    else:
        image[0, point_idx_start[1]:point_idx_end[1], point_idx_start[0]:point_idx_end[0]] = 0.5
    return image


def rgb_to_grayscale(image: np.ndarray):
    image = image[0] * .2989 + image[1] *.587 + image[2] *.114
    return image


def sample_black_pixel(image: np.ndarray):
    image = rgb_to_grayscale(image.copy())
    black_indices = np.argwhere(~np.isclose(image, np.ones_like(image, dtype="float32"), atol=0.5) != 0)
    black_idx = black_indices[randint(0, len(black_indices) - 1)].astype("float32")
    black_idx[0] /= image.shape[0]
    black_idx[1] /= image.shape[1]
    black_idx = black_idx[[1, 0]]
    return black_idx


def numpy_to_bezier(points: np.ndarray):
    if len(points) == 2:
        return Line(*(complex(point[0], point[1]) for point in points))
    elif len(points) == 3:
        return QuadraticBezier(*(complex(point[0], point[1]) for point in points))
    elif len(points) == 4:
        return CubicBezier(*(complex(point[0], point[1]) for point in points))


def center_on_point(image, point, new_width=None, new_height=None):
    _, height, width = image.shape
    if new_width is None:
        new_width = width
    if new_height is None:
        new_height = height
    half_width = round(width / 2)
    half_height = round(height / 2)
    point = point.copy()
    point[0] *= width
    point[1] *= height
    point = point.round().astype(int)
    top=half_height - (half_height - point[1])
    left=half_width - (half_width - point[0])
    padded = np.pad(image, ((0, 0), (half_height, half_height), (half_width, half_width)), constant_values=1)
    cropped = padded[:, top:top+new_height, left:left+new_width]
    return cropped


def reverse_center_on_point(paths, point):
    for i in range(len(paths)):
        paths[i, :, 0] -= 0.5 - point[i, 0]
        paths[i, :, 1] -= 0.5 - point[i, 1]


def save_as_svg(curves: np.ndarray, filename, img_width, img_height, stroke_width=2.0):
    svg_paths = [numpy_to_bezier(curve) for curve in curves]
    output_attributes = [{"stroke": "black", "stroke-width": stroke_width, "stroke-linecap": "round", "fill": "none"}] * len(svg_paths)
    svg_attributes = {"width": f"{img_width}px", "height": f"{img_height}px"}
    wsvg(svg_paths, attributes=output_attributes, svg_attributes=svg_attributes, filename=filename)


def save_as_png(filename: str, image: np.ndarray):
    image = np.moveaxis(image.copy(), 0, 2)
    image *= 255
    imwrite(filename, image.round().astype("uint8")) 


def setup_model(model_path):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    ort_sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    return ort_sess


def vectorize_image(input_image_path, model: Union[str, ort.InferenceSession], output=None, threshold_ratio=0.1, stroke_width=0.512, width=512, height=512, binarization_threshold=0, force_grayscale=False):
    if type(model) is str:
        ort_sess = setup_model(model)
    elif type(model) is ort.InferenceSession:
        ort_sess = model
    else:
        raise ValueError("Invalid value for the model argument")

    # Get dimensions expected by the model
    _, channels, height, width = ort_sess.get_inputs()[0].shape
    input_image = imread(input_image_path, pilmode="RGB") / 255
    original_height, original_width, _ = input_image.shape
    # scale and white pad image to dimensions expected by the model
    if original_height >= original_width:
        scale = height / original_height
    else:
        scale = width / original_width
    print(f"Rescale factor: {scale}")
    input_image = rescale(input_image, scale, channel_axis=2, order=5)
    scaled_height, scaled_width = input_image.shape[:2]
    raster_img = np.ones((height, width, channels), dtype="float32")
    raster_img[:input_image.shape[0], :input_image.shape[1]] = input_image
    # convert CHW
    raster_img = np.moveaxis(raster_img, 2, 0)
    if binarization_threshold > 0:
        raster_img[raster_img < binarization_threshold] = 0.
    width = raster_img.shape[2]
    height = raster_img.shape[1]
    curve_pixels = (raster_img < .5).sum()
    threshold = curve_pixels * threshold_ratio
    print(f"Reconstruction candidate pixels: {curve_pixels}")
    print(f"Reconstruction threshold: {threshold.astype(int)}")
    recons_points = None
    recons_img = np.ones_like(raster_img, dtype="float32")
    remaining_img = raster_img.copy()
    while (remaining_img < .5).sum() > threshold:
        remaining_img = diff_remaining_img(raster_img, recons_img)
        try:
            mark = sample_black_pixel(remaining_img)
        except ValueError:
            break
        centered_img = remaining_img.copy()
        mark_real = mark.copy()
        mark_real[0] *= width
        mark_real[1] *= height
        centered_img = place_point_on_img(centered_img, mark_real)
        centered_img = center_on_point(centered_img, mark)
        result = ort_sess.run(None, {"marked_raster_image": np.expand_dims(centered_img, 0)})
        points = result[0]
        reverse_center_on_point(points, np.expand_dims(mark, 0))
        points = np.expand_dims(points, 1)
        if recons_points is None:
            recons_points = points
        else:
            recons_points = np.concatenate((recons_points, points), axis=1)
        recons_img, _ = raster_bezier_hard(recons_points.squeeze(0), image_width=width, image_height=height, stroke_width=stroke_width)

    output_filepath = splitext(basename(input_image_path))[0] + ".svg"
    if output is not None:
        if isdir(output):
            makedirs(output, exist_ok=True)
            output_filepath = join(output, output_filepath)
        elif type(output) is str and output.endswith(".svg"):
            output_filepath = output
    recons_points = recons_points.squeeze(0)
    recons_points[:, :, 0] *= width * (1 / scale)
    recons_points[:, :, 1] *= height * (1 / scale)
    save_as_svg(recons_points, output_filepath, original_width, original_height, stroke_width=stroke_width)

def main():
    parser = ArgumentParser(description="Inference script for the marked curve reconstruction model in ONNX format.")
    parser.add_argument("model", metavar="FIlE", help="path to the *.onnx file")
    parser.add_argument("-i", "--input_images", nargs="*", metavar="FILE", help="one or multiple paths to raster images that should be vectorized.")
    parser.add_argument("-d", "--input_dir", metavar="DIR", help="path to a directory of raster images that should be vectorized.")
    parser.add_argument("-o", "--output", help="optional output directory or file")
    parser.add_argument("--threshold_ratio", "-t", default=0.1, type=float, help="The ratio of black pixels which need to be reconstructed before the algorithm terminates")
    parser.add_argument("--stroke_width", "-r", default=0.512, type=float, help="stroke width if it should be different from the one specified in the model")
    parser.add_argument("--seed", "-s", default=1234, help="Fixed random number generation seed. Set to negative number to deactivate")
    parser.add_argument("-b", "--binarization_threshold", default=0., type=float, help="Set to a value in (0,1) to binarize the image.")
    
    args = parser.parse_args()
    
    if args.seed >= 0:
        np.random.seed(args.seed)
    if args.input_images is not None:
        input_images = args.input_images
    elif args.input_dir is not None and isdir(args.input_dir):
        input_images = [join(args.input_dir, f) for f in listdir(args.input_dir)]
    else:
        print("-i or -d need to be passed")
        exit(1)
    for input_image in input_images:
        vectorize_image(input_image, args.model, output=args.output, threshold_ratio=args.threshold_ratio, stroke_width=args.stroke_width, binarization_threshold=args.binarization_threshold, force_grayscale=False)


if __name__ == "__main__":
    main()

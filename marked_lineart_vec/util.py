from itertools import chain
from random import randint
from re import sub

import torch
from svgpathtools import CubicBezier, QuadraticBezier, Line, Path, wsvg
from torchvision.transforms.functional import rgb_to_grayscale, pad, crop


def tensor_to_bezier(points: torch.Tensor):
    if len(points) == 2:
        return Line(*(complex(point[0], point[1]) for point in points))
    elif len(points) == 3:
        return QuadraticBezier(*(complex(point[0], point[1]) for point in points))
    elif len(points) == 4:
        return CubicBezier(*(complex(point[0], point[1]) for point in points))


def get_bezier_curve_length(points: torch.Tensor):
    return tensor_to_bezier(points).length()


def sample_random_point(paths: torch.Tensor):
    bs = paths.shape[0]
    point_t = 0.1 * torch.randint(low=1, high=9, size=(bs,))
    marks_complex = [tensor_to_bezier(path).point(point_t[i]) for i, path in enumerate(paths)]
    marks = torch.tensor([[mark.real, mark.imag] for mark in marks_complex], device=paths.device)
    return marks


def sample_black_pixel(image: torch.Tensor):
    image = enforce_grayscale(image.clone()).squeeze()
    # sample only points where at least one consecutive pixel is black (pooling?)
    # image = enforce_grayscale(image.clone())
    # pooled_img = F.avg_pool2d(image, kernel_size=2)
    # image = F.resize(pooled_img, size=image.shape[1:])
    # image = image.squeeze()
    black_indices = torch.nonzero(~torch.isclose(image, torch.ones_like(image).float(), atol=0.5))
    black_idx = black_indices[randint(0, len(black_indices) - 1)].float()
    black_idx[0] /= image.shape[0]
    black_idx[1] /= image.shape[1]
    black_idx = black_idx[[1, 0]]
    return black_idx

def switch_major(image):
    return image.permute(0, 2, 1)

def center_on_point(image, point, new_width=None, new_height=None):
    _, height, width = image.shape
    if new_width is None:
        new_width = width
    if new_height is None:
        new_height = height
    if type(width) is torch.Tensor:
        half_width = (width / 2).round().to(torch.int64)
        half_height = (height / 2).round().to(torch.int64)
    else:
        half_width = round(width / 2)
        half_height = round(height / 2)
    point = point.clone()
    point[0] *= width
    point[1] *= height
    point = point.round().int()
    image = crop(
        pad(image, [half_width, half_height], 1),
        top=half_height - (half_height - point[1]),
        left=half_width - (half_width - point[0]),
        height=new_height,
        width=new_width
    )
    return image


def reverse_center_on_point(paths, point):
    for i in range(len(paths)):
        paths[i, :, 0] -= 0.5 - point[i, 0]
        paths[i, :, 1] -= 0.5 - point[i, 1]


def center_path_on_point(path, point):
    path[:, 0] += 0.5 - point[0]
    path[:, 1] += 0.5 - point[1]
    return path


def place_point_on_img(image, point):
    if torch.any(point == point.int()):
        point_idx_start = point.int()
        point_idx_end = point.int() + 1
    else:
        point_idx_start = point.floor().int()
        point_idx_end = point.ceil().int()
    if image.shape[0] == 3:
        image[0, point_idx_start[1]:point_idx_end[1], point_idx_start[0]:point_idx_end[0]] = 0
        image[1, point_idx_start[1]:point_idx_end[1], point_idx_start[0]:point_idx_end[0]] = 0
        image[2, point_idx_start[1]:point_idx_end[1], point_idx_start[0]:point_idx_end[0]] = 1
    else:
        image[0, point_idx_start[1]:point_idx_end[1], point_idx_start[0]:point_idx_end[0]] = 0.5
    return image


def convert_to_cubic(path, recursive=True):
    if type(path) is Line:
        yield CubicBezier(path.start, path.start, path.end, path.end)
    elif type(path) is QuadraticBezier:
        yield CubicBezier(path.start, (path.start + 2 * path.control) / 3, (path.end + 2 * path.control) / 3, path.end)
    elif type(path) is CubicBezier:
        yield path
    elif type(path) is Path:
        if not recursive:
            if len(path._segments) == 1:
                segment = next(iter(path))
                if type(segment) is Line:
                    yield CubicBezier(segment.start, segment.start, segment.end, segment.end)
                elif type(segment) is QuadraticBezier:
                    yield CubicBezier(segment.start, (segment.start + 2 * segment.control) / 3, (segment.end + 2 * segment.control) / 3, segment.end)
                else:
                    yield segment
        else:
            for segment in iter(path):
                if type(segment) is Line:
                    yield CubicBezier(segment.start, segment.start, segment.end, segment.end)
                elif type(segment) is QuadraticBezier:
                    yield CubicBezier(segment.start, (segment.start + 2 * segment.control) / 3, (segment.end + 2 * segment.control) / 3, segment.end)
                else:
                    yield segment
    else:
        yield path


def all_paths_as_cubic(paths, recursive=True):
    return list(chain(*(convert_to_cubic(path, recursive=recursive) for path in paths)))


def distance_from_point_to_line(point, line):
    s = line[0]  # start point
    e = line[1]  # end point
    length = torch.sqrt(torch.square(e[0] - s[0]) + torch.square(e[1] - s[1]))
    area = torch.abs((e[0] - s[0]) * (e[1] - point[1]) - (s[0] - point[0]) * (e[1] - s[1]))
    return area / length


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def enforce_grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.shape[-3] == 3:
        image = rgb_to_grayscale(image)
    return image


def binary_cross_entropy(x, y, p=1.0):
    x = x + 0.00000000000001
    loss = p * y * torch.clamp(torch.log(x), min=-100) + (1 - y) * torch.clamp(torch.log(1 - x), min=-100)
    loss = -loss
    return loss.mean()


def combine_image_and_canvas(image: torch.Tensor, canvas: torch.Tensor, in_channels=4):
    canvas_single_channel = enforce_grayscale(canvas).squeeze()
    shape = list(image.shape)
    shape[1] = in_channels
    image_canvas = torch.empty(shape, device=image.device)
    image_canvas[:, :in_channels - 1] = image
    image_canvas[:, in_channels - 1] = canvas_single_channel
    return image_canvas


def save_as_svg(curves: torch.Tensor, filename, img_width, img_height, stroke_width=2.0):
    svg_paths = [tensor_to_bezier(curve) for curve in curves]
    output_attributes = [{"stroke": "black", "stroke-width": stroke_width, "stroke-linecap": "round", "fill": "none"}] * len(svg_paths)
    svg_attributes = {"width": f"{img_width}px", "height": f"{img_height}px"}
    wsvg(svg_paths, attributes=output_attributes, svg_attributes=svg_attributes, filename=filename)


def diff_remaining_img(raster_img: torch.Tensor, recons_img: torch.Tensor):
    remaining_img = raster_img.clone()
    tmp_remaining_img = remaining_img.clone()
    tmp_remaining_img[tmp_remaining_img < 1] = 0.
    recons_img[recons_img < 1] = 0.
    same_mask = (tmp_remaining_img == recons_img).clone()
    remaining_img[same_mask] = 1
    return remaining_img


def path_is_in_view(path, view_width, view_height):
    return path.bbox()[0] > 0 and path.bbox()[1] < view_width and path.bbox()[2] > 0 and path.bbox()[3] < view_height

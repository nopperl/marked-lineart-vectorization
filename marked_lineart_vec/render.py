from io import BytesIO

from PIL import Image
import torch
import numpy as np
from cairosvg import svg2png
from svgpathtools import disvg
from torchvision.transforms.functional import to_tensor

from marked_lineart_vec.util import tensor_to_bezier
try: import pydiffvg
except: pass


def raster_bezier(all_points, image_width=128, image_height=128, stroke_width=2., colors=None, white_background=True, mark=None, seed=1234):
    num_curves = all_points.shape[0]
    if colors is None:
        colors = [[0, 0, 0, 1]] * num_curves
    assert len(colors[0]) == 4
    if mark is not None:
        raise NotImplementedError
    # all_points = all_points + 0.5
    all_points = all_points + 0
    all_points[:, :, 0] = all_points[:, :, 0] * image_width
    all_points[:, :, 1] = all_points[:, :, 1] * image_height
    num_ctrl_pts = torch.full(size=[1], fill_value=all_points.shape[1] - 2, dtype=torch.int32).to(all_points.device)
    colors = torch.tensor(colors).float().to(all_points.device)
    shapes = []
    shape_groups = []
    for k in range(num_curves):
        # Get point parameters from network
        points = all_points[k].contiguous()#[self.sort_idx[k]] # .cpu()
        path = pydiffvg.Path(
            num_control_points=num_ctrl_pts, points=points,
            stroke_width=torch.tensor(stroke_width).to(all_points.device),
            is_closed=False)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]),
            fill_color=None,
            stroke_color=colors[k])
        shape_groups.append(path_group)
    scene_args = pydiffvg.RenderFunction.serialize_scene(image_width, image_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    out = render(image_width,  # width
                 image_height,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 seed,  # seed
                 None,
                 *scene_args)
    out = out.permute(2, 0, 1)

    # map to [-1, 1]
    if white_background:
        alpha = out[3:4, :, :]
        out = out[:3]*alpha + (1-alpha)
    del colors, num_ctrl_pts, shapes, shape_groups, scene_args
    return out, all_points


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
    # all_points = all_points + 0.5
    all_points = all_points + 0
    all_points[:, :, 0] *= image_width
    all_points[:, :, 1] *= image_height
    bezier_curves = [tensor_to_bezier(points) for points in all_points]
    attributes = [{"stroke": colors[i], "stroke-width": str(stroke_width), "fill": "none"} for i in range(len(bezier_curves))]
    if mark is not None:
        # mark += 0.5
        mark = mark + 0
        mark[0] *= image_width
        mark[1] *= image_height
        mark_path = tensor_to_bezier(torch.cat([mark.unsqueeze(0) - stroke_width, mark.unsqueeze(0) + stroke_width]))
        mark_attr = {"stroke": "blue", "stroke-width": str(stroke_width * 2), "fill": "blue"}
        bezier_curves.append(mark_path)
        attributes.append(mark_attr)
    svg_attributes = {"width": f"{image_width}px", "height": f"{image_height}px"}
    svg_string = disvg(bezier_curves, attributes=attributes, svg_attributes=svg_attributes, paths2Drawing=True).tostring()
    png_string = svg2png(bytestring=svg_string, background_color=background_color)
    image = Image.open(BytesIO(png_string))
    output = to_tensor(image)
    return output, all_points


def raster_bezier_batch(all_points, image_width=128, image_height=128, stroke_width=2., colors=None, white_background=True, marks=None, mode="soft"):
    bs = all_points.shape[0]
    outs = torch.empty((bs, 3, image_height, image_width)).to(all_points.device)
    return_points = torch.empty((bs, *all_points.shape[1:])).to(all_points.device)
    raster_fn = raster_bezier if mode == "soft" else raster_bezier_hard
    marks = [None] * bs if marks is None else marks
    for batch_nr in range(bs):
        output, points = raster_fn(all_points[batch_nr], image_width, image_height, stroke_width, colors, white_background, marks[batch_nr])
        outs[batch_nr] = output
        return_points[batch_nr] = points
    return outs, return_points

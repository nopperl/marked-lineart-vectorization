from itertools import chain
from os import listdir
from os.path import splitext, join
from random import random, randint, uniform, sample as random_sample

import torch
from kornia.filters import GaussianBlur2d
from svgpathtools import svg2paths, QuadraticBezier, CubicBezier, Path, svg2paths2
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pad, crop, center_crop, to_pil_image

from marked_lineart_vec.render import raster_bezier_batch, raster_bezier_hard
from marked_lineart_vec.util import get_bezier_curve_length, tensor_to_bezier, center_on_point, sample_random_point, \
    all_paths_as_cubic, place_point_on_img, center_path_on_point, path_is_in_view

gauss = GaussianBlur2d((11, 11), (10.5, 10.5))


def generate_visible_paths(batch_size, max_paths, num_points, img_width, img_height, stroke_width):
    # Randomly generate paths and ensure that every path is visible
    # curves_batch = torch.empty((batch_size, max_paths - 1, num_points, 2))
    # num_required_samples = batch_size
    # while num_required_samples > 0:
    #     paths = torch.rand(size=(curves_batch.shape[0] * curves_batch.shape[1], *curves_batch.shape[2:]),
    #                        requires_grad=False) * 2 - 1
    #     path_imgs, _ = raster_bezier_batch(paths.unsqueeze(1), image_width=img_width,
    #                                        image_height=img_height, stroke_width=stroke_width,
    #                                        mode="hard")
    #     white_ratio = path_imgs.sum(dim=[1, 2, 3]) / torch.tensor(path_imgs.shape[1:]).prod()
    #     visible_paths = paths[white_ratio <= 0.94]
    #     fitting_number = int(len(visible_paths) / curves_batch.shape[1]) * curves_batch.shape[1]
    #     visible_paths = visible_paths[:fitting_number]
    #     visible_paths = visible_paths[:num_required_samples * curves_batch.shape[1]]
    #     visible_paths = visible_paths.view(-1, *curves_batch.shape[1:])
    #     cursor = batch_size - num_required_samples
    #     curves_batch[cursor:cursor + len(visible_paths)] = visible_paths
    #     num_required_samples = num_required_samples - len(visible_paths)
    curves_batch = torch.rand(size=(batch_size, max_paths - 1, num_points, 2), requires_grad=False)
    curves_batch = torch.stack(
        [torch.stack(sorted(curves, key=lambda curve: get_bezier_curve_length(curve))) for curves in curves_batch])
    return curves_batch


def transform_paths(paths, width, height, dropout=True, reverse=True, flip=True, p=0.5):
    # drop paths
    if dropout and random() < p:
        drop_indices = random_sample(range(len(paths)), int(len(paths) * uniform(0.1, 0.9)))
        paths = [path for i, path in enumerate(paths) if i not in drop_indices]
    # reverse orientation of paths
    if reverse and random() < p:
        paths = [path.reversed() for path in paths]
    # translate paths - not really necessary due to mark translation
    # step = min(width, height) / 4
    # random_translation = complex(uniform(step, step), uniform(step, step))
    # paths = [path.translated(random_translation) for path in paths]
    # mirror paths
    if flip and random() < p:
        mir = complex(width, height)
        paths = [CubicBezier(mir - path.start, mir - path.control1, mir - path.control2, mir - path.end) for path in paths]
    # sanity check
    paths = [path for path in paths if path_is_in_view(path, width, height)]
    return paths


def generate_visible_marks(target_path):
    bs = target_path.shape[0]
    marks = torch.empty((bs, 2), device=target_path.device)
    for i, path in enumerate(target_path):
        mark = 1+1j
        while not (mark.real < 1 and mark.real > 0 and mark.imag < 1 and mark.imag > 0):
            mark = tensor_to_bezier(path).point(random())
        marks[i] = torch.tensor([mark.real, mark.imag])


class RasterVectorDataset(ImageFolder):
    def __init__(self, num_points, stroke_width=0.7, raster_loss_size=None, batch_size=None, return_canvas=True, *args, **kwargs):
        super(RasterVectorDataset, self).__init__(*args, **kwargs)
        self.num_points = num_points
        self.stroke_width = stroke_width
        self.raster_loss_size = raster_loss_size
        self.batch_size = batch_size
        self.return_canvas = return_canvas

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        file_path, _ = self.samples[index]
        vector_file_path = splitext(file_path)[0] + ".svg"
        paths_pre = svg2paths(vector_file_path)[0]
        paths = all_paths_as_cubic(paths_pre)
        paths = sorted(paths, key=lambda path: path.length())
        width = sample.shape[1]
        height = sample.shape[2]
        points = list(chain(*(path.bpoints() for path in paths)))
        points = torch.tensor([(point.real, point.imag) for point in points])
        points = points.view(-1, self.num_points, 2)
        points[:, :, 0] /= width
        points[:, :, 1] /= height
        if not self.return_canvas:
            return (sample, points), 0
        num_paths = randint(0, len(points))
        selected_curves_batch = points[:, :num_paths].clone()
        remaining_curves_batch = points[:, num_paths:].clone()
        selected_sample_batch, _ = raster_bezier_batch(selected_curves_batch, image_width=width,
                                                       image_height=height,
                                                       stroke_width=self.stroke_width, mode="hard")
        return (sample, 0, remaining_curves_batch, selected_sample_batch, selected_curves_batch, 0), 0


class MarkedRasterVectorDataset(ImageFolder):
    def __init__(self, num_points, stroke_width=0.7, raster_loss_size=None, path_recursion=True, data_augmentation=False, transform=None, *args, **kwargs):
        super(MarkedRasterVectorDataset, self).__init__(*args, **kwargs)
        self.num_points = num_points
        self.stroke_width = stroke_width
        self.raster_loss_size = raster_loss_size
        self.path_recursion = path_recursion
        self.data_augmentation = data_augmentation
        self.transform = transform

    def __getitem__(self, index):
        _, target = super().__getitem__(index)
        file_path, _ = self.samples[index]
        vector_file_path = splitext(file_path)[0] + ".svg"
        paths_pre, _, svg_attributes = svg2paths2(vector_file_path)
        paths = all_paths_as_cubic(paths_pre, recursive=self.path_recursion)
        paths = sorted(paths, key=lambda path: path.length())
        width = float(svg_attributes["width"].replace("px", ""))
        height = float(svg_attributes["height"].replace("px", ""))
        paths = [path for path in paths if path.length() > 1 and path_is_in_view(path, width, height)]
        if self.data_augmentation:
            paths = transform_paths(paths, width, height)
        target_path_idx = randint(0, len(paths) - 1)
        target_path_length = torch.tensor(paths[target_path_idx].length() / width)
        points = chain(*(path.bpoints() for path in paths))
        points = torch.tensor([(point.real, point.imag) for point in points]).float()
        points = points.view(-1, self.num_points, 2)
        # Random rotation
        if self.data_augmentation and random() < .5:
            swap = points[:, :, 0].clone()
            points[:, :, 0] = points[:, :, 1]
            points[:, :, 1] = swap
            swap = width
            width = height
            height = swap
        target_path = points[target_path_idx].clone()
        mark = sample_random_point(target_path.unsqueeze(0)).squeeze().to(target_path.device)
        target_path[:, 0] /= width
        target_path[:, 1] /= height
        points[:, :, 0] /= width
        points[:, :, 1] /= height
        sample, _ = raster_bezier_hard(points, image_width=width, image_height=height, stroke_width=self.stroke_width)
        sample = place_point_on_img(sample, mark)
        if self.transform is not None:
            sample = self.transform(to_pil_image(sample))
            width = sample.shape[1]
            height = sample.shape[2]
        mark[0] /= width
        mark[1] /= height
        if self.raster_loss_size is not None:
            # sample = center_crop(center_on_point(sample, mark), self.raster_loss_size)
            width = self.raster_loss_size
            height = self.raster_loss_size
        target_path_img, _ = raster_bezier_hard(target_path.unsqueeze(0), image_width=width, image_height=height, stroke_width=self.stroke_width)
        target_path_img = center_on_point(target_path_img, mark)
        target_path = center_path_on_point(target_path, mark)
        return (sample, 0, mark, target_path, target_path_img, target_path_length), target


class GeneratedRasterVectorDatasetFixedStored(Dataset):
    def __init__(self, data_dir, device="cpu"):
        super().__init__()
        self.data_dir = data_dir
        self.device = device

    def __getitem__(self, index):
        index_dir = join(self.data_dir, str(index))
        return (torch.load(join(index_dir, "raster_img.pt")).to(self.device),
                torch.load(join(index_dir, "remaining.pt")).to(self.device),
                torch.load(join(index_dir, "remaining_points.pt")).to(self.device),
                torch.load(join(index_dir, "canvas_img.pt")).to(self.device),
                torch.load(join(index_dir, "canvas_points.pt")).to(self.device),
                None), 0

    def __len__(self):
        return len(listdir(self.data_dir))


class GeneratedRasterVectorDatasetFixed(Dataset):
    def __init__(self, num_images, min_paths, max_paths, img_width, img_height, num_points, stroke_width=2.0, canvas_noise=True, canvas_blur=False, batch_size=None, indiv_raster_remaining_only=True, device="cpu"):
        super().__init__()
        self.num_images = num_images
        self.min_paths = min_paths
        self.max_paths = max_paths
        self.img_width = img_width
        self.img_height = img_height
        self.num_points = num_points
        self.device = device
        self.stroke_width = stroke_width
        self.canvas_noise = canvas_noise
        self.canvas_blur = canvas_blur
        self.batch_size = batch_size
        self.indiv_raster_remaining_only = indiv_raster_remaining_only
        self._generate_dataset()

    def _generate_dataset(self):
        with torch.no_grad():
            num_paths = randint(self.min_paths, self.max_paths - 1 - 1)
            self.all_sample_batch = torch.empty((self.num_images, self.batch_size, 3, self.img_width, self.img_height), device=self.device)
            if self.indiv_raster_remaining_only:
                self.all_remaining_batch = torch.empty(self.num_images, self.batch_size, self.max_paths - num_paths - 1, 3, self.img_width,
                                              self.img_height).to(self.device)
            else:
                self.all_remaining_batch = torch.empty(self.num_images, self.batch_size, self.max_paths - 1, 3, self.img_width,
                                          self.img_height).to(self.device)
            self.all_remaining_curves_batch = torch.empty((self.num_images, self.batch_size, self.max_paths - num_paths - 1, self.num_points, 2),
                                                          device=self.device)
            self.all_selected_sample_batch = torch.empty((self.num_images, self.batch_size, 3, self.img_width, self.img_height), device=self.device)
            self.all_selected_curves_batch = torch.empty((self.num_images, self.batch_size, num_paths, self.num_points, 2), device=self.device)
            dataset = GeneratedRasterVectorDataset(self.num_images, self.min_paths, self.max_paths, self.img_width, self.img_height, self.num_points, stroke_width=self.stroke_width, batch_size=self.batch_size, canvas_noise=self.canvas_noise, canvas_blur=self.canvas_blur, indiv_raster_remaining_only=self.indiv_raster_remaining_only, num_paths=num_paths, device=self.device)
            dataset_loader = iter(dataset)
            for i in range(self.num_images):
                (sample_batch, remaining_batch, remaining_curves_batch, selected_sample_batch, selected_curves_batch, _), _ = next(dataset_loader)
                self.all_sample_batch[i] = sample_batch
                self.all_remaining_batch[i] = remaining_batch
                self.all_remaining_curves_batch[i] = remaining_curves_batch
                self.all_selected_sample_batch[i] = selected_sample_batch
                self.all_selected_curves_batch[i] = selected_curves_batch

    def __getitem__(self, index):
        return (self.all_sample_batch[index], self.all_remaining_batch[index], self.all_remaining_curves_batch[index], self.all_selected_sample_batch[index], self.all_selected_curves_batch[index], None), 0

    def __len__(self):
        return self.num_images


class GeneratedRasterVectorDataset(Dataset):
    def __init__(self, num_images, min_paths, max_paths, img_width, img_height, num_points, stroke_width=2.0, batch_size=None, canvas_noise=True, canvas_blur=False, indiv_raster_remaining_only=True, num_paths=None, device="cpu"):
        super(GeneratedRasterVectorDataset, self).__init__()
        self.num_images = num_images
        self.min_paths = min_paths
        self.max_paths = max_paths
        self.img_width = img_width
        self.img_height = img_height
        self.num_points = num_points
        self.device = device
        self.stroke_width = stroke_width
        self.canvas_noise = canvas_noise
        self.canvas_blur = canvas_blur
        self.batch_size = batch_size
        self.indiv_raster_remaining_only = indiv_raster_remaining_only
        self.num_paths = num_paths

    def __getitem__(self, index):
        with torch.no_grad():
            num_paths = randint(self.min_paths, self.max_paths - 1 - 1) if self.num_paths is None else self.num_paths  # TODO: remove one -1 to learn stop
            fill_value = -min(self.img_width, self.img_height)
            if self.batch_size is None:
                curves = torch.rand(size=(num_paths, self.num_points, 2), requires_grad=False) * 2 - 1
                # x_min, x_max = int(-0.5 * self.img_width), int(1.5 * self.img_width)
                # y_min, y_max = int(-0.5 * self.img_height), int(1.5 * self.img_height)
                # x_rand = torch.randint(low=x_min, high=x_max, size=(num_paths, self.num_points), device=self.device)
                # y_rand = torch.randint(low=y_min, high=y_max, size=(num_paths, self.num_points), device=self.device)
                # curves = torch.stack((x_rand, y_rand), dim=2)
                curves = torch.stack(sorted(curves, key=lambda curve: get_bezier_curve_length(curve)))
                sample, _ = raster_bezier_hard(curves, image_width=self.img_width, image_height=self.img_height,
                                                    stroke_width=self.stroke_width)
                out_curves = torch.full(size=(self.max_paths, self.num_points, 2), fill_value=fill_value, device=self.device)
                out_curves[:num_paths] = curves
                del curves
                return (sample, out_curves), 0
            else:
                curves_batch = generate_visible_paths(self.batch_size, self.max_paths, self.num_points, int(self.img_width / 4), int(self.img_height / 4), self.stroke_width)
                selected_curves_batch = curves_batch[:, :num_paths].clone()
                remaining_curves_batch = curves_batch[:, num_paths:].clone()
                if self.canvas_noise:
                    selected_curves_batch += torch.rand_like(selected_curves_batch) * 1e-1
                sample_batch, _ = raster_bezier_batch(curves_batch, image_width=self.img_width, image_height=self.img_height,
                                                    stroke_width=self.stroke_width, mode="hard")
                selected_sample_batch, _ = raster_bezier_batch(selected_curves_batch, image_width=self.img_width, image_height=self.img_height,
                                                    stroke_width=self.stroke_width, mode="hard")
                if self.canvas_blur:
                    selected_sample_batch = gauss(selected_sample_batch)
                # remaining_curves_batch = torch.full(size=(self.batch_size, self.max_paths - num_paths, self.num_points, 2), fill_value=fill_value, device=self.device, requires_grad=False)
                # remaining_curves_batch[:, :self.max_paths - num_paths - 1] = curves_batch[:, num_paths:]
                if self.indiv_raster_remaining_only:
                    nr_remaining_curves = remaining_curves_batch.shape[1]
                    individual_batch = torch.empty(self.batch_size, nr_remaining_curves, 3, self.img_width, self.img_height).to(self.device)
                    for nr_path in range(nr_remaining_curves):
                        image_batch, _ = raster_bezier_batch(remaining_curves_batch[:, nr_path].unsqueeze(1), image_width=self.img_width,
                                                                       image_height=self.img_height,
                                                                       stroke_width=self.stroke_width, mode="hard")
                        individual_batch[:, nr_path] = image_batch
                else:
                    individual_batch = torch.empty(self.batch_size, curves_batch.shape[1], 3, self.img_width, self.img_height).to(self.device)
                    for nr_path in range(curves_batch.shape[1]):
                        image_batch, _ = raster_bezier_batch(curves_batch[:, nr_path].unsqueeze(1), image_width=self.img_width,
                                                                       image_height=self.img_height,
                                                                       stroke_width=self.stroke_width, mode="hard")
                        individual_batch[:, nr_path] = image_batch
                stop = torch.tensor(1. if remaining_curves_batch.shape[1] == 0 else 0.)
                del curves_batch
                return (sample_batch, individual_batch, remaining_curves_batch, selected_sample_batch, selected_curves_batch, stop), 0

    def __len__(self):
        return self.num_images


class MarkedGeneratedRasterVectorDatasetFixed(Dataset):
    def __init__(self, num_images, min_paths, max_paths, img_width, img_height, num_points, stroke_width=2.0, batch_size=None, device="cpu"):
        super().__init__()
        self.num_images = num_images
        self.min_paths = min_paths
        self.max_paths = max_paths
        self.img_width = img_width
        self.img_height = img_height
        self.num_points = num_points
        self.device = device
        self.stroke_width = stroke_width
        self.batch_size = batch_size
        self._generate_dataset()

    def _generate_dataset(self):
        with torch.no_grad():
            self.all_sample_batch = torch.empty((self.num_images, self.batch_size, 3, self.img_width, self.img_height), device=self.device)
            self.all_marks = torch.empty((self.num_images, self.batch_size, 2), device=self.device)
            self.all_target_path = torch.empty((self.num_images, self.batch_size, self.num_points, 2), device=self.device)
            self.all_target_path_length = torch.empty((self.num_images, self.batch_size), device=self.device)
            self.all_target_path_img = torch.empty((self.num_images, self.batch_size, 3, self.img_width, self.img_height), device=self.device)
            self.all_curves_batch = torch.empty((self.num_images, self.batch_size, self.max_paths - 1, self.num_points, 2), device=self.device)
            dataset = MarkedGeneratedRasterVectorDataset(self.num_images, self.min_paths, self.max_paths, self.img_width, self.img_height, self.num_points, stroke_width=self.stroke_width, batch_size=self.batch_size, device=self.device)
            dataset_loader = iter(dataset)
            for i in range(self.num_images):
                (sample_batch, curves_batch, marks, target_path, target_path_img, target_path_length), _ = next(dataset_loader)
                self.all_sample_batch[i] = sample_batch
                self.all_curves_batch[i] = curves_batch
                self.all_marks[i] = marks
                self.all_target_path[i] = target_path
                self.all_target_path_length[i] = target_path_length
                self.all_target_path_img[i] = target_path_img

    def __getitem__(self, index):
        return (self.all_sample_batch[index], self.all_curves_batch[index], self.all_marks[index], self.all_target_path[index], self.all_target_path_img[index], self.all_target_path_length[index]), 0

    def __len__(self):
        return self.num_images


class MarkedGeneratedRasterVectorDataset(Dataset):
    def __init__(self, num_images, min_paths, max_paths, img_width, img_height, num_points, stroke_width=2.0, batch_size=None, device="cpu"):
        super(MarkedGeneratedRasterVectorDataset, self).__init__()
        self.num_images = num_images
        self.min_paths = min_paths
        self.max_paths = max_paths
        self.img_width = img_width
        self.img_height = img_height
        self.num_points = num_points
        self.device = device
        self.stroke_width = stroke_width
        self.batch_size = 1 if batch_size is None else batch_size

    def __getitem__(self, index):
        with torch.no_grad():
            num_paths = randint(self.min_paths, self.max_paths - 1 - 1)
            curves_batch = generate_visible_paths(self.batch_size, self.max_paths, self.num_points,
                                                  int(self.img_width / 4), int(self.img_height / 4), self.stroke_width)
            target_path = curves_batch[:, num_paths].clone()
            target_path_length = torch.tensor([tensor_to_bezier(t).length() for t in target_path])
            marks = sample_random_point(target_path).to(target_path.device)
            mark = marks.clone()

            sample_batch, _ = raster_bezier_batch(curves_batch, image_width=self.img_width, image_height=self.img_height,
                                                stroke_width=self.stroke_width, marks=marks, mode="hard")
            target_path_img, _ = raster_bezier_batch(target_path.unsqueeze(1), image_width=self.img_width, image_height=self.img_height,
                                                stroke_width=self.stroke_width, mode="hard")
            target_path_img = torch.cat([center_on_point(im, marks[i]).unsqueeze(0) for i, im in enumerate(target_path_img)])
            # target_path_img = pad(target_path_img, [int(round(self.img_width / 2)), int(round(self.img_height / 2))], 1)
            # target_path_img = torch.cat([center_on_point(pad(im, [int(round(self.img_width / 2)), int(round(self.img_height / 2))], 1), marks[i]).unsqueeze(0) for i, im in enumerate(target_path_img)])

            if self.batch_size > 1:
                return (sample_batch, 0, mark, target_path, target_path_img, target_path_length), 0
            else:
                return (sample_batch.squeeze(0), 0, mark.squeeze(0), target_path.squeeze(0), target_path_img.squeeze(0), target_path_length.squeeze(0)), 0

    def __len__(self):
        return self.num_images


class AlternatingDataset(Dataset):
    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        with torch.no_grad():
            if index < len(self.dataset1):
                return self.dataset1[index]
            else:
                return self.dataset2[index - len(self.dataset1)]

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

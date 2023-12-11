from os import makedirs
from os.path import join, exists, dirname
from typing import Optional

import torch
from PIL.ImageOps import pad
from torch import optim, Tensor
from torch.nn import Threshold
from torchmetrics import IoU
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch_optimizer as optim_

from torchvision.transforms.functional import rgb_to_grayscale

from marked_lineart_vec.datasets import RasterVectorDataset, GeneratedRasterVectorDatasetFixed, GeneratedRasterVectorDataset, \
    MarkedGeneratedRasterVectorDataset, MarkedGeneratedRasterVectorDatasetFixed, MarkedRasterVectorDataset, \
    AlternatingDataset
from marked_lineart_vec.models import IterativeModel, vae_models, VectorVAEnLayers
from marked_lineart_vec.models.line_identification import LineIdentificationModel
from marked_lineart_vec.models.marked_reconstruction import MarkedReconstructionModel
from marked_lineart_vec.render import raster_bezier_batch
from marked_lineart_vec.util import enforce_grayscale, place_point_on_img, switch_major


class VAEExperiment(pl.LightningModule):
    def __init__(self, model_params: dict, params: dict) -> None:
        super(VAEExperiment, self).__init__()
        self.model = vae_models[model_params['name']](**model_params)
        self.params = params
        self.curr_device = None
        self.beta_scale = 2.0  #1.4
        self.image_dir = None
        self.save_hyperparameters()
        self.iou = IoU(num_classes=2)

    def setup(self, stage: Optional[str] = None) -> None:
        super(VAEExperiment, self).setup(stage)
        if hasattr(self.logger, "log_dir"):
            self.image_dir = join(dirname(self.logger.log_dir), "images")
            makedirs(self.image_dir, exist_ok=True)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs)

    def _common_step(self, batch):
        if type(self.model) is MarkedReconstructionModel:
            (raster_img, _, marks, target_path, target_image, target_path_length), _ = batch
            if self.params.get("force_grayscale"):
                raster_img = rgb_to_grayscale(raster_img)
            results = self.forward(raster_img, mark=marks)
            recons_image, mu, log_var = results[:3]
            train_detail = self.model.loss_function(image=recons_image,
                                                    target_image=target_image,
                                                    mu=mu,
                                                    log_var=log_var,
                                                    bezier_points=results[3] if len(results) > 3 else None,
                                                    target_points=target_path,
                                                    target_path_length=target_path_length)
        else:
            (raster_img, individual_imgs, remaining_points, canvas_img, canvas_points, stop), labels = batch
            individual_imgs = rgb_to_grayscale(individual_imgs)
            if self.params.get("force_grayscale"):
                raster_img = rgb_to_grayscale(raster_img)
                canvas_img = rgb_to_grayscale(canvas_img)
            if type(self.model) in [IterativeModel, LineIdentificationModel]:
                target_points = remaining_points
            else:
                target_points = torch.cat((canvas_points, remaining_points), dim=1).to(self.curr_device)
            if self.model in [IterativeModel, LineIdentificationModel]:
                with torch.no_grad():
                    target_image, _ = raster_bezier_batch(remaining_points, image_width=self.params["img_size"],
                                                          image_height=self.params["img_size"],
                                                          stroke_width=self.params["radius"], mode="soft")
                    if self.params.get("force_grayscale"):
                        target_image = rgb_to_grayscale(target_image)
            else:
                target_image = raster_img
            results = self.forward(raster_img, canvas=canvas_img, canvas_points=canvas_points,
                                   num_paths=target_points.shape[1])
            recons_image, mu, log_var = results[:3]
            train_detail = self.model.loss_function(image=recons_image,
                                                    target_image=target_image,
                                                    mu=mu,
                                                    log_var=log_var,
                                                    bezier_points=results[3] if len(results) > 3 else None,
                                                    target_points=target_points,
                                                    target_path_images=individual_imgs,
                                                    stop_true=stop)
        return raster_img, recons_image, target_image, train_detail

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        raster_img, recons_image, target_image, train_detail = self._common_step(batch)
        self.curr_device = raster_img.device
        train_loss = train_detail.pop("loss")
        self.log("train_loss", train_loss.item())
        self.log("train_loss_details", train_detail["progress_bar"], prog_bar=True)
        self.iou(enforce_grayscale(recons_image).round().int(), enforce_grayscale(target_image).round().int())
        self.log("train_iou_step", self.iou)
        if batch_idx == 0:
            self.logger.experiment.add_images("raster_train_2", raster_img.data, self.global_step)
            self.logger.experiment.add_images("target_train_2", target_image.data, self.global_step)
            self.logger.experiment.add_images("recons_train_2", recons_image.data, self.global_step)
            self.sample_images(draw_control_points=self.params.get("show_control_points", False), batch=batch)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        raster_img, recons_image, target_image, val_detail = self._common_step(batch)
        val_loss = val_detail.pop("loss")
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_loss_details", val_detail["progress_bar"], prog_bar=True)
        if type(self.model) is IterativeModel:
            (_, _, remaining_points, canvas_img, canvas_points, _), _ = batch
            full_recons_image, _ = self.model.generate_whole(raster_img, remaining_points.shape[1], canvas_img, canvas_points, canvas_blur=self.params.get("canvas_blur"))
            self.iou(enforce_grayscale(full_recons_image).round().int(), enforce_grayscale(raster_img).round().int())
        if type(self.model) is LineIdentificationModel:
            (_, _, remaining_points, canvas_img, canvas_points, _), _ = batch
            points, loss = self.model.generate_whole(raster_img, remaining_points, canvas_img, canvas_points, canvas_blur=self.params.get("canvas_blur"))
            # TODO: other metric
            self.log("val_loss_whole", loss, prog_bar=True)
        elif type(self.model) is MarkedReconstructionModel:
            self.iou(enforce_grayscale(recons_image).round().int(), enforce_grayscale(target_image).round().int())
        #     (_, all_paths, _, _, _), _ = batch
        #     full_recons_image, _ = self.model.generate_whole(raster_img, all_paths.shape[1], all_paths)
        #     self.iou(enforce_grayscale(full_recons_image).round().int(), enforce_grayscale(raster_img).round().int())
        else:
            self.iou(enforce_grayscale(recons_image).round().int(), enforce_grayscale(raster_img).round().int())
        self.log("val_iou_step", self.iou)
        return val_loss

    # def test_step(self, *args, **kwargs):
    #     return self.validation_step(*args, **kwargs)

    def training_epoch_end(self, outputs):
        super(VAEExperiment, self).training_epoch_end(outputs)
        self.sample_images(self.params.get("show_control_points", False))
        self.log("train_iou_epoch", self.iou)
        print('learning rate: ', self.trainer.optimizers[0].param_groups[0]["lr"])

    def validation_epoch_end(self, outputs) -> None:
        super(VAEExperiment, self).validation_epoch_end(outputs)
        self.log("val_iou_epoch", self.iou)

    def sample_images(self, draw_control_points=False, batch=None):
        # Get sample reconstruction image
        if type(self.model) is MarkedReconstructionModel:
            if batch is None:
                (test_image, _, marks, _, _, _), test_label = next(iter(self.test_dataloader()))
            else:
                (test_image, _, marks, _, _, _), test_label = batch
        else:
            if batch is None:
                (test_image, _, _, test_canvas, test_canvas_points, _), test_label = next(iter(self.test_dataloader()))
            else:
                (test_image, _, _, test_canvas, test_canvas_points, _), test_label = batch
            test_canvas = test_canvas.to(self.curr_device)
            test_canvas_points = test_canvas_points.to(self.curr_device)
        test_image = test_image.to(self.curr_device)
        if self.params.get("force_grayscale"):
            test_image = rgb_to_grayscale(test_image)
        if type(self.model) is IterativeModel:
            points = self.model.generate(test_image, test_canvas, canvas_points=test_canvas_points)
            shape = list(test_canvas_points.shape)
            shape[1] += 1
            all_points = torch.empty(shape)
            all_points[:, :-1] = test_canvas_points
            all_points[:, -1] = points
            # all_points /= self.params["img_size"]
            # all_points -= 0.5
            colors = [[0., 0., 0., 1.]] * test_canvas_points.shape[1] + [[0., 0., 1., 1.]]
            # all_points = (points.unsqueeze(1) / self.params["img_size"]) - 0.5
            recons, _ = raster_bezier_batch(all_points, image_width=self.params["img_size"], image_height=self.params["img_size"], stroke_width=self.params["radius"], mode="hard", colors=colors)
        elif type(self.model) is VectorVAEnLayers:
            recons, _ = self.model.generate(test_image)
        elif type(self.model) is MarkedReconstructionModel:
            points = self.model.generate(test_image, marks)
            recons, _ = raster_bezier_batch(points.unsqueeze(1), image_width=test_image.shape[3], image_height=test_image.shape[2], stroke_width=self.params["radius"], mode="hard")
        elif type(self.model) is LineIdentificationModel:
            points = self.model.generate(test_image, test_canvas, canvas_points=test_canvas_points)
            recons = test_image.clone()
            points *= self.params["img_size"]
            for i in range(len(points)):
                recons[i] = place_point_on_img(recons[i], points[i])
        else:
            recons = self.model.generate(test_image)

        if draw_control_points:
            points = points.clamp(min=0, max=self.params["img_size"] - 1)
            points = points.round().long()
            colors = [0.565, 0.392, 0.173] + [0.094, 0.310, 0.635] * (points.shape[1] - 1)
            colors = torch.tensor(colors).to(recons.device).view(3, points.shape[1])
            for i in range(len(recons)):
                recons[i, :, points[i,:,1], points[i,:,0]] = colors

        gen_label = "generated_images" if batch is None else "generated_images_train"
        real_label = "real_images" if batch is None else "real_images_train"
        self.logger.experiment.add_images(gen_label, recons.data, self.global_step)
        self.logger.experiment.add_images(real_label, test_image.data, self.global_step)

        # if type(self.image_dir) is str and batch is None:
        #     vutils.save_image(recons.data,
        #                       f"{self.image_dir}/"
        #                       f"recons_{self.current_epoch:04d}.png",
        #                       normalize=False,
        #                       nrow=12)
        #
        #     vutils.save_image(test_image.data,
        #                       f"{self.image_dir}/"
        #                       f"real_img_{self.current_epoch:04d}.png",
        #                       normalize=False,
        #                       nrow=12)

        # del test_image, test_canvas, test_canvas_points, test_label, recons#, points
        del test_image, test_label, recons  # , points

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.params["LR"], weight_decay=self.params.get("weight_decay", 0.))

    def train_dataloader(self):
        transform = self.data_transforms()
        if type(self.model) is MarkedReconstructionModel:
            generated_dataset = MarkedGeneratedRasterVectorDataset(num_images=self.params.get("num_synthetic_images", self.params["num_images"]),
                                                     min_paths=self.params["min_paths"],
                                                     max_paths=self.params["max_paths"],
                                                     img_width=self.params["synthetic_img_size"],
                                                     img_height=self.params["synthetic_img_size"],
                                                     num_points=self.params["points"],
                                                     stroke_width=self.params["radius"],
                                                     batch_size=self.params["batch_size"])
            douga_dataset = MarkedRasterVectorDataset(self.params["points"], self.params.get("radius"),
                                                self.params.get("raster_loss_size"),
                                                path_recursion=self.params.get("path_recursion"),
                                                root=self.params['data_path'],
                                                data_augmentation=self.params.get("data_augmentation"),
                                                transform=transform)
        else:
            generated_dataset = GeneratedRasterVectorDataset(num_images=self.params.get("num_synthetic_images", self.params["num_images"]),
                                               min_paths=self.params["min_paths"],
                                               max_paths=self.params["max_paths"],
                                               img_width=self.params["synthetic_img_size"],
                                               img_height=self.params["synthetic_img_size"],
                                               num_points=self.params["points"],
                                               stroke_width=self.params["radius"],
                                               canvas_noise=self.params.get("canvas_noise"),
                                               canvas_blur=self.params.get("canvas_blur"),
                                               batch_size=self.params["batch_size"],
                                               indiv_raster_remaining_only=self.params.get(
                                                   "indiv_raster_remaining_only"))
            douga_dataset = RasterVectorDataset(self.params["points"], self.params.get("radius"),
                                          self.params.get("raster_loss_size"), root=self.params['data_path'],
                                          transform=transform)
        if self.params["dataset"] == "combined":
            generated_dataset.batch_size = 1
            combined_dataset = AlternatingDataset(generated_dataset, douga_dataset)
            train_dataloader = DataLoader(combined_dataset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True, num_workers=4)
        elif self.params["dataset"] == "generated":
            # generated_dataset = GeneratedRasterVectorDatasetFixedStored(data_dir=self.params["data_path"])
            train_dataloader = DataLoader(generated_dataset, batch_size=None, shuffle=False, num_workers=4)
        else:
            train_dataloader = DataLoader(douga_dataset,
                      batch_size= self.params['batch_size'],
                      shuffle = self.params.get("train_shuffle", True),
                      drop_last=True, num_workers=4)

        return train_dataloader

    def val_dataloader(self):
        transform = self.data_transforms()
        if self.params["dataset"] == "generated":
            if type(self.model) is MarkedReconstructionModel:
                test_dataset = MarkedGeneratedRasterVectorDatasetFixed(num_images=self.params["num_val_images"],
                                                                 min_paths=self.params["min_paths"],
                                                                 max_paths=self.params["max_paths"],
                                                                 img_width=self.params["synthetic_img_size"],
                                                                 img_height=self.params["synthetic_img_size"],
                                                                 num_points=self.params["points"],
                                                                 stroke_width=self.params["radius"],
                                                                 batch_size=self.params["val_batch_size"])
            else:
                test_dataset = GeneratedRasterVectorDatasetFixed(num_images=self.params["num_val_images"],
                                                                 min_paths=self.params["min_paths"],
                                                                 max_paths=self.params["max_paths"],
                                                                 img_width=self.params["synthetic_img_size"],
                                                                 img_height=self.params["synthetic_img_size"],
                                                                 num_points=self.params["points"],
                                                                 stroke_width=self.params["radius"],
                                                                 canvas_noise=self.params.get("canvas_noise"),
                                                                 canvas_blur=self.params.get("canvas_blur"),
                                                                 batch_size=self.params["val_batch_size"],
                                                       indiv_raster_remaining_only=self.params.get("indiv_raster_remaining_only"))
            sample_dataloader = DataLoader(test_dataset, batch_size=None, shuffle=False, num_workers=2)
        else:
            test_dataset = self.params['data_path'].replace('train','validation')
            if exists(test_dataset):
                if type(self.model) is MarkedReconstructionModel:
                    test_dataset = MarkedRasterVectorDataset(self.params["points"], self.params.get("radius"), self.params.get("raster_loss_size"), path_recursion=self.params.get("path_recursion"),
                                                        root=test_dataset, data_augmentation=False, transform=transform)
                else:
                    test_dataset = RasterVectorDataset(self.params["points"], self.params.get("radius"), self.params.get("raster_loss_size"), root=test_dataset, transform=transform)
            else:
                test_dataset = self.train_dataloader().dataset
            sample_dataloader = DataLoader(test_dataset,
                                                 batch_size= self.params['val_batch_size'],
                                                 shuffle = self.params['val_shuffle'],
                                                 drop_last=True, num_workers=4)
        return sample_dataloader

    def test_dataloader(self):
        return self.val_dataloader()

    def data_transforms(self):
        SetRange = transforms.Lambda(lambda X: (2 * X - 1.))
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))
        SwitchMajor = transforms.Lambda(lambda X: switch_major(X))

        if self.params["dataset"] in ["douga", "combined"]:
            transform_functions = []
            if self.params.get("img_size"):
                transform_functions.extend([transforms.Lambda(lambda X: pad(X, (self.params['img_size'], self.params['img_size']), color="white", centering=(0, 0))),
                                    transforms.CenterCrop(self.params['img_size']),
                                    ])
            transform_functions.extend([
                                    transforms.ToTensor(),
                                    Threshold(self.params.get("binarization_threshold", 0.), 0.)
                                    ])
            if self.params.get("force_grayscale"):
                transform_functions.append(Grayscale(num_output_channels=3))
            transform = transforms.Compose(transform_functions)
        else:
            transform_functions = [#transforms.RandomHorizontalFlip(),
                                    transforms.Resize(self.params['img_size']),
                                    # transforms.RandomRotation([0, 360], resample=3, fill=(255,255,255)),
                                    # transforms.RandomAffine([0, 0], (0.0,0.05), (1.0,1.0), resample=3, fillcolor=(255,255,255)),
                                    transforms.CenterCrop(self.params['img_size']),
                                    transforms.ToTensor(),
                                    ]
            if self.params.get("force_grayscale"):
                transform_functions.append(BlackAndWhite)
            transform = transforms.Compose(transform_functions)
        return transform

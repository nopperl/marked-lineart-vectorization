from typing import Optional, List, Tuple

import torch
from kornia.geometry.transform import PyrDown
from lpips import LPIPS
from segmentation_models_pytorch.losses import LovaszLoss, JaccardLoss
from segmentation_models_pytorch.losses._functional import soft_tversky_score
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import rgb_to_grayscale, pad

from marked_lineart_vec.models import BaseVAE
from marked_lineart_vec.models.fusion import ImageSequenceFusion
from marked_lineart_vec.models.image_encoder import SimpleConvEncoder, ResnetEncoder
from marked_lineart_vec.models.losses.dice_loss import BinaryDiceLoss
from marked_lineart_vec.models.losses.focal import binary_focal_loss_with_logits
from marked_lineart_vec.models.path_encoder import RnnPathEncoder, CnnPathEncoder
from marked_lineart_vec.render import raster_bezier_batch
from marked_lineart_vec.util import enforce_grayscale, sample_random_point, \
    center_on_point, reverse_center_on_point

from transformer_encoder import TransformerEncoder

dsample = PyrDown()


class MarkedReconstructionModel(BaseVAE):
    def __init__(self,
                 in_channels=3,
                 latent_dim=128,
                 hidden_dims: Optional[List[int]] = None,
                 loss_fn: str = 'MSE',
                 reparameterize=False,
                 num_points=4,
                 image_encoder="simple",
                 **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_img_size = None
        self.reparameterize = reparameterize
        self.in_channels = in_channels
        self.path_latent_dim = kwargs.get("path_latent_dim", self.latent_dim)
        self.pad_image = kwargs.get("pad_image")
        if kwargs.get("point_loss_fn") == "MAE":
            self.point_loss_fn = nn.L1Loss(reduction="none")
        elif kwargs.get("point_loss_fn") == "MSE":
            self.point_loss_fn = nn.MSELoss(reduction="none")
        else:
            self.point_loss_fn = nn.SmoothL1Loss(beta=0.1, reduction="none")  # nn.SmoothL1Loss(beta=1.0) == nn.HuberLoss() (requires torch >1.10)
        self.stop_loss_weight = kwargs.get("stop_loss_weight", 100)
        if loss_fn == 'BCE':
            # self.loss_fn = lambda x, y, **kwargs: F.binary_cross_entropy(x, y, **kwargs)
            self.loss_fn = lambda x, y, **kwargs: F.binary_cross_entropy_with_logits(x, y, pos_weight=torch.tensor([1 + (1 - (y.sum() / torch.tensor(y.shape).prod()))], device="cuda"), **kwargs)
        elif loss_fn == "MSE":
            self.loss_fn = F.mse_loss
        elif loss_fn == "huber":
            self.loss_fn = F.smooth_l1_loss
        elif loss_fn == "focal":
            self.loss_fn = binary_focal_loss_with_logits
        else:
            self.loss_fn = lambda x, y, **kwargs: torch.zeros(x.shape, device=x.device)

        self.curve_length_loss_weight = kwargs.get("curve_length_loss_weight", 1.)
        self.seg_loss_weight = kwargs.get("seg_loss_weight", 0)
        self.raster_loss_weight = kwargs.get("raster_loss_weight", 0)
        self.vector_loss_weight = kwargs.get("vector_loss_weight", 1)
        self.raster_loss_size = kwargs.get("raster_loss_size")
        seg_loss = kwargs.get("seg_loss", "none")
        if seg_loss == "none":
            self.seg_loss_fn = lambda x, y: 0
        elif seg_loss == "white_penalty":
            self.seg_loss_fn = lambda x, y: torch.clamp(x.sum(dim=[1,2,3]) / (1 * 3) - y.sum(dim=[1,2,3]) / (1 * 3), 0)
        elif seg_loss == "dice":
            self.seg_loss_fn = BinaryDiceLoss(reduction="none", smooth=0)
        elif seg_loss == "lovasz":
            self.seg_loss_fn = LovaszLoss("binary")
        elif seg_loss == "tversky":
            self.seg_loss_fn = lambda x, y: soft_tversky_score(x, y, alpha=0.7, beta=0.3)
        elif seg_loss == "jaccard":
            self.seg_loss_fn = JaccardLoss("binary", from_logits=False)

        if kwargs.get("perceptual_loss") == "lpips":
            self.perceptual_loss = LPIPS(net="vgg")
        else:
            self.perceptual_loss = lambda x, y: torch.tensor(0., device=x.device)
        self.num_points = num_points
        self.radius = kwargs.get("radius", 2.)

        if image_encoder == "simple":
            if kwargs.get("global_pooling"):
                self.image_encoder = SimpleConvEncoder(in_channels=in_channels, latent_dim=latent_dim, hidden_dims=hidden_dims, weights=kwargs.get("image_encoder_weights"), global_pooling=True)
            else:
                img_size = kwargs.get("input_img_size", kwargs.get("img_size", 512))
                if self.pad_image:
                    img_size *= 2
                self.image_encoder = SimpleConvEncoder(in_channels=in_channels, latent_dim=latent_dim, imsize=img_size, hidden_dims=hidden_dims, weights=kwargs.get("image_encoder_weights"), global_pooling=False)
        elif image_encoder == "resnet":
            self.image_encoder = ResnetEncoder()

        if kwargs.get("path_encoder") == "rnn":
            self.path_encoder = RnnPathEncoder(path_embedding_size=latent_dim, use_last=False)
        elif kwargs.get("path_encoder") == "cnn":
            self.path_encoder = CnnPathEncoder(num_points=num_points, path_embedding_size=latent_dim)
        elif kwargs.get("path_encoder") == "transformer":
            self.path_encoder = TransformerEncoder(d_model=latent_dim, d_ff=latent_dim)
        else:
            self.path_encoder = None

        if kwargs.get("path_encoder") and kwargs.get("path_encoder") != "none":
            self.img_seq_fusion = ImageSequenceFusion(latent_dim, latent_dim, latent_dim, int(latent_dim / 2))

        self.add_mark_features = False
        if kwargs.get("add_mark_features"):
            self.mark_projection = nn.Linear(2, self.latent_dim)
            nn.init.zeros_(self.mark_projection.weight)
            self.add_mark_features = True
        
        if kwargs.get("temporal"):
            self.point_rnn = nn.LSTM(latent_dim, 2 + 1, 2, bidirectional=True)
            self.fc_stop = nn.Sequential(
                nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
                nn.ReLU(),
                nn.Linear(int(self.latent_dim / 2), 1),
                nn.Sigmoid()
            )
        self.bezier_point_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
            nn.BatchNorm1d(int(self.latent_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(self.latent_dim / 2), self.num_points * 2),
            nn.Sigmoid(),  # bound to -1,1
        )

        weights = kwargs.get("weights")
        if weights is not None:
            pretrained_dict = torch.load(weights)["state_dict"]
            pretrained_dict = {k.replace("model.", ""): v for k, v in pretrained_dict.items() if k.startswith("model.")}
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward(self, image: torch.Tensor, mark: Optional[torch.Tensor] = None, return_only_points=False, **kwargs) -> List[torch.Tensor]:
        # pad image to double the size
        image_width = image.shape[3]
        image_height = image.shape[2]
        if self.pad_image:
            image = pad(image, [int(round(image_width / 2)), int(round(image_height / 2))], 1)
        if mark is not None:
            # marks = (mark + 0.5) * self.imsize
            marks = mark.clone()
            if self.input_img_size is not None:
                new_width, new_height = self.input_img_size, self.input_img_size
            else:
                new_width, new_height = None, None
            centered_image = torch.empty_like(image, device=image.device)
            for i in range(image.shape[0]):
                centered_image[i] = center_on_point(image[i], marks[i], new_width=new_width, new_height=new_height)
            image = centered_image
        z = self.image_encoder(image)
        if self.path_encoder is not None and "canvas_points" in kwargs:
            z_curves = self.path_encoder(kwargs["canvas_points"])
            z = self.img_seq_fusion(z, z_curves)
        if mark is not None and self.add_mark_features:
            z = z + self.mark_projection(mark)
        points = self.decode(z, num_points=kwargs.get("num_points"))
        if return_only_points:
            return points
        if self.raster_loss_weight > 0:
            if self.raster_loss_size:
                image_width = self.raster_loss_size
                image_height = self.raster_loss_size
            out_image, _ = raster_bezier_batch(points.unsqueeze(1), image_width=image_width, image_height=image_height, stroke_width=self.radius, mode="soft")
            if self.pad_image:
                out_image = pad(out_image, [int(round(image_width / 2)), int(round(image_height / 2))], 1)
            if self.in_channels <= 2:
                out_image = rgb_to_grayscale(out_image)
        else:
            out_image = image

        return [out_image, None, None, points]

    def decode(self, z: torch.Tensor, **kwargs):
        bs = z.shape[0]
        # num_points = kwargs["num_points"] if "num_points" in kwargs else self.num_points
        # TODO: use rnn
        points = self.bezier_point_predictor(z)
        points = points.view(bs, self.num_points, 2)
        # points = points + 0.5
        # points = points * self.imsize
        # TODO: predict other parameters (color, radius) from z
        # stop = self.fc_stop(z)
        return points

    def loss_function(self, image, **kwargs) -> dict:
        # Temp raster loss
        recon_loss_raster = torch.tensor(0., device=image.device)
        if self.curve_length_loss_weight > 1 and "target_path_length" in kwargs:
            path_weights = kwargs["target_path_length"] * self.curve_length_loss_weight
        else:
            path_weights = torch.ones(image.shape[0]).to(image.device)
        if self.raster_loss_weight > 0:
            target_path_images = kwargs.get("target_path_images")
            target_image = kwargs.get("target_image")
            if target_image is not None:
                image = enforce_grayscale(image)
                target_image = enforce_grayscale(target_image)
                recon_loss_raster = self.seg_loss_fn(image.squeeze(), target_image.squeeze())
                recon_loss_raster = (recon_loss_raster * path_weights).mean()
            elif target_path_images is not None:
                bs, nr_images, _, _, _ = target_path_images.shape
                image_losses = torch.empty(bs, nr_images).to(image.device)
                image = enforce_grayscale(image)
                target_path_images = enforce_grayscale(target_path_images)
                for nr_image in range(nr_images):
                    image_loss = self.seg_loss_fn(image.squeeze(), target_path_images[:, nr_image].squeeze())
                    image_losses[:, nr_image] = image_loss
                recon_loss_raster = torch.min(image_losses, dim=1).values
                recon_loss_raster = recon_loss_raster.mean()

        # Vector supervision loss.
        bezier_points = kwargs.get("bezier_points")
        recon_loss_vec = torch.tensor(0., device=image.device)
        if bezier_points is not None and self.vector_loss_weight > 0:
            target_points = kwargs["target_points"]
            # === loop implpementation ===
            # bs = target_points.shape[0]
            # # bezier_points = torch.repeat_interleave(bezier_points.unsqueeze(1), target_points.shape[1], dim=1)
            # for k in range(bs):
            #     loss_per_path = self.point_loss_fn(bezier_points[k], target_points[k]).mean(dim=[1, 2])
            #     recon_loss_vec = recon_loss_vec + torch.min(loss_per_path)
            # recon_loss_vec = recon_loss_vec / bs
            # === non-loop implementation ===
            # bezier_points = bezier_points.unsqueeze(1)
            # # bezier_points = torch.repeat_interleave(bezier_points, target_points.shape[1], dim=1)  # repeat to get the same shape as input_points
            # loss_per_path = self.point_loss_fn(bezier_points, target_points).mean(dim=[2, 3])
            # recon_loss_vec = torch.min(loss_per_path, dim=1).values.mean()
            recon_loss_vec = self.point_loss_fn(bezier_points, target_points)
            recon_loss_vec = (recon_loss_vec.mean(dim=[1, 2]) * path_weights).mean()

        loss = self.raster_loss_weight * recon_loss_raster + self.vector_loss_weight * recon_loss_vec
        logs = {"Reconstruction_Loss_Raster": recon_loss_raster.item(), "Reconstruction_Loss_Vec": recon_loss_vec.item()}

        if self.reparameterize:
            log_var = kwargs["log_var"]
            mu = kwargs["mu"]
            kld_weight = kwargs.get("M_N", .1)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            logs["KLD"] = -kld_loss.item()
            loss = loss + kld_weight * kld_loss

        stop_pred = kwargs.get("stop_pred")
        if stop_pred is not None:
            stop_true = kwargs["stop_true"]
            stop_loss = F.binary_cross_entropy(stop_pred, stop_true)
            loss = loss + self.stop_loss_weight * stop_loss
            logs["Stop_Loss"] = stop_loss.item()
        return {'loss': loss, 'progress_bar': logs}

    def generate(self, image: torch.Tensor, mark: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Given an input image and optionally an image containing already drawn paths, reconstructs one of the remaining paths and returns it.
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x num_points]
        """
        paths = self.forward(image, mark=mark, return_only_points=True, **kwargs)
        if mark is not None:
            reverse_center_on_point(paths, mark)
        return paths

    def generate_whole(self, image: torch.Tensor, num_paths: int, all_paths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given an input image (and optionally already drawn paths), generate an output image containing all remaining paths.
        Args:
            image: The full raster image
            num_paths: The number of remaining paths to be drawn
            canvas: If given, needs to correspond with canvas_points
            canvas_points: If given, needs to correspond with canvas

        Returns:
            * The resulting raster image containing all paths
            * The parameters of all paths
        """
        canvas_paths = None
        image_width = image.shape[2]
        image_height = image.shape[1]
        for nr_path in range(num_paths):
            # TODO: find quick way to generate mark without paths
            marks = sample_random_point(all_paths[:, nr_path])
            image, _ = raster_bezier_batch(all_paths, image_width=image_width, image_height=image_height,
                                         stroke_width=self.radius, mode="hard", marks=marks)
            points = self.generate(image, mark=marks)
            points = points.unsqueeze(1)
            if canvas_paths is None:
                canvas_paths = points
            else:
                canvas_paths = torch.cat((canvas_paths, points), dim=1).to(points.device)

        out_image, _ = raster_bezier_batch(canvas_paths, image_width=image_width, image_height=image_height,
                                         stroke_width=self.radius, mode="hard")
        if self.in_channels <= 2:
            out_image = rgb_to_grayscale(out_image)
        return out_image, canvas_paths

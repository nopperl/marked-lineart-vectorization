from typing import Optional, List, Tuple

import torch
from kornia.filters import GaussianBlur2d
from lpips import LPIPS
from segmentation_models_pytorch.losses import LovaszLoss, JaccardLoss
from segmentation_models_pytorch.losses._functional import soft_tversky_score
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from transformer_encoder import TransformerEncoder

from marked_lineart_vec.models import BaseVAE
from marked_lineart_vec.models.fusion import ImageSequenceFusion
from marked_lineart_vec.models.image_encoder import SimpleConvEncoder, ResnetEncoder
from marked_lineart_vec.models.losses.dice_loss import BinaryDiceLoss
from marked_lineart_vec.models.losses.focal import binary_focal_loss_with_logits
from marked_lineart_vec.models.path_encoder import RnnPathEncoder, CnnPathEncoder
from marked_lineart_vec.render import raster_bezier_batch
from marked_lineart_vec.util import enforce_grayscale, combine_image_and_canvas
from kornia.geometry.transform import PyrDown

dsample = PyrDown()
gauss = GaussianBlur2d((11, 11), (10.5, 10.5))


class IterativeModel(BaseVAE):
    def __init__(self,
                 in_channels=4,
                 latent_dim=128,
                 hidden_dims: Optional[List[int]] = None,
                 loss_fn: str = 'MSE',
                 img_size: int = 128,
                 reparameterize=False,
                 num_points=4,
                 image_encoder="simple",
                 **kwargs) -> None:
        super(IterativeModel, self).__init__()
        self.latent_dim = latent_dim
        self.imsize = img_size
        self.reparameterize = reparameterize
        self.in_channels = in_channels
        self.path_latent_dim = kwargs.get("path_latent_dim", self.latent_dim)
        if kwargs.get("point_loss_fn") == "MAE":
            self.point_loss_fn = nn.L1Loss(reduction="none")
        elif kwargs.get("point_loss_fn") == "MSE":
            self.point_loss_fn = nn.MSELoss(reduction="none")
        else:
            self.point_loss_fn = nn.SmoothL1Loss(beta=0.1, reduction="none")  # nn.SmoothL1Loss(beta=1.0) == nn.HuberLoss() (requires torch >1.10)
        self.stop_loss_weight = kwargs.get("stop_loss_weight", 100)
        if loss_fn == 'BCE':
            self.loss_fn = lambda x, y, **kwargs: F.binary_cross_entropy(x, y, **kwargs)
        elif loss_fn == "MSE":
            self.loss_fn = F.mse_loss
        elif loss_fn == "huber":
            self.loss_fn = F.smooth_l1_loss
        elif loss_fn == "focal":
            self.loss_fn = binary_focal_loss_with_logits
        else:
            self.loss_fn = lambda x, y, **kwargs: torch.zeros(x.shape, device=x.device)

        self.seg_loss_weight = kwargs.get("seg_loss_weight", 0)
        self.raster_loss_weight = kwargs.get("raster_loss_weight", 0)
        self.vector_loss_weight = kwargs.get("vector_loss_weight", 1)
        seg_loss = kwargs.get("seg_loss", "none")
        if seg_loss == "none":
            self.seg_loss_fn = lambda x, y: 0
        elif seg_loss == "white_penalty":
            self.seg_loss_fn = lambda x, y: torch.clamp(x.sum(dim=[1,2,3]) / (1 * 3) - y.sum(dim=[1,2,3]) / (1 * 3), 0)
        elif seg_loss == "dice":
            self.seg_loss_fn = BinaryDiceLoss(reduction="none")
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
            self.image_encoder = SimpleConvEncoder(in_channels=in_channels, latent_dim=latent_dim, imsize=self.imsize, hidden_dims=hidden_dims, weights=kwargs.get("image_encoder_weights"), global_pooling=kwargs.get("global_pooling"))
        elif image_encoder == "resnet":
            self.image_encoder = ResnetEncoder()

        if kwargs.get("encode_image_canvas_separately"):
            self.encode_image_canvas_separately = True
            self.image_feature_fusion = nn.Linear(latent_dim * 2, latent_dim)

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

        def get_computational_unit(in_channels, out_channels, unit):
            if unit == 'conv':
                return nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, padding_mode='circular', stride=1,
                                 dilation=1)
            else:
                return nn.Linear(in_channels, out_channels)

        self.point_rnn = nn.LSTM(latent_dim, 2 + 1, 2, bidirectional=True)
        self.fc_stop = nn.Sequential(
            get_computational_unit(self.latent_dim, int(self.latent_dim / 2), 'mlp'),
            nn.ReLU(),
            get_computational_unit(int(self.latent_dim / 2), 1, 'mlp'),
            nn.Sigmoid()
        )
        self.bezier_point_predictor = nn.Sequential(
            get_computational_unit(self.latent_dim, int(self.latent_dim / 2), 'mlp'),
            nn.BatchNorm1d(int(self.latent_dim / 2)),
            nn.ReLU(),
            get_computational_unit(int(self.latent_dim / 2), self.num_points * 2, 'mlp'),
            nn.Tanh(),  # bound to -1,1
        )
        self.encode_image_canvas_separately = kwargs.get("encode_image_canvas_separately")

    def forward(self, image: torch.Tensor, canvas: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        if self.encode_image_canvas_separately:
            z_image = self.image_encoder(image)
            z_canvas = self.image_encoder(canvas)
            z = self.image_feature_fusion(torch.cat((z_image, z_canvas), dim=-1))
        else:
            image_canvas = combine_image_and_canvas(image, canvas, in_channels=self.in_channels)
            z = self.image_encoder(image_canvas)
        if self.path_encoder is not None and "canvas_points" in kwargs:
            z_curves = self.path_encoder(kwargs["canvas_points"])
            z = self.img_seq_fusion(z, z_curves)
        points = self.decode(z, num_points=kwargs.get("num_points"))
        if self.raster_loss_weight > 0:
            out_image, _ = raster_bezier_batch(points.unsqueeze(1), image_width=self.imsize, image_height=self.imsize, stroke_width=self.radius, mode="soft")
            if self.in_channels <= 2:
                out_image = rgb_to_grayscale(out_image)
        else:
            out_image = image
        if self.vector_loss_weight == 0:
            points = None

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

    def gaussian_pyramid_loss(self, recons, input_img):
        recon_loss = self.loss_fn(recons, input_img, reduction="none").mean(dim=[1, 2, 3]) + self.seg_loss_weight * self.seg_loss_fn(recons.squeeze(), input_img.squeeze())
        for j in range(2, 5):
            recons = dsample(recons)
            input_img = dsample(input_img)
            recon_loss = recon_loss + (self.loss_fn(recons, input_img, reduction="none").mean(dim=[1, 2, 3]) + self.seg_loss_weight * self.seg_loss_fn(recons.squeeze(), input_img.squeeze())) / j
        return recon_loss

    def loss_function(self, image, **kwargs) -> dict:
        # Temp raster loss
        recon_loss_raster = torch.tensor(0., device=image.device)
        if self.raster_loss_weight > 0:
            target_path_images = kwargs.get("target_path_images")
            target_image = kwargs.get("target_image")
            if target_image is not None:
                perceptual_loss = self.perceptual_loss(image, target_image).mean()
                image = enforce_grayscale(image)
                target_image = enforce_grayscale(target_image)
                recon_loss_raster = perceptual_loss + self.gaussian_pyramid_loss(image, target_image).mean()
            elif target_path_images is not None:
                bs, nr_images, _, _, _ = target_path_images.shape
                image_losses = torch.empty(bs, nr_images).to(image.device)
                for nr_image in range(nr_images):
                    perceptual_loss = self.perceptual_loss(image, target_path_images[:, nr_image]).squeeze()
                    image_losses[:, nr_image] = perceptual_loss
                image = enforce_grayscale(image)
                target_path_images = enforce_grayscale(target_path_images)
                for nr_image in range(nr_images):
                    image_loss = self.gaussian_pyramid_loss(image, target_path_images[:, nr_image])
                    image_losses[:, nr_image] += image_loss
                recon_loss_raster = torch.min(image_losses, dim=1).values.mean()

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
            bezier_points = bezier_points.unsqueeze(1)
            # bezier_points = torch.repeat_interleave(bezier_points, target_points.shape[1], dim=1)  # repeat to get the same shape as input_points
            loss_per_path = self.point_loss_fn(bezier_points, target_points).mean(dim=[2, 3])
            recon_loss_vec = torch.min(loss_per_path, dim=1).values.mean()
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

    def generate(self, image: torch.Tensor, canvas: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Given an input image and optionally an image containing already drawn paths, reconstructs one of the remaining paths and returns it.
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x num_points]
        """
        if canvas is None:
            canvas = torch.ones(size=image.shape(), device=image.device)
        if self.encode_image_canvas_separately:
            z_image = self.image_encoder(image)
            z_canvas = self.image_encoder(canvas)
            z = self.image_feature_fusion(torch.cat((z_image, z_canvas), dim=-1))
        else:
            image_canvas = combine_image_and_canvas(image, canvas, in_channels=self.in_channels)
            z = self.image_encoder(image_canvas)
        if self.path_encoder is not None and "canvas_points" in kwargs:
            z_curves = self.path_encoder(kwargs["canvas_points"])
            z = self.img_seq_fusion(z, z_curves)
        points = self.decode(z)
        return points

    def generate_whole(self, image: torch.Tensor, num_paths: int, canvas: Optional[torch.Tensor] = None,
                                canvas_points: Optional[torch.Tensor] = None, canvas_blur=False) -> Tuple[torch.Tensor, torch.Tensor]:
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
        for nr_path in range(num_paths):
            points = self.generate(image, canvas=canvas, canvas_points=canvas_points)
            points = points.unsqueeze(1)
            canvas_points = torch.cat((canvas_points, points), dim=1).to(points.device)
            canvas, _ = raster_bezier_batch(canvas_points, image_width=self.imsize, image_height=self.imsize,
                                         stroke_width=self.radius, mode="soft")
            if canvas_blur:
                canvas = gauss(canvas)

        if self.in_channels <= 2:
            canvas = rgb_to_grayscale(canvas)
        return canvas, canvas_points

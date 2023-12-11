from typing import List

import torch

from marked_lineart_vec.models.base import BaseVAE
from torch import nn
from torch.nn import functional as F

from .image_encoder import SimpleConvEncoder, ResnetEncoder
from .losses.dice_loss import BinaryDiceLoss
from kornia.geometry.transform import PyrDown

from .losses.focal import binary_focal_loss_with_logits
from .losses.multires import gaussian_pyramid_loss
from ..render import raster_bezier_batch
from ..util import enforce_grayscale

dsample = PyrDown()


class VanillaVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 imsize=64,
                 image_encoder="simple",
                 loss_fn="BCE",
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        if loss_fn == 'BCE':
            # self.loss_fn = lambda x, y, **kwargs: F.binary_cross_entropy_with_logits(x, y, pos_weight=torch.tensor([1 - (y.sum() / torch.tensor(y.shape).prod())], device="cuda"), **kwargs)
            self.loss_fn = lambda x, y, **kwargs: F.binary_cross_entropy(x, y, **kwargs)
        elif loss_fn == "MSE":
            self.loss_fn = F.mse_loss
        elif loss_fn == "huber":
            self.loss_fn = F.smooth_l1_loss
        elif loss_fn == "focal":
            self.loss_fn = binary_focal_loss_with_logits
        else:
            self.loss_fn = lambda x, y, **kwargs: torch.zeros(x.shape, device=x.device)
        self.seg_loss_fn = BinaryDiceLoss(reduction="none")
        self.seg_loss_weight = kwargs.get("seg_loss_weight", 0)

        if image_encoder == "simple":
            self.image_encoder = SimpleConvEncoder(in_channels=in_channels, latent_dim=latent_dim, imsize=imsize, hidden_dims=hidden_dims, reparameterize=True)
        elif image_encoder == "resnet":
            self.image_encoder = ResnetEncoder(reparameterize=True)

        # Build Decoder
        modules = []
        self.outsize = int(imsize / (2 ** len(hidden_dims)))

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.outsize * self.outsize)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=1,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, self.outsize, self.outsize)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, image: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        z, mu, log_var = self.image_encoder(image)
        recon_image = self.decode(z)
        return [recon_image, mu, log_var]

    def loss_function(self, image, target_image, mu, log_var, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :return:
        """
        kld_weight = 1 / image.shape[0]
        kld_weight = 0
        image = enforce_grayscale(image)
        target_image = enforce_grayscale(target_image)
        target_path_images = kwargs.get("target_path_images")
        if target_path_images:  # min difference to an individual target path
            target_path_images = enforce_grayscale(target_path_images)
            bs, nr_images, _, _, _ = target_path_images.shape
            image_losses = torch.empty(bs, nr_images).to(image.device)
            for nr_image in range(nr_images):
                image_loss = gaussian_pyramid_loss(image, target_path_images[:, nr_image], self.loss_fn, self.seg_loss_fn, self.seg_loss_weight)
                image_losses[:, nr_image] = image_loss
            recons_loss = torch.min(image_losses, dim=1).values.mean()
        else:  # difference to whole target image
            recons_loss = gaussian_pyramid_loss(image, target_image, self.loss_fn, self.seg_loss_fn, self.seg_loss_weight).mean()
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = 1 * recons_loss + kld_weight * kld_loss
        logs = {"Reconstruction_Loss_Raster": recons_loss.item(), "KLD": -kld_loss.item()}
        return {"loss": loss, "progress_bar": logs}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
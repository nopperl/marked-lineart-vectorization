import random
from typing import List

# import torch
from scipy.optimize import linear_sum_assignment

from marked_lineart_vec.models import BaseVAE
import torch
from torch import nn
from torch.nn import functional as F

from marked_lineart_vec.models.losses.focal import binary_focal_loss_with_logits
import numpy as np
import matplotlib.pyplot as plt
from marked_lineart_vec.models.losses.dice_loss import BinaryDiceLoss
from kornia.geometry.transform import PyrDown

from marked_lineart_vec.util import enforce_grayscale

dsample = PyrDown()
try: import pydiffvg
except: pass


class VectorVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 loss_fn: str = 'MSE',
                 imsize: int = 128,
                 paths: int = 4,
                 num_points: int = 4,
                 **kwargs) -> None:
        super(VectorVAE, self).__init__()

        self.latent_dim = latent_dim
        self.path_latent_dim = kwargs.get("path_latent_dim", self.latent_dim)
        self.imsize = imsize
        self.beta = kwargs.get('beta', 0)
        self.other_losses_weight = 0
        self.reparametrize_ = False
        if 'other_losses_weight' in kwargs.keys():
            self.other_losses_weight = kwargs['other_losses_weight']
        if 'reparametrize' in kwargs.keys():
            self.reparametrize_ = kwargs['reparametrize']

        self.curves = paths
        self.in_channels = in_channels
        self.scale_factor = kwargs.get('scale_factor', 1)
        self.learn_sampling = kwargs.get('learn_sampling', False)
        self.only_auxillary_training = kwargs.get('only_auxillary_training', False)
        self.memory_leak_training = kwargs.get('memory_leak_training', False)

        self.memory_leak_epochs = 105
        if 'memory_leak_epochs' in kwargs.keys():
            self.memory_leak_epochs = kwargs['memory_leak_epochs']

        if loss_fn == 'BCE':
            self.loss_fn = F.binary_cross_entropy_with_logits
        elif loss_fn == "MSE":
            self.loss_fn = F.mse_loss
        elif loss_fn == "focal":
            self.loss_fn = binary_focal_loss_with_logits
        else:
            self.loss_fn = lambda x, y, **kwargs: torch.zeros(x.shape, device=x.device)
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim
        outsize = int(imsize/(2**len(hidden_dims)))
        self.fc_mu = nn.Linear(hidden_dims[-1]*outsize*outsize, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*outsize*outsize, latent_dim)
        self.encoder = nn.Sequential(*modules)

        self.circle_rad = kwargs['radius']
        self.number_of_points = self.curves * 3

        sample_rate = 1
        angles = torch.arange(0, self.number_of_points, dtype=torch.float32) *6.28319/ self.number_of_points
        id = self.sample_circle(self.circle_rad, angles, sample_rate)
        base_control_features = torch.tensor([[1,0],[0,1],[0,1]], dtype=torch.float32)
        self.id = id[:,:]
        self.angles = angles
        self.register_buffer('base_control_features', base_control_features)
        self.deformation_range = 6.28319/ 4

        def get_computational_unit(in_channels, out_channels, unit):
            if unit=='conv':
                return nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, padding_mode='circular', stride=1, dilation=1)
            else:
                return nn.Linear(in_channels, out_channels)
            # Build Decoder

        unit='conv'
        if unit=='conv':
            self.decode_transform = lambda x: x.permute(0, 2, 1)
        else:
            self.decode_transform = lambda x: x
        num_one_hot = base_control_features.shape[1]
        fused_latent_dim = self.path_latent_dim + num_one_hot+ (sample_rate*2)
        self.decoder_input = get_computational_unit(fused_latent_dim, fused_latent_dim*2, unit)

        self.point_predictor = nn.ModuleList([
            get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
            get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
            get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
            get_computational_unit(fused_latent_dim*2, fused_latent_dim*2, unit),
            get_computational_unit(fused_latent_dim*2, 2, unit),
            # nn.Sigmoid()  # bound spatial extent
        ])
        if self.learn_sampling:
            self.sample_deformation = nn.Sequential(
                get_computational_unit(self.path_latent_dim + 2+ (sample_rate*2), self.path_latent_dim*2, unit),
                nn.ReLU(),
                get_computational_unit(self.path_latent_dim * 2, self.path_latent_dim * 2, unit),
                nn.ReLU(),
                get_computational_unit(self.path_latent_dim*2, 1, unit),
            )
        self.aux_network = nn.Sequential(
            get_computational_unit(latent_dim, latent_dim*2, 'mlp'),
            nn.LeakyReLU(),
            get_computational_unit(latent_dim * 2, latent_dim * 2, 'mlp'),
            nn.LeakyReLU(),
            get_computational_unit(latent_dim * 2, latent_dim * 2, 'mlp'),
            nn.LeakyReLU(),
            get_computational_unit(latent_dim*2, 3, 'mlp'),
        )
        self.latent_lossvpath = {}
        self.save_lossvspath = False
        if self.only_auxillary_training:
            self.save_lossvspath = True
            for name, param in self.named_parameters():
                if 'aux_network' in name:
                    print(name)
                    param.requires_grad =True
                else:
                    param.requires_grad =False
        # self.lpips = VGGPerceptualLoss(False)
        seg_loss = kwargs.get("seg_loss", "none")
        if seg_loss == "none":
            self.seg_loss_fn = lambda x, y: 0
        elif seg_loss == "white_penalty":
            self.seg_loss_fn = lambda x, y: torch.clamp(x.sum(dim=[1,2,3]) / (1 * 3) - y.sum(dim=[1,2,3]) / (1 * 3), 0)
        elif seg_loss == "dice":
            self.seg_loss_fn = BinaryDiceLoss(reduction="none")
        self.seg_loss_weight = kwargs.get("seg_loss_weight", 0)
        self.raster_loss_weight = kwargs.get("raster_loss_weight", 0)
        self.vector_loss_weight = kwargs.get("vector_loss_weight", 1)
        self.num_points = num_points

    def redo_features(self, n):
        self.curves = n
        self.number_of_points = self.curves * 3
        self.angles = (torch.arange(0, self.number_of_points, dtype=torch.float32) *6.28319/ self.number_of_points)

        id = self.sample_circle(self.circle_rad, self.angles, 1)
        self.id = id[:,:]

    def control_polygon_distance(self, all_points):
        def distance(vec1, vec2):
            return ((vec1-vec2)**2).mean()

        loss =0
        for idx in range(self.number_of_points):
            c_0 = all_points[:, idx - 1, :]
            c_1 = all_points[:, idx, :]
            loss = loss + distance(c_0, c_1)
        return loss

    def sample_circle(self, r, angles, sample_rate=10):
        pos = []
        for i in range(1, sample_rate+1):
            x = (torch.cos(angles*(sample_rate/i)) * r)# + r
            y = (torch.sin(angles*(sample_rate/i)) * r)# + r
            pos.append(x)
            pos.append(y)
        return torch.stack(pos, dim=-1)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.Tensor) Input torch.Tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def raster(self, all_points, color=[0,0,0, 1], verbose=False, white_background=True):
        assert len(color) == 4
        # print('1:', process.memory_info().rss*1e-6)
        render_size = self.imsize
        bs = all_points.shape[0]
        if verbose:
            render_size = render_size*2
        outputs = []
        all_points = all_points*render_size
        num_ctrl_pts = torch.zeros(self.curves, dtype=torch.int32).to(all_points.device) + 2
        color = torch.tensor(color).to(all_points.device)
        for k in range(bs):
            # Get point parameters from network
            render = pydiffvg.RenderFunction.apply
            shapes = []
            shape_groups = []
            points = all_points[k].contiguous()#[self.sort_idx[k]] # .cpu()

            if verbose:
                np.random.seed(0)
                colors = np.random.rand(self.curves, 4)
                high = np.array((0.565, 0.392, 0.173, 1))
                low = np.array((0.094, 0.310, 0.635, 1))
                diff = (high-low)/(self.curves)
                colors[:, 3] = 1
                for i in range(self.curves):
                    scale = diff*i
                    color = low + scale
                    color[3] = 1
                    color = torch.tensor(color)
                    num_ctrl_pts = torch.zeros(1, dtype=torch.int32) + 2
                    if i*3 + 4 > self.curves * 3:
                        curve_points = torch.stack([points[i*3], points[i*3+1], points[i*3+2], points[0]])
                    else:
                        curve_points = points[i*3:i*3 + 4]
                    path = pydiffvg.Path(
                        num_control_points=num_ctrl_pts, points=curve_points,
                        is_closed=False, stroke_width=torch.tensor(4))
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([i]),
                        fill_color=None,
                        stroke_color=color)
                    shapes.append(path)
                    shape_groups.append(path_group)
                # for i in range(self.curves * 3):
                #     scale = diff*(i//3)
                #     color = low + scale
                #     color[3] = 1
                #     color = torch.tensor(color)
                #     if i%3==0:
                #         # color = torch.tensor(colors[i//3]) #green
                #         shape = pydiffvg.Rect(p_min = points[i]-8,
                #                              p_max = points[i]+8)
                #         group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.curves+i]),
                #                                            fill_color=color)
                #
                #     else:
                #         # color = torch.tensor(colors[i//3]) #purple
                #         shape = pydiffvg.Circle(radius=torch.tensor(8.0),
                #                                  center=points[i])
                #         group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.curves+i]),
                #                                            fill_color=color)
                #     shapes.append(shape)
                #     shape_groups.append(group)

            else:

                path = pydiffvg.Path(
                    num_control_points=num_ctrl_pts, points=points,
                    is_closed=True)

                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=color,
                    stroke_color=color)
                shape_groups.append(path_group)
            scene_args = pydiffvg.RenderFunction.serialize_scene(render_size, render_size, shapes, shape_groups)
            out = render(render_size,  # width
                         render_size,  # height
                         3,  # num_samples_x
                         3,  # num_samples_y
                         102,  # seed
                         None,
                         *scene_args)
            out = out.permute(2, 0, 1).view(4, render_size, render_size)#[:3]#.mean(0, keepdim=True)
            outputs.append(out)
        output =  torch.stack(outputs).to(all_points.device)

        # map to [-1, 1]
        if white_background:
            alpha = output[:, 3:4, :, :]
            output_white_bg = output[:, :3, :, :]*alpha + (1-alpha)
            output = torch.cat([output_white_bg, alpha], dim=1)
        del num_ctrl_pts, color
        return output

    def raster_bezier(self, all_points, color=[0,0,0, 1], verbose=False, white_background=True):
        assert len(color) == 4
        # print('1:', process.memory_info().rss*1e-6)
        render_size = self.imsize
        bs = all_points.shape[0]
        outputs = []
        # print(all_points*render_size)
        all_points = all_points + 0.5
        all_points = all_points * render_size
        # all_points = all_points * 2
        # print(all_points)
        eps = 1e-4
        all_points = all_points + eps*torch.randn_like(all_points)
        num_ctrl_pts = torch.zeros(1, dtype=torch.int32).to(all_points.device) + (self.num_points - 2)
        color = torch.tensor(color).to(all_points.device)
        for k in range(bs):
            # Get point parameters from network
            render = pydiffvg.RenderFunction.apply
            shapes = []
            shape_groups = []
            points = all_points[k].contiguous()#[self.sort_idx[k]] # .cpu()
            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                stroke_width=torch.tensor(self.circle_rad).to(all_points.device),
                is_closed=False)

            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color)
            shape_groups.append(path_group)
            scene_args = pydiffvg.RenderFunction.serialize_scene(render_size, render_size, shapes, shape_groups)
            out = render(render_size,  # width
                         render_size,  # height
                         2,  # num_samples_x
                         2,  # num_samples_y
                         102,  # seed
                         None,
                         *scene_args)
            out = out.permute(2, 0, 1).view(4, render_size, render_size)#[:3]#.mean(0, keepdim=True)
            outputs.append(out)
        output =  torch.stack(outputs).to(all_points.device)

        # map to [-1, 1]
        if white_background:
            alpha = output[:, 3:4, :, :]
            output_white_bg = output[:, :3, :, :]*alpha + (1-alpha)
            output = torch.cat([output_white_bg, alpha], dim=1)
        # del num_ctrl_pts
        del color
        return output, all_points


    def decode(self, z: torch.Tensor, point_predictor=None, verbose=False) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (torch.Tensor) [B x D]
        :return: (torch.Tensor) [B x C x H x W]
        """
        if point_predictor==None:
            point_predictor = self.point_predictor
        self.id = self.id.to(z.device)

        bs = z.shape[0]
        z = z[:, None, :].repeat([1, self.curves *3, 1])
        base_control_features = self.base_control_features[None, :, :].repeat(bs, self.curves, 1 )
        z_base = torch.cat([z, base_control_features], dim=-1)
        z_base_transform = self.decode_transform(z_base)
        if self.learn_sampling:
            self.angles = self.angles.to(z.device)
            angles= self.angles[None, :, None].repeat(bs, 1, 1)
            x = torch.cos(angles)# + r
            y = torch.sin(angles)# + r
            z_angles = torch.cat([z_base, x, y], dim=-1)

            angles_delta = self.sample_deformation(self.decode_transform(z_angles))
            angles_delta = F.tanh(angles_delta/50)*self.deformation_range
            angles_delta = self.decode_transform(angles_delta)

            new_angles = angles + angles_delta
            x = (torch.cos(new_angles) * self.circle_rad)# + r
            y = (torch.sin(new_angles) * self.circle_rad)# + r
            z = torch.cat([z_base, x, y], dim=-1)
        else:
            id = self.id[None, :, :].repeat(bs, 1, 1)
            z = torch.cat([z_base, id], dim=-1)

        all_points = self.decoder_input(self.decode_transform(z))
        for compute_block in point_predictor:
            all_points = F.relu(all_points)
            # all_points = torch.cat([z_base_transform, all_points], dim=1)
            all_points = compute_block(all_points)
        all_points = self.decode_transform(F.sigmoid(all_points/self.scale_factor))
        return all_points

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (torch.Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (torch.Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (torch.Tensor) [B x D]
        """
        if self.reparametrize_:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        all_points = self.decode(z)
        if not self.only_auxillary_training or self.save_lossvspath:
            output = self.raster(all_points, white_background=True)
        else:
            output = torch.zeros([1,3,64,64])
        return  [output, input, mu, log_var]

    def bilinear_downsample(self, tensor: torch.Tensor, size):
        return torch.nn.functional.interpolate(tensor, size, mode='bilinear')

    def gaussian_pyramid_loss(self, recons, input):
        recon_loss = self.loss_fn(recons, input, reduction='none').mean(dim=[1,2,3]) + self.seg_loss_weight * self.seg_loss_fn(recons, input) #+ self.lpips(recons, input)*0.1
        for j in range(2,5):
            recons = dsample(recons)
            input = dsample(input)
            recon_loss = recon_loss + (self.loss_fn(recons, input, reduction='none').mean(dim=[1,2,3]) + self.seg_loss_weight * self.seg_loss_fn(recons, input))
        return recon_loss

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = kwargs["image"][:, :3, :, :]
        recons = enforce_grayscale(recons)
        input = kwargs["target_image"]
        mu = 0
        log_var = 0
        other_losses = 0
        if len(args)==5:
            other_losses = 0
        aux_loss = 0
        kld_loss = 0

        kld_weight = kwargs.get('M_N', 1) # Account for the minibatch samples from the dataset
        if not self.only_auxillary_training or self.save_lossvspath:
            recon_loss = self.gaussian_pyramid_loss(recons, input)
        else:
            recon_loss = torch.zeros([1])
        if self.only_auxillary_training:
            recon_loss_non_reduced = recon_loss[:, None].clone().detach()
            spacing = self.aux_network(mu.clone().detach())
            latents = mu.cpu().numpy()
            num_latents = latents.shape[0]
            if self.save_lossvspath:
                recon_loss_non_reduced_cpu = recon_loss_non_reduced.cpu().numpy()
                keys  = self.latent_lossvpath.keys()
                for i in range(num_latents):
                    if np.array2string(latents[i]) in keys:
                        pair = torch.tensor([self.curves, recon_loss_non_reduced_cpu[i, 0], ])[None, :].to(mu.device)
                        self.latent_lossvpath[np.array2string(latents[i])]\
                            = torch.cat([self.latent_lossvpath[np.array2string(latents[i])], pair], dim=0)
                    else:
                        self.latent_lossvpath[np.array2string(latents[i])] = torch.tensor([[self.curves, recon_loss_non_reduced_cpu[i, 0]], ]).to(mu.device)
                num = torch.ones_like(spacing[:, 0]) * self.curves
                est_loss = spacing[:,2] + 1/torch.exp(num*spacing[:,0] - spacing[:,1])
                # est_loss = spacing[:, 2] + (spacing[i, 0] / num)

                aux_loss = torch.abs(num*(est_loss - recon_loss_non_reduced)).mean() * 10
            else:
                aux_loss = 0
                for i in range(num_latents):
                    pair = self.latent_lossvpath[np.array2string(latents[i])]
                    est_loss = spacing[i, 2] + 1 / torch.exp(pair[:, 0] * spacing[i, 0] - spacing[i, 1])

                    # est_loss = spacing[i, 2] + (spacing[i, 0] / pair[:, 0])
                    aux_loss = aux_loss + torch.abs(pair[:, 0]*(est_loss - pair[:, 1])).mean()
            loss =  aux_loss
            kld_loss = 0#self.beta*kld_weight * kld_loss
            logs = {'Reconstruction_Loss': recon_loss.mean(), 'KLD': -kld_loss, 'aux_loss': aux_loss}
            return {'loss': loss, 'progress_bar': logs }

        recon_loss = recon_loss.mean()
        if self.beta>0:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            kld_loss = self.beta*kld_weight * kld_loss
        recon_loss = recon_loss * self.raster_loss_weight

        # Vector supervision loss.
        bezier_points = kwargs.get("bezier_points", None)
        recon_loss_vec = 0
        if bezier_points is not None:
            input_points = kwargs["target_points"]
            bs, nr_paths, _, _ = bezier_points.shape
            path_losses = torch.empty((bs, nr_paths, nr_paths), device=bezier_points.device)
            for i in range(nr_paths):
                for j in range(nr_paths):
                    path_losses[:, i, j] = F.mse_loss(bezier_points[:, i], input_points[:, j])
            recon_loss_per_sample = torch.empty(bs, device=bezier_points.device)
            path_losses_clone = path_losses.clone().cpu().detach()
            for i in range(bs):
                matches = linear_sum_assignment(path_losses_clone[i])
                recon_loss_per_sample[i] = path_losses[i, matches[0], matches[1]].sum()
            recon_loss_vec = recon_loss_per_sample.mean()

        loss =  recon_loss + self.vector_loss_weight * recon_loss_vec + kld_loss + other_losses*self.other_losses_weight
        logs = {'Reconstruction_Loss': recon_loss.item(), "Reconstruction_Loss_Vec": recon_loss_vec.item(), 'KLD': -kld_loss, 'aux_loss': aux_loss, 'other losses': other_losses*self.other_losses_weight}
        return {'loss': loss, 'progress_bar': logs }

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        all_points = self.decode(z)
        samples = self.raster(all_points)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  self.raster(self.decode(z), verbose=random.choice([True, False]))
 # .type(torch.Floattorch.Tensor).to(device)

    def save(self, x, save_dir, name):
        z, log_var = self.encode(x)
        all_points = self.decode(z)
        # print(all_points.std(dim=1))
        # all_points = ((all_points-0.5)*2 + 0.5)*self.imsize
        # if type(self.sort_idx) == type(None):
        #     angles = torch.atan(all_points[:,:,1]/all_points[:,:,0]).detach()
        #     self.sort_idx = torch.argsort(angles, dim=1)
        # Process the batch sequentially
        outputs = []
        for k in range(1):
            # Get point parameters from network
            shapes = []
            shape_groups = []
            points = all_points[k].cpu()#[self.sort_idx[k]]

            color = torch.cat([torch.tensor([0,0,0,1]),])
            num_ctrl_pts = torch.zeros(self.curves, dtype=torch.int32) + 2

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                is_closed=True)

            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=color,
                stroke_color=color)
            shape_groups.append(path_group)
            pydiffvg.save_svg(f"{save_dir}{name}/{name}.svg",
                              self.imsize, self.imsize, shapes, shape_groups)


    def interpolate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(mu.shape[0]):
            z = self.interpolate_vectors(mu[2], mu[i], 10)
            all_points = self.decode(z)
            all_interpolations.append(self.raster(all_points, verbose=kwargs['verbose']))
        return all_interpolations

    def interpolate2D(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        y_axis = self.interpolate_vectors(mu[7], mu[6], 10)
        for i in range(10):
            z = self.interpolate_vectors(y_axis[i], mu[3], 10)
            all_points = self.decode(z)
            all_interpolations.append(self.raster(all_points, verbose=kwargs['verbose']))
        return all_interpolations


    def naive_vector_interpolate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_points = self.decode(mu)
        all_interpolations = []
        for i in range(mu.shape[0]):
            z = self.interpolate_vectors(all_points[2], all_points[i], 10)
            all_interpolations.append(self.raster(z, verbose=kwargs['verbose']))
        return all_interpolations

    def visualize_sampling(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(5,27):
            self.redo_features(i)
            all_points = self.decode(mu)
            all_interpolations.append(self.raster(all_points, verbose=kwargs['verbose']))
        return all_interpolations

    def sampling_error(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """
        error = []
        figure = plt.figure(figsize=(6, 6))
        bs = x.shape[0]
        for i in range(7,25):
            self.redo_features(i)
            results = self.forward(x)
            recons = results[0][:, :3, :, :]
            input_batch = results[1]

            recon_loss = self.gaussian_pyramid_loss(recons, input_batch)
            print(recon_loss)
            error.append(recon_loss)
        etn = torch.stack(error, dim=1).numpy()
        np.savetxt('sample_error.csv', etn, delimiter=',')
        y = np.arange(7,25)
        for i in range(bs):
            plt.plot(y, etn[i,:], label=str(i+1))
        plt.legend(loc='upper right')
        img = fig2data(figure)
        return img

    def visualize_aux_error(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        bs = mu.shape[0]
        all_spacing = []
        figure = plt.figure(figsize=(6, 6))

        for i in np.arange(7,25):
            spacing = self.aux_network(mu.clone().detach())
            num = torch.ones_like(spacing[:,0])*i
            # est_loss = spacing[:,2] + 1/torch.exp(num*spacing[:,0] + spacing[:,1])
            est_loss =     spacing[:,2] + (spacing[:,0]/num)

            # print(i, spacing[0])
            all_spacing.append(est_loss)
        all_spacing = torch.stack(all_spacing, dim=1).detach().cpu().numpy()
        y = np.arange(7,25)
        for i in range(bs):
            plt.plot(y, all_spacing[i,:], label=str(i+1))
        plt.legend(loc='upper right')
        img = fig2data(figure)
        return img

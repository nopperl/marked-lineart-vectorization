import torch
from torch import nn
from torchvision.models import resnet50

from marked_lineart_vec.util import reparameterize


class SimpleConvEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, imsize=64, stride=2, hidden_dims=None, reparameterize=False, weights=None, global_pooling=False):
        super(SimpleConvEncoder, self).__init__()
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.global_pooling = global_pooling
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim
        if global_pooling:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels=latent_dim,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(latent_dim),
                nn.ReLU())
            )
        self.encoder = nn.Sequential(*modules)
        if global_pooling:
            self.pool_method = nn.AdaptiveAvgPool2d(1)
        else:
            outsize = int(imsize / (stride ** len(hidden_dims)))
            self.fc_mu = nn.Linear(hidden_dims[-1] * outsize * outsize, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1] * outsize * outsize, latent_dim)
        self.reparameterize = reparameterize
        if weights is not None:
            pretrained_dict = torch.load(weights)["state_dict"]
            pretrained_dict = {k.replace("model.image_encoder.", ""): v for k, v in pretrained_dict.items() if k.startswith("model.image_encoder.")}
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward(self, x: torch.Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.

        :param x: (torch.Tensor) Input torch.Tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of latent codes
        """
        result = self.encoder(x)
        if self.global_pooling:
            z = self.pool_method(result)
            z = z.squeeze(dim=2)
            z = z.squeeze(dim=2)
            return z
        else:
            result = torch.flatten(result, start_dim=1)
            if self.reparameterize:
                mu = self.fc_mu(result)
                log_var = self.fc_var(result)
                z = reparameterize(mu, log_var)
                return z, mu, log_var
            else:
                z = self.fc_mu(result)
                return z


class ResnetEncoder(nn.Module):
    def __init__(self, reparameterize=False):
        super(ResnetEncoder, self).__init__()
        backbone = resnet50(pretrained=False)
        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
        self.pool_method = nn.AdaptiveMaxPool2d(1)
        self.reparameterize = reparameterize
        if reparameterize:
            self.fc_mu = nn.Linear(2048, 2048)
            self.fc_var = nn.Linear(2048, 2048)

    def forward(self, x):
        x = self.features(x)
        x = self.pool_method(x).view(-1, 2048)
        if self.reparameterize:
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            z = reparameterize(mu, log_var)
            return z, mu, log_var
        return x

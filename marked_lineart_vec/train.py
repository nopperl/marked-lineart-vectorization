#!/usr/bin/env python3
from os import listdir, getcwd
from os.path import join, getmtime, exists, dirname
from shutil import copytree, ignore_patterns, rmtree
import json

import argparse
import numpy as np

from pytorch_lightning.loggers import TensorBoardLogger
from yaml import safe_load

from marked_lineart_vec.models import *
from marked_lineart_vec.experiment import VAEExperiment
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
import click

rootdir = getcwd()
parser = argparse.ArgumentParser(description='Generic train script for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/iterative.yaml')

args = parser.parse_args()

with open(args.filename, 'r') as yaml_file:
    config = safe_load(yaml_file)

tb_logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    sub_dir="tensorboard",
    version=config['logging_params']['version'],
)

resume = False
version_dir = dirname(tb_logger.log_dir)
print(version_dir)
# Copying the folder
if exists(version_dir):
    if click.confirm('Folder exists do you want to override?', default=True):
        rmtree(version_dir)
        copytree(rootdir, join(version_dir, "code"), ignore=ignore_patterns('*.pyc', 'tmp*', 'logs*', 'data*', 'venv', ".*"))
    else:
        resume = True
else:
    copytree(rootdir, join(version_dir, "code"), ignore=ignore_patterns('*.pyc', 'tmp*', 'logs*', 'data*', 'venv', ".*"))

# with open(model_save_path + 'hyperparameters.txt', 'w') as f:
#     json.dump(args.__dict__, f, indent=2)


# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False
print(config['model_params'])

experiment = VAEExperiment(config["model_params"], config['exp_params'])

model_path = None
if resume:
    checkpoint_dir = join(version_dir, "checkpoints")
    weights = [join(checkpoint_dir, x) for x in listdir(checkpoint_dir) if '.ckpt' in x]
    weights.sort(key=lambda x: getmtime(x))
    if len(weights) > 0:
        model_path = weights[-1]
        print('loading: ', weights[-1])

print(config['exp_params'], config['logging_params']['save_dir']+config['logging_params']['name'])
runner = Trainer(enable_checkpointing=True,
                 logger=tb_logger,
                 log_every_n_steps=100,
                 # track_grad_norm=2,
                 stochastic_weight_avg=True,
                 # reload_dataloaders_every_n_epochs=10,
                 # gradient_clip_val=0.5,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, ckpt_path=model_path)

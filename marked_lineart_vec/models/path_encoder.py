from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F


class CnnPathEncoder(nn.Module):
    def __init__(self, num_points=2, point_embedding_size=32, path_hidden_sizes: Optional[List] = None, path_embedding_size=128):
        """
        Convolutional encoder for parameterized paths
        Args:
            num_points:
            point_embedding_size:
            path_hidden_sizes: Length imposes constraint on minimum sequence length
            path_embedding_size:
        """
        super(CnnPathEncoder, self).__init__()
        if path_hidden_sizes is None:
            path_hidden_sizes = [64]
        self.point_conv = nn.Sequential(
            nn.Conv2d(num_points, point_embedding_size, kernel_size=(1, 2), stride=(1, 1)),
            # nn.BatchNorm2d(point_embedding_size),
            nn.ReLU(),
        )
        self.path_conv = nn.Sequential()
        for idx, path_hidden_size in enumerate(path_hidden_sizes):
            self.path_conv.add_module(
                str(idx),
                nn.Sequential(
                    nn.Conv1d(in_channels=point_embedding_size, out_channels=path_hidden_size, kernel_size=2, stride=2),
                    nn.ReLU(),
                )
            )
        self.path_conv.add_module(
            str(len(path_hidden_sizes)),
            nn.Sequential(
                # nn.Conv1d(in_channels=path_hidden_sizes[-1], out_channels=path_embedding_size, kernel_size=2, stride=2),
                nn.Conv1d(in_channels=path_hidden_sizes[-1], out_channels=path_embedding_size, padding=1, kernel_size=2, stride=2),
                nn.ReLU(),
            )
        )

    def forward(self, paths):
        """

        Args:
            paths: [batch_size,num_paths,num_points,2]

        Returns:
            path_embedding: [batch_size,path_embedding_size]
        """
        paths = paths.transpose(1, 3)
        paths = paths.transpose(2, 3)
        point_embedding = self.point_conv(paths)
        point_embedding = point_embedding.squeeze(-1)
        path_embeddings = self.path_conv(point_embedding)
        # Global average pooling
        path_embedding = F.adaptive_avg_pool1d(path_embeddings, output_size=1)
        path_embedding = path_embedding.squeeze(-1)
        return path_embedding


class RnnPathEncoder(nn.Module):
    def __init__(self, hidden_size=128, path_embedding_size=128, use_last=True):
        super(RnnPathEncoder, self).__init__()
        self.canvas_point_rnn = nn.LSTM(2, int(hidden_size / 2), 2, bidirectional=True, batch_first=True)
        self.canvas_curve_rnn = nn.LSTM(hidden_size, int(path_embedding_size / 2), 2, bidirectional=True,
                                        batch_first=True)
        self.hidden_size = hidden_size
        self.use_last = use_last

    def forward(self, paths):
        bs, num_paths, num_points, num_coordinates = paths.shape
        path_embeddings = torch.empty(size=(bs, num_paths, self.hidden_size), device=paths.device)
        for path in range(num_paths):
            outputs, _ = self.canvas_point_rnn(paths[:, path])
            path_embeddings[:, path] = outputs[:, -1]
        outputs, _ = self.canvas_curve_rnn(path_embeddings)
        if self.use_last:
            paths_embedding = outputs[:, -1]
        else:
            paths_embedding = outputs.mean(dim=1)
        return paths_embedding

import torch
import torch.nn as nn


class ImageSequenceFusion(nn.Module):
    """ Concat + FC fusion strategy for image and sequence features """
    def __init__(self, img_feat_dim=128, seq_feat_dim=256, hidden_size=128, fused_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_fc = nn.Linear(seq_feat_dim, hidden_size)
        self.fc = nn.Linear(img_feat_dim + hidden_size, fused_size)

    def forward(self, img_feat, seq_feat):
        """

        Args:
            img_feat: [batch_size, img_feat_dim]
            seq_feat: [batch_size, seq_feat_dim]

        Returns: [batch_size, fused_size]
        """
        feat_cat = torch.cat((img_feat, self.seq_fc(seq_feat)), dim=-1)
        fused = self.fc(feat_cat)
        return fused

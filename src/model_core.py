"""
Two-Stream Neural Network for Face Forgery Detection

This module defines the core architecture of the detection model. It combines
spatial and frequency domain features through SRM filters, spatial/channel attention,
and dual cross-modal attention modules to detect forged face images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.attention import ChannelAttention, SpatialAttention, DualCrossModalAttention
from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from networks.xception import TransferModel


class SRMPixelAttention(nn.Module):
    """
    Spatial attention guided by SRM-filtered images.

    Applies SRM filtering followed by a convolutional encoder and a spatial
    attention mechanism to generate attention maps highlighting forgery regions.
    """
    
    def __init__(self, in_channels):
        super(SRMPixelAttention, self).__init__()
        self.srm = SRMConv2d_simple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.pa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input RGB image.

        Returns:
            Tensor: Attention map.
        """
        
        x_srm = self.srm(x)
        fea = self.conv(x_srm)        
        att_map = self.pa(fea)
        
        return att_map


class FeatureFusionModule(nn.Module):
    """
    Feature Fusion using channel attention.

    Fuses RGB and SRM feature maps using 1x1 conv and channel attention.
    """
    
    def __init__(self, in_chan=2048*2, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        """
        Args:
            x (Tensor): Feature from RGB stream.
            y (Tensor): Feature from SRM stream.

        Returns:
            Tensor: Fused feature map.
        """
        
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        """Initialize weights for conv layers."""
        
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Two_Stream_Net(nn.Module):
    """
    Main model combining RGB and SRM feature extraction streams.

    This two-stream architecture extracts features from both the spatial (RGB)
    and frequency (SRM) domains, then fuses them with attention mechanisms to
    perform classification.
    """
    
    def __init__(self):
        super().__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)

        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)
        self.relu = nn.ReLU(inplace=True)

        self.att_map = None
        self.srm_sa = SRMPixelAttention(3)
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.dual_cma1 = DualCrossModalAttention(in_dim=728, ret_att=False)

        self.fusion = FeatureFusionModule()

        self.att_dic = {}

    def features(self, x):
        """
        Extract multi-modal features from input image.

        Args:
            x (Tensor): Input image tensor (B, 3, H, W).

        Returns:
            Tensor: Fused feature representation.
        """
        
        srm = self.srm_conv0(x)

        x = self.xception_rgb.model.fea_part1_0(x)
        y = self.xception_srm.model.fea_part1_0(srm) \
            + self.srm_conv1(x)
        y = self.relu(y)

        x = self.xception_rgb.model.fea_part1_1(x)
        y = self.xception_srm.model.fea_part1_1(y) \
            + self.srm_conv2(x)
        y = self.relu(y)

        # srm guided spatial attention
        self.att_map = self.srm_sa(srm)
        x = x * self.att_map + x
        x = self.srm_sa_post(x)

        x = self.xception_rgb.model.fea_part2(x)
        y = self.xception_srm.model.fea_part2(y)

        x, y = self.dual_cma0(x, y)


        x = self.xception_rgb.model.fea_part3(x)        
        y = self.xception_srm.model.fea_part3(y)
 

        x, y = self.dual_cma1(x, y)

        x = self.xception_rgb.model.fea_part4(x)
        y = self.xception_srm.model.fea_part4(y)

        x = self.xception_rgb.model.fea_part5(x)
        y = self.xception_srm.model.fea_part5(y)

        fea = self.fusion(x, y)
                

        return fea

    def classifier(self, fea):
        """
        Apply classifier head to fused features.

        Args:
            fea (Tensor): Input feature tensor.

        Returns:
            Tuple: (output logits, feature representation)
        """
        
        out, fea = self.xception_rgb.classifier(fea)
        return out, fea

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input image tensor (B, 3, H, W)

        Returns:
            tuple: (logits, features, attention map)
        """
        
        out, fea = self.classifier(self.features(x))

        return out, fea, self.att_map
    
if __name__ == '__main__':
    model = Two_Stream_Net()
    dummy = torch.rand((1,3,256,256))
    out = model(dummy)
    print(model)
    
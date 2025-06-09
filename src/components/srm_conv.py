"""
SRM Convolution Modules for Frequency-Aware Feature Extraction

Implements simple and separate versions of SRMConv2D using hand-crafted filters.
These modules are used to extract high-frequency noise features, which are useful
for detecting subtle artifacts in image forensics tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    

class SRMConv2d_simple(nn.Module):
    """
    Simple SRM Convolution Layer.

    Applies 3 fixed high-pass filters across input channels using 2D convolution.
    These filters help emphasize high-frequency noise patterns.
    """
    
    def __init__(self, inc=3, learnable=False):
        """
        Args:
            inc (int): Number of input channels.
            learnable (bool): Whether the SRM filters are learnable.
        """
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Output tensor after SRM filtering and truncation
        """
        
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        """
        Builds 3 SRM high-pass filters and applies them to all input channels.

        Returns:
            Tensor: Filters of shape (3, inc, 5, 5)
        """
        
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],#, filter1, filter1],
                   [filter2],#, filter2, filter2],
                   [filter3]]#, filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)    # (3,3,5,5)
        return filters

class SRMConv2d_Separate(nn.Module):
    """
    Separate SRM Convolution Layer.

    Applies grouped SRM filtering on each input channel, then projects the
    combined output to a target number of channels.
    """
    
    def __init__(self, inc, outc, learnable=False):
        """
        Args:
            inc (int): Number of input channels.
            outc (int): Number of output channels.
            learnable (bool): Whether the SRM filters are learnable.
        """
        
        super(SRMConv2d_Separate, self).__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(3*inc, outc, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

        for ly in self.out_conv.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Processed features after SRM and projection
        """
        
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=self.inc)
        out = self.truc(out)
        out = self.out_conv(out)

        return out

    def _build_kernel(self, inc):
        """
        Builds SRM filters for each channel separately.

        Returns:
            Tensor: Filters of shape (3 * inc, 1, 5, 5)
        """
        
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],#, filter1, filter1],
                   [filter2],#, filter2, filter2],
                   [filter3]]#, filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        # filters = np.repeat(filters, inc, axis=1)
        filters = np.repeat(filters, inc, axis=0)
        filters = torch.FloatTensor(filters)    # (3,3,5,5)
        # print(filters.size())
        return filters



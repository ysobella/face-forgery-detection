"""
AM-Softmax Loss and Angle-Based Linear Layer

Implements the Angular Margin Softmax Loss (AM-Softmax) for classification tasks,
often used in face verification/recognition tasks to improve inter-class separability.

Includes:
- AngleSimpleLinear: Computes cosine similarity between inputs and learned weights.
- AMSoftmaxLoss: Margin-based softmax loss with optional cosine or arc margin.
- focal_loss: Optional focal loss wrapper.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class AngleSimpleLinear(nn.Module):
    """
    Angular linear layer that computes cosine similarity between input features and weight vectors.

    This is typically used before applying an angular margin-based softmax loss.
    """
    
    def __init__(self, in_features, out_features):
        """
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Number of classes / output units.
        """
        
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        """
        Computes cosine similarity between input and normalized weight vectors.

        Args:
            x (Tensor): Input tensor of shape (B, in_features)

        Returns:
            Tensor: Cosine similarities (B, out_features)
        """
        
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)


def focal_loss(input_values, gamma):
    """
    Computes the focal loss.

    Args:
        input_values (Tensor): Cross-entropy loss per sample.
        gamma (float): Focusing parameter for modulating factor (1 - p_t)^gamma.

    Returns:
        Tensor: Scalar focal loss.
    """
    
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class AMSoftmaxLoss(nn.Module):
    """
    Additive Margin Softmax Loss (AM-Softmax).

    Supports two types of margin: 'cos' and 'arc'. Improves decision margin in angular space.

    Attributes:
        margin_type (str): Type of margin ('cos' or 'arc').
        gamma (float): Focal loss gamma. If 0, no focal loss is applied.
        m (float): Margin value.
        s (float): Scaling factor.
        t (float): Optional hard example mining scalar.
    """
    
    margin_types = ['cos', 'arc']

    def __init__(self, margin_type='cos', gamma=0., m=0.5, s=30, t=1.):
        """
        Args:
            margin_type (str): 'cos' or 'arc' margin.
            gamma (float): Focal loss gamma.
            m (float): Margin value to subtract or modify angle.
            s (float): Scaling factor for logits.
            t (float): Threshold adjustment for hard samples (t >= 1).
        """
        
        super(AMSoftmaxLoss, self).__init__()
        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m > 0
        self.m = m
        assert s > 0
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        assert t >= 1
        self.t = t

    def forward(self, cos_theta, target):
        """
        Forward pass to compute AM-Softmax loss.

        Args:
            cos_theta (Tensor): Cosine similarity predictions (B, num_classes)
            target (Tensor): Ground truth labels (B,)

        Returns:
            Tensor: Loss value
        """
        
        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.m
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m #cos(theta+m)
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        if self.gamma == 0 and self.t == 1.:
            return F.cross_entropy(self.s*output, target)

        if self.t > 1:
            h_theta = self.t - 1 + self.t*cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)
            return F.cross_entropy(self.s*output, target)

        return focal_loss(F.cross_entropy(self.s*output, target, reduction='none'), self.gamma)
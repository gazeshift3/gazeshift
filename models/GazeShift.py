import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.core import LightningModule
from sklearn.preprocessing import PolynomialFeatures
from sklearn.manifold import MDS
from PIL import Image, ImageOps
import random
import torchvision
import pandas as pd
from models.mbnv2 import MobileNet_v2
from models.VAE import VAE
import matplotlib.pyplot as plt
import torch.nn.init as init
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh



import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import math
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class PositionalEncoding2D(nn.Module):
    def __init__(self, C, N):
        super(PositionalEncoding2D, self).__init__()
        # Learnable positional encoding with shape [1, C, N, N] (broadcastable across batch)
        self.positional_encoding = nn.Parameter(torch.randn(1, C, N, N))

    def forward(self, x):
        """
        x: Input tensor of shape [B, C, N, N]
        Returns: Tensor with positional encoding added
        """
        return x + self.positional_encoding



# class PositionalEncoding2D(nn.Module):
#     """
#     2D Sinusoidal Positional Encoding for image patches.
#
#     Args:
#         d_model: Embedding dimension (must be even).
#         height:  Height of the 2D grid.
#         width:   Width of the 2D grid.
#
#     Usage:
#         x shape: (B, d_model, H, W)
#         out = x + pe, where pe has shape (1, d_model, H, W)
#     """
#
#     def __init__(self, d_model, height, width):
#         super().__init__()
#         if d_model % 2 != 0:
#             raise ValueError("d_model must be even for 2D positional encoding.")
#
#         # Create a buffer so it's not trainable
#         pe = torch.zeros(d_model, height, width)  # (d_model, H, W)
#
#         # d_model is split into d_model/2 for Y, d_model/2 for X
#         d_model_half = d_model // 2
#
#         # Each half is further split into sin/cos pairs:
#         # so we only need d_model_half/2 "pairs" for y and x each
#         # but typically you see it as loops over i in [0, d_model_half) stepping by 2.
#
#         # -- For Y (rows) --
#         # shape: (height, )
#         y_pos = torch.arange(height, dtype=torch.float).unsqueeze(1)  # (H, 1)
#         div_term_y = torch.exp(
#             torch.arange(0, d_model_half, 2).float()
#             * -(math.log(10000.0) / d_model_half)
#         )  # (d_model_half/2, )
#
#         # pe for y of shape: (d_model_half, height)
#         # We'll broadcast sin/cos across each row
#         for i in range(0, d_model_half, 2):
#             i2 = i // 2  # index into div_term_y
#             # sin
#             pe[i, :, :] = torch.sin(y_pos * div_term_y[i2]).transpose(0, 1)
#             # cos
#             pe[i + 1, :, :] = torch.cos(y_pos * div_term_y[i2]).transpose(0, 1)
#
#         # -- For X (columns) --
#         # shape: (width, )
#         x_pos = torch.arange(width, dtype=torch.float).unsqueeze(1)  # (W, 1)
#         div_term_x = torch.exp(
#             torch.arange(0, d_model_half, 2).float()
#             * -(math.log(10000.0) / d_model_half)
#         )  # (d_model_half/2, )
#
#         # pe for x of shape: (d_model_half, width)
#         # We'll broadcast sin/cos across each column, but must offset the index by d_model_half
#         for i in range(0, d_model_half, 2):
#             i2 = i // 2
#             # sin
#             pe[d_model_half + i, :, :] = torch.sin(x_pos * div_term_x[i2]).transpose(0, 1)
#             # cos
#             pe[d_model_half + i + 1, :, :] = torch.cos(x_pos * div_term_x[i2]).transpose(0, 1)
#
#         # shape is (d_model, H, W)
#         pe = pe.unsqueeze(0)  # (1, d_model, H, W)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Add 2D positional embeddings to the input feature map.
#
#         x shape: (B, d_model, H, W)
#         returns: (B, d_model, H, W)
#         """
#         # Make sure the spatial dims match
#         _, _, H, W = x.shape
#         peH, peW = self.pe.shape[-2], self.pe.shape[-1]
#         if (H != peH) or (W != peW):
#             raise ValueError(
#                 f"PositionalEncoding2D mismatch: Input is {H}x{W}, but PE is {peH}x{peW}."
#             )
#         return x + self.pe  # broadcast over batch


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of the Transformer Decoder.

    d_model:         Dimensionality of embeddings (both tgt and memory).
    nhead:           Number of heads in multi-head attention.
    dim_feedforward: Dimensionality of the inner layer in the FFN.
    dropout:         Dropout probability.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # Self-Attention (decoder’s own attention)
        self.self_attn = nn.MultiheadAttention(
            batch_first=True,
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )

        # Cross-Attention (attending to the memory from the encoders)
        self.cross_attn = nn.MultiheadAttention(
            batch_first=True,
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )

        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorms and Dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            tgt,  # shape: (T, B, d_model)
            memory,  # shape: (S, B, d_model)
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            pos_encoding=None
    ):
        """
        Arguments:
          tgt:        Decoder input embeddings (e.g. appearance embeddings).
          memory:     Encoder output embeddings (e.g. gaze embeddings).
          tgt_mask:   Optional attention mask for tgt (shape [T, T]).
          memory_mask:Optional attention mask for memory (shape [T, S]).
          tgt_key_padding_mask:   ByteTensor mask for tgt keys per batch (shape [B, T]).
          memory_key_padding_mask:ByteTensor mask for memory keys per batch (shape [B, S]).

        Returns:
          out:        The updated decoder representations.
        """

        # ---- 1. Self-Attention on the decoder input (tgt) ----
        # shape of x2 is (T, B, d_model)
        #tgt is the appearance tokens
        #memory is the gaze tokens
        x2, sa_att_weights = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True,
            average_attn_weights=True

        )
        # Residual connection + layer normalization
        tgt = tgt + self.dropout(x2)
        tgt = self.norm1(tgt)

        # ---- 2. Cross-Attention (attend to the memory) ----
        memory = memory.repeat(1, tgt.shape[1], 1)
        memory = self.norm4(memory)
        x2, ca_att_weights = self.cross_attn(
            query=memory,
            key=tgt,
            value=tgt,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )

        #plt.imshow(sa_att_weights[0, 0, :].cpu().detach().reshape(10, 10).numpy(), cmap='gray')
        #self.att_maps = sa_att_weights.detach().sum(dim=1)
        self.att_maps = sa_att_weights.detach().sum(dim=1)
        # Residual connection + layer normalization
        tgt = tgt + self.dropout(x2)
        # tgt = tgt + self.dropout(x2)
        tgt = self.norm2(tgt)
        # tgt = tgt.sum(1).unsqueeze(1)
        #tgt = x2
        # ---- 3. Feed Forward ----
        # x2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        # tgt = tgt + self.dropout(x2)
        # tgt = self.norm3(tgt)

        return tgt


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer Decoder.

    d_model:         Dimensionality of embeddings (both tgt and memory).
    nhead:           Number of heads in multi-head attention.
    dim_feedforward: Dimensionality of the inner layer in the FFN.
    dropout:         Dropout probability.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # Self-Attention (decoder’s own attention)
        self.self_attn = nn.MultiheadAttention(
            batch_first=True,
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )


        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorms and Dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            tgt,  # shape: (T, B, d_model)
            pos_encoding=None
    ):
        """
        Arguments:
          tgt:        Decoder input embeddings (e.g. appearance embeddings).
          memory:     Encoder output embeddings (e.g. gaze embeddings).
          tgt_mask:   Optional attention mask for tgt (shape [T, T]).
          memory_mask:Optional attention mask for memory (shape [T, S]).
          tgt_key_padding_mask:   ByteTensor mask for tgt keys per batch (shape [B, T]).
          memory_key_padding_mask:ByteTensor mask for memory keys per batch (shape [B, S]).

        Returns:
          out:        The updated decoder representations.
        """

        # ---- 1. Self-Attention on the decoder input (tgt) ----
        # shape of x2 is (T, B, d_model)
        #tgt is the appearance tokens
        #memory is the gaze tokens
        x2, sa_att_weights = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            need_weights=True,
            average_attn_weights=False

        )
        # Residual connection + layer normalization
        tgt = tgt + self.dropout(x2)
        tgt = self.norm1(tgt)

        return tgt

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder made up of multiple TransformerDecoderLayers.

    d_model:   Dimensionality of embeddings.
    nhead:     Number of heads in multi-head attention.
    num_layers:Number of TransformerDecoderLayer layers to stack.
    """

    def __init__(
            self,
            d_model,
            nhead,
            num_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            dgaze=30
    ):
        super(TransformerDecoder, self).__init__()
        self.upscale_gaze = nn.Linear(dgaze, d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.num_layers = num_layers
        self.reduce_linear = nn.Linear(d_model*2, d_model)

    def forward(
            self,
            tgt,  # shape: (T, B, d_model)
            memory,  # shape: (S, B, d_model)
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
    ):
        """
        Arguments:
          tgt:        Decoder input embeddings (e.g. appearance embeddings).
          memory:     Encoder output embeddings (e.g. gaze embeddings).
          tgt_mask:   Optional attention mask for tgt.
          memory_mask:Optional attention mask for memory.
          tgt_key_padding_mask:   ByteTensor mask for tgt keys per batch.
          memory_key_padding_mask:ByteTensor mask for memory keys per batch.

        Returns:
          out:        The final decoder output embeddings.
        """

        B, C, H, W = tgt.shape
        tgt = tgt.view(B, C, H * W)  # [B, C, H*W]
        tgt = tgt.permute(0, 2, 1)  # [B, H*W, C]
        memory = self.upscale_gaze(memory).tanh()
        memory = memory.unsqueeze(1)

        output = tgt
        att_maps = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            att_maps.append(layer.att_maps.unsqueeze(1))
        att_maps = torch.cat(att_maps, dim=1).mean(dim=1)
        #att_maps = torch.cat(att_maps).mean(dim=1)
        #output = output + tgt
        #output = output.reshape(output.shape[0], -1)
        #output = self.reduce_linear(output).unsqueeze(1)
        #output = output.sum(dim=1).unsqueeze(1)
        #memory = memory.repeat(1, output.shape[1], 1)
        #output = torch.cat((output, memory), dim=-1)
        #output = self.reduce_linear(output)
        return output, att_maps


class TransformerEncoder(nn.Module):
    """
    Transformer Decoder made up of multiple TransformerDecoderLayers.

    d_model:   Dimensionality of embeddings.
    nhead:     Number of heads in multi-head attention.
    num_layers:Number of TransformerDecoderLayer layers to stack.
    """

    def __init__(
            self,
            d_model,
            nhead,
            num_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            dgaze=30
    ):
        super(TransformerEncoder, self).__init__()
        self.upscale_gaze = nn.Linear(dgaze, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.num_layers = num_layers
        self.simple_att = nn.Linear(d_model, 1)
        self.reduce_linear = nn.Linear(d_model*2, d_model)
        self.gaze_encoding = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(
            self,
            tgt,  # shape: (T, B, d_model)
            memory,  # shape: (S, B, d_model)
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
    ):
        """
        Arguments:
          tgt:        Decoder input embeddings (e.g. appearance embeddings).
          memory:     Encoder output embeddings (e.g. gaze embeddings).
          tgt_mask:   Optional attention mask for tgt.
          memory_mask:Optional attention mask for memory.
          tgt_key_padding_mask:   ByteTensor mask for tgt keys per batch.
          memory_key_padding_mask:ByteTensor mask for memory keys per batch.

        Returns:
          out:        The final decoder output embeddings.
        """

        B, C, H, W = tgt.shape
        app = tgt.view(B, C, H * W)  # [B, C, H*W]

        #s_att = self.simple_att(tgt).sigmoid()
        #app = app * s_att
        app = app.permute(0, 2, 1)  # [B, H*W, C]
        gaze = self.upscale_gaze(memory).unsqueeze(1)
        gaze = gaze + self.gaze_encoding
        all_data = torch.cat((gaze, app), dim=1)

        output = all_data
        for layer in self.layers:
            output = layer(
                output,
            )
        output = output[:,1:,:]
        #output = output + tgt
        #output = output.reshape(output.shape[0], -1)
        #output = self.reduce_linear(output).unsqueeze(1)
        #output = output.sum(dim=1).unsqueeze(1)
        #memory = memory.repeat(1, output.shape[1], 1)
        #output = torch.cat((output, memory), dim=-1)
        #output = self.reduce_linear(output)
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F

class Appearance2ImageDecoder(nn.Module):
    """
    Decoder that takes appearance features of shape (B, K, K, d) = (B,10,10,32)
    and decodes them into an image of shape (B, 1, 400, 400).
    """

    def __init__(self):
        super(Appearance2ImageDecoder, self).__init__()

        # We will do multiple upsampling steps (scale_factor=2) until we reach 320x320,
        # and then do one final upsampling to get 400x400.
        #
        # For simplicity, each step:
        #  1) Upsamples (bilinear interpolation)
        #  2) Applies a Conv2D
        #  3) Applies a ReLU
        #
        # You can adjust the intermediate channel sizes as needed.

        # 1) (B,32,10,10) -> (B,64,20,20)
        channels = 64
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # 2) (B,64,20,20) -> (B,128,40,40)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=channels*2, out_channels=channels*4, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # 3) (B,128,40,40) -> (B,64,80,80)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=channels*4, out_channels=channels*2, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # 4) (B,64,80,80) -> (B,32,160,160)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # 5) (B,32,160,160) -> (B,16,320,320)
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=channels, out_channels=channels//2, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # 6) (B,16,320,320) -> (B,8,400,400)
        # Here, we specify size=(400, 400) because it's not a clean x2 factor from 320 to 400.
        self.up6 = nn.Sequential(
            nn.Upsample(size=(400, 400), mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=channels//2, out_channels=channels//4, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # Final 1x1 convolution: (B,8,400,400) -> (B,1,400,400)
        self.final_conv = nn.Conv2d(in_channels=channels//4, out_channels=1, kernel_size=1)

    def forward(self, x):
        """
        x is of shape (B, K, K, d) = (B, 10, 10, 32).
        We permute it to (B, d, K, K) = (B, 32, 10, 10) for 2D CNNs.
        """
        # Permute to (B, 32, 10, 10)
        x = x.permute(0, 3, 1, 2)

        x = self.up1(x)   # -> (B,64,20,20)
        x = self.up2(x)   # -> (B,128,40,40)
        x = self.up3(x)   # -> (B,64,80,80)
        x = self.up4(x)   # -> (B,32,160,160)
        x = self.up5(x)   # -> (B,16,320,320)
        x = self.up6(x)   # -> (B,8,400,400)

        x = self.final_conv(x)  # -> (B,1,400,400)
        return x



# Define the MLP class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_size
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.Tanh())  # Activation function
            in_dim = hidden_size
        # Add the output layer
        layers.append(nn.Linear(in_dim, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



def diffusion_map(X, sigma=1, dim=2):
    # Compute the affinity matrix
    distances = squareform(pdist(X, 'euclidean'))
    K = np.exp(-distances ** 2 / sigma ** 2)

    # Construct the Markov matrix
    row_sums = K.sum(axis=1)
    P = K / row_sums[:, np.newaxis]

    # Eigen decomposition
    eigenvalues, eigenvectors = eigh(P)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Return the projection onto the first 'dim' non-trivial eigenvectors
    return eigenvectors[:, 1:dim + 1]


class EyeIdentityEncoder(torch.nn.Module):
    def __init__(self, eye_id_dim, variational=False):
        super(EyeIdentityEncoder, self).__init__()
        bottleneckLayerDetails = [
            (6, 4, 1, 2),
            (6, 8, 1, 2),
            (6, 16, 1, 2),
            (6, 128, 1, 2),
        ]
        self.eyeid_encoder = MobileNet_v2(bottleneckLayerDetails, width_multiplier=2, in_fts=1)
        ouput_dim = 10*10*128
        #self.head = torch.nn.Linear(ouput_dim, eye_id_dim)
        self.head = MLP(ouput_dim, [ouput_dim // 2], eye_id_dim)
        self.fc_mean = torch.nn.Linear(ouput_dim, eye_id_dim)
        self.fc_log_var = torch.nn.Linear(ouput_dim, eye_id_dim)
        self.variational = variational
        # self.left_eye_id = nn.Parameter(torch.randn(1, 256))
        # self.right_eye_id = nn.Parameter(torch.randn(1, 256))


    def forward(self, images):
        x = self.eyeid_encoder(images).squeeze()
        # if is_left_eye:
        #     x = self.left_eye_id.view(1, 256,1,1) + x
        # else:
        #     x = self.right_eye_id.view(1, 256,1,1) + x

        #x = x.reshape(x.shape[0], -1)
        if self.variational:
            mu = self.fc_mean(x)
            log_var = self.fc_log_var(x)
            z = reparameterize(mu, log_var)
        else:
            z = x
            #z = self.head(x)
        return z

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

class GazeEncoder(torch.nn.Module):
    def __init__(self, gaze_dim, feature_size=128, varitional=False):
        super(GazeEncoder, self).__init__()
        bottleneckLayerDetails = [
            (1, 4, 1, 2),
            (2, 8, 1, 2),
            (2, 16, 1, 2),
            (2, 32, 1, 2),
            (2, 48, 1, 2),
            (2, 64, 1, 2),
        ]
        self.gaze_encoder = MobileNet_v2(bottleneckLayerDetails, width_multiplier=2, in_fts=1)
        self.head = torch.nn.Linear(feature_size, gaze_dim)
        #self.head = MLP(feature_size, [feature_size * 2], gaze_dim)
        self.fc_mean = torch.nn.Linear(feature_size, gaze_dim)
        self.fc_log_var = torch.nn.Linear(feature_size, gaze_dim)
        self.varitional = varitional
        # self.left_eye_id = nn.Parameter(torch.randn(1, feature_size))
        # self.right_eye_id = nn.Parameter(torch.randn(1, feature_size))

    def forward(self, images):
        x = self.gaze_encoder(images).squeeze()
        if self.varitional:
            mu = self.fc_mean(x)
            log_var = self.fc_log_var(x)
            z = reparameterize(mu, log_var)
        else:
            # if is_left_eye:
            #     x = x + self.left_eye_id
            # else:
            #     x = x + self.right_eye_id
            z = self.head(x)
            #z = x
        return z

class GazeShift(LightningModule):

    def __init__(self, hparams, channels=1):
        super(GazeShift, self).__init__()
        self.automatic_optimization = True
        self.save_hyperparameters()  # sets self.hparams
        self.params = self.hparams['hparams']
        args = self.hparams['hparams']
        #self.cvm_vae_negative_margin = args.cvm_vae_negative_margin
        self.gamma = 1.0
        self.gaze_dim = self.params.gaze_dim
        self.res_loss_weight = 0.0
        # self.fc_mean = torch.nn.Linear(self.params.backbone_feature_size, self.params.backbone_feature_size)
        # self.fc_log_var = torch.nn.Linear(self.params.backbone_feature_size, self.params.backbone_feature_size)
        # self.projection = torch.nn.Linear(self.params.backbone_feature_size, 2)
        self.error_list = []
        d_model = 256
        self.gaze_encoder = GazeEncoder(self.gaze_dim)
        self.eyeid_encoder = EyeIdentityEncoder(128 - self.gaze_dim)

#        self.pos_enc_2d = PositionalEncoding2D(C=d_model, N=10)
        self.att = TransformerDecoder(
            d_model=d_model,
            nhead=1,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.0,
            dgaze=self.gaze_dim
        )
        self.backbone_feaure_size = self.params.backbone_feature_size

        self.channels = 1
        self.decoder = Decoder(latent_dim=d_model)
        if hparams.ckpt_path != '':
            self.load_state_dict(torch.load(hparams.ckpt_path, map_location=self.device)['state_dict'], strict=True)
        elif hparams.vae_path != '':
            self.vae.load_state_dict(torch.load(hparams.vae_path, map_location=self.device)['state_dict'], strict=True)
            self.encoder.load_state_dict(self.vae.encoder.state_dict(), strict=True)
            #copy the weights from vae to contrastive encoder
        else:
            pass

        #Freeze vae
        # for param in self.vae.parameters():
        #     param.requires_grad = False



    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std



    def training_validation_step_ssl(self, x):

        loss = self.cross_encoder_loss_att(x)

        self.log('ce_loss_val', loss, on_step=False, on_epoch=True,
                 sync_dist=True)
        # self.log('kld_val', kld, on_step=False, on_epoch=True,
        #          sync_dist=True)


    def training_validation_step_supervised(self, x):
        gaze_labels, left_images, right_images, label = x

        gaze_left, left_eyeid = self.variational_embedding_att(left_images)
        gaze_right, right_eyeid = self.variational_embedding_att(right_images)


        gaze_lefts = gaze_left.cpu().float().numpy()
        gaze_rights = gaze_right.cpu().float().numpy()
        gaze_labels = gaze_labels.cpu().float().numpy()
        person_ids = label['person_id']

        for person_id, gaze_label,gaze_left, gaze_right in\
                zip(person_ids, gaze_labels, gaze_lefts, gaze_rights):
            new_row = {
                'mu_left': gaze_left,
                'mu_right': gaze_right,
                'person_id': person_id,
                'gt': gaze_label
            }
            self.validation_table_data.append(new_row)

        pass

    def draw_scatter_plot(self, pca_features, gaze_labels):
        # Creating the plot with larger dimensions
        fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figure size if needed
        ax.set_xlim(pca_features[:, 0].min(), pca_features[:, 0].max())
        ax.set_ylim(pca_features[:, 1].min(), pca_features[:, 1].max())

        # Adding images to the scatter plot with a very large zoom factor
        for i in range(len(gaze_labels)):

            #image = img_to_array(image, scale_factor=0.2)
            x, y = pca_features[i, 0], pca_features[i, 1]
            label_text = f"({gaze_labels[i,0]:.2f}, {gaze_labels[i,1]:.2f}, {gaze_labels[i,2]:.2f})"
            ax.text(x, y, label_text, fontsize=9, ha='right', va='bottom')

        ax.set_xlabel('Contrastive embeddings x')
        ax.set_ylabel('Contrastive embedding y')
        plt.title('Contrastive embeddings with ground truth labels')
        plt.show()
        pass

    def apply_pca_and_apply_on_test(self, features_calib, features_test, gaze_labels, side='left'):

        features_calib = np.concatenate(features_calib, axis=0).reshape(len(features_calib), len(features_calib[0]))
        features_test = np.concatenate(features_test, axis=0).reshape(len(features_test), len(features_test[0]))

        #scaler = StandardScaler()
        #features_calib_scaled = scaler.fit_transform(features_calib)
        pca = PCA(n_components=10)

        calib_features_pca = pca.fit_transform(features_calib)
        test_features_pca = pca.transform(features_test)
        mds = MDS(n_components=2, random_state=42)

        # Fit the MDS model and transform the data
        calib_features_pca_mds = mds.fit_transform(features_calib)
        if side == 'left':
            self.draw_scatter_plot(calib_features_pca_mds, gaze_labels)
        else:
            self.draw_scatter_plot(calib_features_pca_mds, gaze_labels)
        # tsne = TSNE(n_components=2, metric='cosine')
        # calib_tsne_features = tsne.fit_transform(features_calib)
        # calib_dmp_features = diffusion_map(features_test_pca, sigma=2, dim=2)
        # self.draw_scatter_plot(calib_tsne_features, gaze_labels, left_images)
        # self.draw_scatter_plot(calib_pca_features, gaze_labels, left_images)


        return calib_features_pca, test_features_pca

    def from_3D_to_yaw_pitch_np(self, gaze_labels):
        x = gaze_labels[:,0]
        y = gaze_labels[:,1]
        z = gaze_labels[:,2]

        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        z = np.expand_dims(z, axis=-1)


        pitch = np.arcsin(y)
        yaw = np.arctan2(x, z)

        concat = np.concatenate([yaw, pitch], axis=1)
        return concat

    def apply_ensemble_calib(self, poly_models, fit_models, por_test_2d ):
        aggregate_por_test = []
        for poly_model, fit_model in zip(poly_models, fit_models):
            por_test_poly = poly_model.fit_transform(por_test_2d)
            t_por_test = fit_model.predict(por_test_poly)
            aggregate_por_test.append(t_por_test)

        aggregate_por_test = np.array(aggregate_por_test)
        avg = np.mean(aggregate_por_test, axis=0)
        return avg

    def from_yaw_pitch_to_3D_np(self, xy_angles):
        pitch = xy_angles[:,1]
        yaw = xy_angles[:,0]

        y = np.sin(pitch)
        z = np.cos(pitch) * np.cos(yaw)
        x = z * np.tan(yaw)

        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        z = np.expand_dims(z, axis=-1)

        result = np.concatenate([x,y,z],axis=1)
        return result

    def calc_metrics_np(self, estimated_gaze, gaze_lables):
        batch_inner_product = np.einsum('ij, ij -> i ', estimated_gaze, gaze_lables)
        cosine_dist = 1 - batch_inner_product
        avg_dist_per_sample = np.mean(cosine_dist)
        eps = 1e-10
        batch_inner_product = np.clip(batch_inner_product, -1+eps, 1-eps )

        error_deg = np.arccos(batch_inner_product) / np.pi * 180.0
        average_error_deg_per_sample = np.mean(error_deg)
        median_error = np.median(error_deg)


        return average_error_deg_per_sample, avg_dist_per_sample, error_deg, median_error


    def extract_ssl_data(self, group_ssl_df, session_cutoff):
        data_for_calib_all = group_ssl_df[group_ssl_df['session_number'] <= session_cutoff]

        left_features_calib_all = data_for_calib_all['mu_left'].values
        left_features_calib_all = np.concatenate(left_features_calib_all, axis=0).reshape(len(left_features_calib_all),
                                                                                  len(left_features_calib_all[0]))
        right_features_calib_all = data_for_calib_all['mu_right'].values
        right_features_calib_all = np.concatenate(right_features_calib_all, axis=0).reshape(len(right_features_calib_all),
                                                                                  len(right_features_calib_all[0]))
        gaze_labels_calib_all = data_for_calib_all['gt'].values
        gaze_labels_calib_all = np.concatenate(data_for_calib_all, axis=0).reshape(len(data_for_calib_all), 3)

    def draw_scatter_plot(self, set_A, set_B, set_C):

        """
        set_A: List of tuples (x, y) representing points in set A
        set_B: List of tuples (x, y) representing points in set B
        set_C: List of tuples (x, y) representing points in set C (used as labels)
        """

        # Check that all sets have the same number of points
        assert len(set_A) == len(set_B) == len(set_C), "All sets must have the same number of points."

        plt.figure(figsize=(8, 8))

        for a, b, c in zip(set_A, set_B, set_C):
            # Plot point from A
            plt.scatter(a[0], a[1], color='blue',
                        label='Left Embeddings' if 'Left Embeddings' not in plt.gca().get_legend_handles_labels()[1] else "")
            # Plot point from B
            plt.scatter(b[0], b[1], color='green',
                        label='Right Embeddings' if 'Right Embeddings' not in plt.gca().get_legend_handles_labels()[1] else "")
            # Draw line between a and b
            plt.plot([a[0], b[0]], [a[1], b[1]], color='black', linestyle='--')
            # Calculate midpoint between a and b
            #midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            # Use the coordinates of point c as the label
            label = f"({c[0]:.2f}, {c[1]:.2f})"
            # Add text label at the midpoint
            plt.text(a[0], a[1], label, fontsize=10, ha='center', va='center', color='red')
            plt.text(b[0], b[1], label, fontsize=10, ha='center', va='center', color='red')

        # Set axis labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('CVM Left and Right Embeddings')

        # Add a legend
        plt.legend()

        # Set equal scaling for x and y axis
        plt.gca().set_aspect('equal', adjustable='box')

        plt.grid(True)
        plt.show()

    def compute_ensemble_fit_merge(self, gaze_labels_calib_fit_2d,
                                        left_features_calib_fit_2d,
                                        right_features_calib_fit_2d,
                                        left_features_test,
                                        right_features_test
                                        ):

        por_test_2d = np.concatenate((left_features_test, right_features_test), axis=1)
        por_calib_2d = np.concatenate((left_features_calib_fit_2d, right_features_calib_fit_2d), axis=1)
        poly_model, fit_models = self.compute_ensemble_fit(gaze_labels_calib_fit_2d, por_calib_2d,
                                                           start_deg=1, end_deg=4)

        #left test


        transformed_test_por = self.apply_ensemble_calib(poly_model, fit_models,
                                                     por_test_2d )

        transformed_test_por_3d = self.from_yaw_pitch_to_3D_np(transformed_test_por)
        return transformed_test_por_3d

    def compute_ensemble_fit_separate(self, gaze_labels_calib_fit_2d,
                                        left_features_calib_fit_2d,
                                        right_features_calib_fit_2d,
                                        left_features_test,
                                        right_features_test
                                        ):
        poly_model_l, fit_models_l = self.compute_ensemble_fit(gaze_labels_calib_fit_2d, left_features_calib_fit_2d, start_deg=1, end_deg=2)
        poly_model_r, fit_models_r = self.compute_ensemble_fit(gaze_labels_calib_fit_2d, right_features_calib_fit_2d, start_deg=1, end_deg=2)

        #left test
        transformed_test_por_l = self.apply_ensemble_calib(poly_model_l, fit_models_l, left_features_test)
        transformed_test_por_r = self.apply_ensemble_calib(poly_model_r, fit_models_r, right_features_test)
        transformed_test_por = (transformed_test_por_l + transformed_test_por_r) / 2.0
        transformed_test_por_3d = self.from_yaw_pitch_to_3D_np(transformed_test_por)
        return transformed_test_por_3d



    def calc_ensemble_calib(self, data_for_calib_for_fit, data_for_test):


        #data_for_calib_for_fit = group_calib_df[group_calib_df['session_number'] <= session_cutoff]

        left_features_calib_fit_2d = data_for_calib_for_fit['mu_left'].values
        left_features_calib_fit_2d = np.concatenate(left_features_calib_fit_2d, axis=0).reshape(len(left_features_calib_fit_2d),
                                                                                  len(left_features_calib_fit_2d[0]))
        right_features_calib_fit_2d = data_for_calib_for_fit['mu_right'].values
        right_features_calib_fit_2d = np.concatenate(right_features_calib_fit_2d, axis=0).reshape(len(right_features_calib_fit_2d),
                                                                                            len(right_features_calib_fit_2d[
                                                                                                    0]))
        gaze_labels_calib_fit_3d = data_for_calib_for_fit['gt'].values
        gaze_labels_calib_fit_3d = np.concatenate(gaze_labels_calib_fit_3d, axis=0).reshape(len(gaze_labels_calib_fit_3d), 3)

        #data_for_test = group_calib_df[group_calib_df['session_number'] > session_cutoff]

        left_features_test_2d = data_for_test['mu_left'].values
        right_features_test_2d = data_for_test['mu_right'].values
        left_features_test = np.concatenate(left_features_test_2d, axis=0).reshape(len(left_features_test_2d),
                                                                                  len(left_features_test_2d[0]))

        right_features_test = np.concatenate(right_features_test_2d, axis=0).reshape(len(right_features_test_2d),
                                                                                  len(right_features_test_2d[0]))

        gaze_labels_test = data_for_test['gt'].values
        gaze_labels_test = np.concatenate(gaze_labels_test, axis=0).reshape(len(gaze_labels_test), 3)

        # left_features_calib_pca, left_feautres_test_pca = self.apply_pca_and_apply_on_test(left_features_calib, left_features_test, decoded_left, decoded_right
        #                                                                                , gaze_labels_fit, side='left')
        # right_features_calib_pca, right_features_test_pca = self.apply_pca_and_apply_on_test(right_features_calib, right_features_test, decoded_left, decoded_right,
        #                                                                                  gaze_labels_fit, side='right')
        gaze_labels_test_2d = self.from_3D_to_yaw_pitch_np(gaze_labels_test)


        gaze_labels_calib_fit_2d = self.from_3D_to_yaw_pitch_np(gaze_labels_calib_fit_3d)

        #self.draw_scatter_plot(left_features_calib_fit_2d, right_features_calib_fit_2d, gaze_labels_calib_fit_2d)
        separate_L_R = False

        if separate_L_R:
            transformed_test_por_3d = self.compute_ensemble_fit_separate(gaze_labels_calib_fit_2d,
                                                                         left_features_calib_fit_2d,
                                                                         right_features_calib_fit_2d,
                                                                         left_features_test,
                                                                         right_features_test)

        else:
            transformed_test_por_3d = self.compute_ensemble_fit_merge(gaze_labels_calib_fit_2d,
                                                                         left_features_calib_fit_2d,
                                                                         right_features_calib_fit_2d,
                                                                         left_features_test,
                                                                         right_features_test)


        average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib = self.calc_metrics_np(transformed_test_por_3d, gaze_labels_test)
        error_deg_after_calib = np.expand_dims(error_deg_after_calib, axis=1)

        return average_error_after_calib, average_error_after_calib, error_deg_after_calib

        pass



        #compute pca for right features
    def compute_fit_ridge(self, estimated_por_calc, gt_calc, poly_deg=2, kernel='linear'):
        poly = PolynomialFeatures(poly_deg)
        model = KernelRidge(alpha=0.001, kernel=kernel)
        #model = KernelRidge(alpha=0.001, kernel='cosine')
        #model = MLPRegressor(hidden_layer_sizes=10, activation='tanh')
        estimated_gaze_poly = poly.fit_transform(estimated_por_calc)
        #estimated_gaze_poly = estimated_por_calc

        #model = Ridge()
        model.fit(estimated_gaze_poly, gt_calc)
        return model, poly

    def compute_ensemble_fit(self, gt_fit_2d, por_fit_2d, start_deg=1,  end_deg=3):
        poly_models = []
        fit_models = []

        #concatenate por_fit_left_2d and por_fit_right_2d along dim=1
        #por_fit = np.concatenate((por_fit_left_2d, por_fit_right_2d), axis=1)
        for poly_deg in range(start_deg, end_deg):
            fit_model, poly_model = self.compute_fit_ridge(por_fit_2d, gt_fit_2d, poly_deg, kernel='linear')
            por_fit_poly = poly_model.fit_transform(por_fit_2d)
            transformed_fit_por = fit_model.predict(por_fit_poly)
            poly_models.append(poly_model)
            fit_models.append(fit_model)

        return poly_models, fit_models

    def per_person_calib(self):
        group_by_id = self.validation_table.groupby('person_id')
        Ks = [17, 30, 40, 50, 60]
        num_iterations = 10
        calibration_data = pd.DataFrame(columns=['K', 'id', 'iter', 'error', 'test_size'])
        for K in Ks:
            for group_id_name, group_id_df in group_by_id:
                #compute PCA for left and right features
                # Sample K rows from the DataFrame
                for k in range(num_iterations):
                    data_for_fit = group_id_df.sample(n=K, random_state=42)
                    data_for_test = group_id_df.drop(data_for_fit.index)
                    average_error_after_calib,\
                    median_error_after,\
                    error_per_label = self.calc_ensemble_calib(data_for_fit, data_for_test)
                    row = {'K': K, 'id': group_id_name, 'iter': k, 'error': average_error_after_calib,
                           'test_size': len(data_for_test)}
                    calibration_data.loc[len(calibration_data)] = row

                # after_calib_avg_err_list_per_id = sum(after_calib_avg_err_list_per_id) / len(after_calib_avg_err_list_per_id)
                # self.log('val_avg_error_id_{}_after_calib'.format(group_id_name), after_calib_avg_err_list_per_id, on_step=False, on_epoch=True,
                #      sync_dist=False)

        for K in Ks:
            calibration_data_per_k = calibration_data[calibration_data['K']==K]
            print( 'K = {}, test_size = {}, acc = {}, std = {}'.format(K, calibration_data_per_k['test_size'].mean(),
                                                                                calibration_data_per_k['error'].mean(),
                                                                                calibration_data_per_k['error'].std()))

        average_error_after_calib = calibration_data['error'].mean()
        std_error =  calibration_data['error'].std()
        test_size = calibration_data['test_size'].mean()
        avg_K = calibration_data['K'].mean()
        self.log('val_error_after_calib_std', std_error, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        self.log('val_error_after_calib', average_error_after_calib, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        self.log('fitting_size', avg_K, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        self.log('avg_testing_size', test_size, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        return

    def person_agnostic_calib(self):
        #self.validation_table.groupby('person_id')
        Ks = [100, 200]
        num_iterations = 10
        calibration_data = pd.DataFrame(columns=['K', 'id', 'iter', 'error', 'test_size'])
        np.random.seed(42)
        for K in Ks:
            for k in range(num_iterations):
                data_for_fit = self.validation_table.sample(n=K)
                data_for_test = self.validation_table.drop(data_for_fit.index)
                average_error_after_calib,\
                median_error_after,\
                error_per_label = self.calc_ensemble_calib(data_for_fit, data_for_test)
                row = {'K': K, 'iter': k, 'error': average_error_after_calib,
                       'test_size': len(data_for_test)}
                calibration_data.loc[len(calibration_data)] = row


        for K in Ks:
            calibration_data_per_k = calibration_data[calibration_data['K']==K]
            print( 'K = {}, test_size = {}, acc = {}, std = {}'.format(K, calibration_data_per_k['test_size'].mean(),
                                                                                calibration_data_per_k['error'].mean(),
                                                                                calibration_data_per_k['error'].std()))

        average_error_after_calib = calibration_data['error'].mean()
        std_error =  calibration_data['error'].std()
        test_size = calibration_data['test_size'].mean()
        avg_K = calibration_data['K'].mean()
        self.log('val_error_after_calib_std', std_error, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        self.log('val_error_after_calib', average_error_after_calib, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        self.log('fitting_size', avg_K, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        self.log('avg_testing_size', test_size, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        return

    def on_validation_epoch_end(self):

        print('on_validation_end called')

        self.validation_table = pd.DataFrame(self.validation_table_data)
        calib_type = 'per_person'
        if calib_type == 'person_agnostic':
            self.person_agnostic_calib()
        else:
            self.per_person_calib()

    def on_validation_epoch_start(self):
        #self.to('cpu')
        #self.validation_table = pd.DataFrame(columns=['subject_id', 'por_estimation', 'gt'])
        self.validation_table_data = []
        self.validation_table_ssl = []

    def validation_step(self, x, batch_idx, dataloader_idx=0):

        #return self.validation(x)
        if dataloader_idx == 0:
            #supervised
            return self.training_validation_step_supervised(x)
        else:
            #ssl
            return self.training_validation_step_ssl(x)

    def variational_embedding_att(self, images):
        encoded_gaze = self.gaze_encoder(images)
        encoded_eyeid = self.eyeid_encoder(images)

        return encoded_gaze, encoded_eyeid


    def variational_embedding(self, images):
        encoded_gaze = self.gaze_encoder(images)
        encoded_eyeid = self.eyeid_encoder(images)

        return encoded_gaze, encoded_eyeid

    def generate_derangement(self, n):
        while True:
            # Generate a permutation of in dices
            perm = torch.randperm(n)
            # Check if any element is in its original position
            if not any(perm == torch.arange(n)):
                return perm



    def loss_kld(self, mu, log_var):
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return KLD

    def loss_same_eye_different_gaze(self, anchor_gaze_negative_eye_id,
                                      negative_gaze_anchor_eye_id,
                                      anchor_images,
                                      negative_images):

        anchor_decode = self.decoder(anchor_gaze_negative_eye_id)
        negative_decode = self.decoder(negative_gaze_anchor_eye_id)
        anchor_mse = torch.nn.functional.mse_loss(anchor_decode, anchor_images, reduction='mean')
        negative_mse = torch.nn.functional.mse_loss(negative_decode, negative_images, reduction='mean')

        res_loss = torch.abs(anchor_images - anchor_decode - (negative_images - negative_decode)).mean()
        loss = (anchor_mse + negative_mse) / 2.0

        loss = (1 - self.res_loss_weight) * loss +  self.res_loss_weight * res_loss
        return loss

    def create_mask(self, tensor):
        # Normalize: Ensure values are in the range [0, 1]
        tensor = tensor.view(tensor.shape[0], 10, 10)
        #tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        tensor = tensor / tensor.max()
        factor = 8.0
        tensor = tensor * factor
        # Rescale to [B, 400, 400] using bilinear interpolation
        tensor = tensor.unsqueeze(1)  # Add a channel dimension for interpolation
        tensor_resized = F.interpolate(tensor, size=(400, 400), mode='bilinear', align_corners=False)
        tensor_resized = tensor_resized.squeeze(1)
        return tensor_resized
    def loss_same_eye_different_gaze_att(self, anchor_gaze_negative_eye_id,
                                         att_maps_anchor_gaze_negative_eye_id,
                                      negative_gaze_anchor_eye_id,
                                      att_maps_negative_gaze_anchor_eye_id,
                                      anchor_images,
                                      negative_images):

        att_maps_anchor_gaze_negative_eye_id = self.create_mask(att_maps_anchor_gaze_negative_eye_id)
        att_maps_negative_gaze_anchor_eye_id = self.create_mask(att_maps_negative_gaze_anchor_eye_id)

        anchor_decode = self.decoder(anchor_gaze_negative_eye_id)
        negative_decode = self.decoder(negative_gaze_anchor_eye_id)
        anchor_mse = torch.nn.functional.mse_loss(anchor_decode, anchor_images, reduction='none')
        anchor_mse = anchor_mse.squeeze() * att_maps_negative_gaze_anchor_eye_id
        anchor_mse = anchor_mse.mean()

        negative_mse = torch.nn.functional.mse_loss(negative_decode, negative_images, reduction='none')
        negative_mse = negative_mse.squeeze() * att_maps_anchor_gaze_negative_eye_id
        negative_mse = negative_mse.mean()

        res_loss = torch.abs(anchor_images - anchor_decode - (negative_images - negative_decode)).mean()
        loss = (anchor_mse + negative_mse) / 2.0

        loss = (1 - self.res_loss_weight) * loss +  self.res_loss_weight * res_loss
        return loss

    def loss_diff_eye_same_gaze(self, positive_gaze_anchor_eye_id,
                                      att_maps_positive_gaze_anchor_eye_id,
                                      anchor_gaze_positive_eye_id,
                                      att_maps_anchor_gaze_positive_eye_id,
                                      anchor_images,
                                      positive_images):

        att_maps_positive_gaze_anchor_eye_id = self.create_mask(att_maps_positive_gaze_anchor_eye_id)
        att_maps_anchor_gaze_positive_eye_id = self.create_mask(att_maps_anchor_gaze_positive_eye_id)

        anchor_decode = self.decoder(positive_gaze_anchor_eye_id)
        positive_decode = self.decoder(anchor_gaze_positive_eye_id)
        anchor_mse = torch.nn.functional.mse_loss(anchor_decode, anchor_images, reduction='none')
        anchor_mse = anchor_mse.squeeze() * att_maps_positive_gaze_anchor_eye_id
        anchor_mse = anchor_mse.mean()

        positive_mse = torch.nn.functional.mse_loss(positive_decode, positive_images, reduction='none')
        positive_mse = positive_mse.squeeze() * att_maps_anchor_gaze_positive_eye_id
        positive_mse = positive_mse.mean()


        loss = (anchor_mse + positive_mse) / 2.0
        res_loss = torch.abs(anchor_images - anchor_decode - (positive_images - positive_decode) ).mean()
        loss = (1 - self.res_loss_weight) * loss + self.res_loss_weight * res_loss
        return loss

    def loss_diff_eye_same_gaze_att(self, positive_gaze_anchor_eye_id,
                                      att_maps_positive_gaze_anchor_eye_id,
                                      anchor_gaze_positive_eye_id,
                                      att_maps_anchor_gaze_positive_eye_id,
                                      anchor_images,
                                      positive_images):

        att_maps_positive_gaze_anchor_eye_id = self.create_mask(att_maps_positive_gaze_anchor_eye_id)
        att_maps_anchor_gaze_positive_eye_id = self.create_mask(att_maps_anchor_gaze_positive_eye_id)

        anchor_decode = self.decoder(positive_gaze_anchor_eye_id)
        positive_decode = self.decoder(anchor_gaze_positive_eye_id)
        anchor_mse = torch.nn.functional.mse_loss(anchor_decode, anchor_images, reduction='none')
        anchor_mse = anchor_mse.squeeze() * att_maps_positive_gaze_anchor_eye_id
        anchor_mse = anchor_mse.mean()

        positive_mse = torch.nn.functional.mse_loss(positive_decode, positive_images, reduction='none')
        positive_mse = positive_mse.squeeze() * att_maps_anchor_gaze_positive_eye_id
        positive_mse = positive_mse.mean()

        loss = (anchor_mse + positive_mse) / 2.0
        res_loss = torch.abs(anchor_images - anchor_decode - (positive_images - positive_decode) ).mean()
        loss = (1 - self.res_loss_weight) * loss + self.res_loss_weight * res_loss
        return loss


    def cross_encoder_loss_att_multi_person_batch(self, x):
        gaze_labels, image_l_1, image_r_1, image_l_2, image_r_2, label = x

        gaze_left_1, eye_id_left_1 = self.variational_embedding_att(image_l_1)
        gaze_right_1, eye_id_right_1 = self.variational_embedding_att(image_r_1)
        gaze_left_2, eye_id_left_2 = self.variational_embedding_att(image_l_2)
        gaze_right_2, eye_id_right_2 = self.variational_embedding_att(image_r_2)

        l1_gaze_l2_eye_id, att_maps_l2_eye_id = self.att(eye_id_left_2, gaze_left_1)
        l2_gaze_l1_eye_id, att_maps_l1_eye_id = self.att(eye_id_left_1, gaze_left_2)
        r1_gaze_r2_eye_id, att_maps_r2_eye_id = self.att(eye_id_right_2, gaze_right_1)
        r2_gaze_r1_eye_id, att_maps_r1_eye_id = self.att(eye_id_right_1, gaze_right_2)

        loss_same_eye_different_gaze_l = self.loss_same_eye_different_gaze_att(l1_gaze_l2_eye_id,
                                                                             att_maps_l2_eye_id,
                                                                             l2_gaze_l1_eye_id,
                                                                             att_maps_l1_eye_id,
                                                                             image_l_1,
                                                                             image_l_2)

        loss_same_eye_different_gaze_r = self.loss_same_eye_different_gaze_att(r1_gaze_r2_eye_id,
                                                                             att_maps_r2_eye_id,
                                                                             r2_gaze_r1_eye_id,
                                                                             att_maps_r1_eye_id,
                                                                             image_r_1,
                                                                             image_r_2)

        loss = (loss_same_eye_different_gaze_r + loss_same_eye_different_gaze_l) / 2.0

        return loss

    def cross_encoder_loss(self, x):
        gaze_labels, left_images, right_images, label = x

        gaze_left, eye_id_left = self.variational_embedding(left_images)
        gaze_right, eye_id_right = self.variational_embedding(right_images)

        batch_size = gaze_labels.shape[0]
        index = self.generate_derangement(batch_size)

        if random.random() < 0.5:
            anchor_gaze = gaze_left
            anchor_eye_id = eye_id_left
            positive_gaze = gaze_right
            positive_eye_id = eye_id_right
            negative_gaze = gaze_left[index]
            negative_eye_id = eye_id_left[index]
            negative_images = left_images[index]
            anchor_images = left_images
            positive_images = right_images


        else:
            anchor_gaze = gaze_right
            anchor_eye_id = eye_id_right
            positive_gaze = gaze_left
            positive_eye_id = eye_id_left
            negative_gaze = gaze_right[index]
            negative_eye_id = eye_id_right[index]
            negative_images = right_images[index]
            anchor_images = right_images
            positive_images = left_images

        positive_gaze_anchor_eye_id = torch.cat((positive_gaze, anchor_eye_id), dim=1)
        anchor_gaze_positive_eye_id = torch.cat((anchor_gaze, positive_eye_id), dim=1)
        anchor_gaze_negative_eye_id = torch.cat((anchor_gaze, negative_eye_id), dim=1)
        negative_gaze_anchor_eye_id = torch.cat((negative_gaze, anchor_eye_id), dim=1)

        loss_gaze = torch.abs(positive_gaze - anchor_gaze).mean()
        loss_id = torch.abs(anchor_eye_id - negative_eye_id).mean()

        loss_same_eye_different_gaze = self.loss_same_eye_different_gaze(anchor_gaze_negative_eye_id,
                                                                    negative_gaze_anchor_eye_id,
                                                                    anchor_images,
                                                                    negative_images)

        loss_diff_eye_same_gaze = self.loss_diff_eye_same_gaze(positive_gaze_anchor_eye_id,
                                                             anchor_gaze_positive_eye_id,
                                                             anchor_images,
                                                             positive_images)

        alpha = 1.0
        ce_loss = (loss_same_eye_different_gaze + loss_diff_eye_same_gaze) / 2.0

        loss_same_property = (loss_gaze + loss_id) / 2.0
        loss = alpha * ce_loss + (1-alpha)*loss_same_property


        #kld_left = self.loss_kld(mu_left, log_var_left)
        #kld_right = self.loss_kld(mu_right, log_var_right)

        #kld = (kld_left + kld_right) / 2.0

        #loss = loss * self.gamma + kld * (1 - self.gamma)


        return loss

    def compute_triplet_loss_label_oracle(self, anchor, positive, negatives, anchor_gt, negative_gts):

        batch_size = anchor.shape[0]
        d_ap = torch.norm(anchor - positive, dim=1)
        sum_negatives = torch.zeros_like(d_ap)
        sum_labels_negatives = torch.zeros_like(d_ap)
        for negative in negatives:
            d_an = torch.norm(anchor - negative, dim=1)
            sum_negatives = sum_negatives + d_an

        #d_an = torch.norm(anchor - negative, dim=1)
        for negative_gt in negative_gts:
            d_label = torch.norm(anchor_gt - negative_gt, dim=1)
            sum_labels_negatives = sum_labels_negatives + d_label


        loss = torch.abs(sum_negatives - sum_labels_negatives)
        loss = torch.mean(loss)
        return loss


        pass

    def oracle_fixed_loss(self, x):
        gaze_labels, left_images, right_images, label = x

        z_proj_left, z_left, mu_left, log_var_left,_ = self.variational_embedding(left_images, vae=False)
        z_proj_right, z_right, mu_right, log_var_right,_ = self.variational_embedding(right_images, vae=False)

        batch_size = gaze_labels.shape[0]
        index = self.generate_derangement(batch_size)

        if random.random() < 0.5:
            left_perm = z_proj_left[index]
            anchor = z_proj_left
            positive = z_proj_right
            negative = left_perm
        else:
            right_perm = z_proj_right[index]
            anchor = z_proj_right
            positive = z_proj_left
            negative = right_perm


        # triplet_loss = (self.compute_triplet_loss(anchor, positive, negative) +
        #                self.compute_triplet_loss(positive, anchor, negative)) / 2.0
        triplet_loss = self.compute_triplet_loss(anchor, positive, negative)

        kld_left = self.loss_kld(mu_left, log_var_left)
        kld_right = self.loss_kld(mu_right, log_var_right)

        kld = (kld_left + kld_right) / 2.0

        loss = triplet_loss * self.gamma + kld * (1 - self.gamma)

        return loss, kld

    def from_3D_to_yaw_pitch(self, vec_3d):
        x = vec_3d[:,0]
        y = vec_3d[:, 1]
        z = vec_3d[:, 2]
        pitch = torch.asin(y).unsqueeze(dim=1)
        yaw = torch.atan2(x, z).unsqueeze(dim=1)

        final = torch.cat((pitch,yaw),dim=1)
        return final


    def training_step(self, x):

        loss = self.cross_encoder_loss_att_multi_person_batch(x)

        self.log('loss_train', loss, on_step=True, on_epoch=True,
                 sync_dist=True)
        return loss

    def configure_optimizers(self):
        #opt = torch.optim.SGD(self.parameters(),lr=0.0)
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams['hparams'].lr,
                                weight_decay=self.hparams['hparams'].weight_decay)

        scheduler = lr_scheduler.MultiStepLR(opt,
                                             milestones=self.hparams['hparams'].lr_milestones,
                                             gamma=self.hparams['hparams'].lr_gamma)

        return [opt], [scheduler]


def np_to_pil_image(arr):
    arr = arr - arr.min()
    arr = arr / arr.max()
    return Image.fromarray(arr)

def tensor_to_pil_image(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return torchvision.transforms.ToPILImage()(tensor)

def loss_function_vae(recon_x, x, mu, log_var):
    MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE, KLD

def loss_function_ae(recon_x, x):
    MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
    return MSE


def deactivate_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


class Head(LightningModule):
    def __init__(self, hparams):
        super(Head, self).__init__()
        self.save_hyperparameters()  # sets self.hparams
        args = self.hparams['hparams']
        # width_mult_to_depth = {
        #     1.0: 32,
        #     0.75: 24,
        #     2.0: 128,
        # }
        width_mult_to_depth = {
            1.0: 64,
            0.75: 24,
            2.0: 128,
            3.0: 192,
        }

        #self.input_feature_depth = 96
        #self.mono_feature_depth = 32
        #self.mono_feature_depth = 24
        #self.mono_feature_depth = 48
        self.mono_feature_depth = width_mult_to_depth[args.width_multiplier]
        #self.input_feature_depth = 32
        self.output_feature_depth = 3
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.1)
        #self.conv1_att = torch.nn.Conv2d(self.input_feature_depth, 1, 1)
        self.conv1_bino = torch.nn.Conv2d(self.mono_feature_depth * 2, self.mono_feature_depth ,1,1)
        self.conv2 = torch.nn.Conv2d(self.mono_feature_depth, 3, 1, 1)

        # for param in self.conv1_bino.parameters():
        #     param.requires_grad = False

        #self.conv3 = torch.nn.Conv2d(self.output_feature_depth // 2, 3, 1, 1)

    def forward_mono(self, backbone_features):

        #concat_features = torch.cat(1(left_backbone_features, right_backbone_features), dim=1)
        #features = self.conv1_bino(backbone_features)
        features = self.conv2(backbone_features).squeeze(-1).squeeze(-1)

        #features = torch.tanh(features)

        por_estimation = torch.tensor([0,0,1.0], device=features.device) + features
        por_estimation = torch.nn.functional.normalize(por_estimation)

        return por_estimation, features

    def forward_bino(self, left_backbone_features, right_backbone_features):

        concat_features = torch.cat((left_backbone_features, right_backbone_features), dim=1)
        features = self.conv1_bino(concat_features)
        features = self.conv2(features).squeeze(-1).squeeze(-1)

        por_estimation = torch.tensor([0,0,1.0], device=features.device) + features
        por_estimation = torch.nn.functional.normalize(por_estimation)

        return por_estimation, concat_features



    def forward(self, left_backbone_features, right_backbone_features=None):
        if right_backbone_features is None:
            return self.forward_mono(left_backbone_features)
        else:
            return self.forward_bino(left_backbone_features, right_backbone_features)

#############################################################################################

class Decoder(LightningModule):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        # Initial projection to 7x7x512
        #self.latent_dim = latent_dim
        self.initial_channel_dim = latent_dim
        #self.initial_channel_dim = 512
        #self.fc = torch.nn.Linear(self.latent_dim, 5 * 5 * self.initial_channel_dim)

        # Upsampling blocks to double the spatial dimensions
        self.block1 = self._upsample_block(self.initial_channel_dim, self.initial_channel_dim // 2)  # Output: 10x10
        self.block2 = self._upsample_block(self.initial_channel_dim // 2, self.initial_channel_dim // 4)  # Output: 20x20
        self.block3 = self._upsample_block(self.initial_channel_dim // 4, self.initial_channel_dim // 8)  # Output: 40x40
        self.block4 = self._upsample_block(self.initial_channel_dim // 8, self.initial_channel_dim // 16)  # Output: 80x80
        self.block5 = self._upsample_block(self.initial_channel_dim // 16,
                                           self.initial_channel_dim // 32)  # Output: 160x160
        self.block6 = self._upsample_block(self.initial_channel_dim // 32,
                                           self.initial_channel_dim // 64)  # Output: 320x320

        self.block7 = torch.nn.Sequential(torch.nn.Upsample(mode='nearest', size=(400,400)),  # Double the spatial dimensions
                torch.nn.Conv2d(self.initial_channel_dim // 64, 1, kernel_size=3, padding=1),
                torch.nn.Tanh()
            )



        # Final convolution to adjust to 3 channels for RGB
        #self.final_conv = nn.Conv2d(self.initial_channel_dim // 16, 3, kernel_size=3, padding=1)

    def _upsample_block(self, in_channels, out_channels,size=None):

        scale_factor = 2 if size==None else None
        if scale_factor == 2:
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                #torch.nn.Upsample(scale_factor=scale_factor, mode='nearest', size=size),  # Double the spatial dimensions
                #torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.Tanh()
            )
        else:
            block = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=scale_factor, mode='nearest', size=size),  # Double the spatial dimensions
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.Tanh()
            )

        return block

    def forward(self, x):
        # x = x.squeeze()

        # x = self.fc(x)
        #x = x.view(-1, self.initial_channel_dim, 5, 5)  # Reshape to a spatial dimension
        #x = x.view(-1, self.initial_channel_dim, 10, 10)  # Reshape to a spatial dimensio
        x = x.permute(0, 2, 1).view(-1, self.initial_channel_dim, 10, 10)
        #
        # # Sequentially pass through upsampling blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        #x = torch.tanh(x)  # Use sigmoid to ensure output pixel values are in [0, 1]
        return x


# class Decoder(LightningModule):
#     def __init__(self, hparams, latent_dim: int):
#         """Decoder.
#
#                Args:
#                   num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
#                   base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
#                   latent_dim : Dimensionality of latent representation z
#                   act_fn : Activation function used throughout the decoder network
#                """
#         super(Decoder, self).__init__()
#         self.save_hyperparameters()  # sets self.hparams
#         args = self.hparams['hparams']
#         width_mult_to_depth = {
#             1.0: 64,
#             0.75: 24,
#             2.0: 128,
#             3.0: 192,
#         }
#         k = 128
#         self.model = torch.nn.Sequential(
#             # First layer: latent_dim -> 1024, with upscaling
#             torch.nn.Linear(latent_dim, k * 5 * 5),
#             torch.nn.ReLU(True),
#             # Reshape to a 4D tensor for convolutional layers
#             torch.nn.Unflatten(1, (k, 5, 5)),
#             # Upsample to 4x4
#             torch.nn.ConvTranspose2d(k, k // 2, kernel_size=4, stride=2, padding=1),
#             torch.nn.ReLU(True),
#             # Upsample to 8x8
#             torch.nn.ConvTranspose2d(k // 2, k // 4, kernel_size=4, stride=2, padding=1),
#             torch.nn.ReLU(True),
#             # Upsample to 16x16
#             torch.nn.ConvTranspose2d(k // 4, k // 16, kernel_size=4, stride=2, padding=1),
#             torch.nn.ReLU(True),
#             # Upsample to 32x32
#             torch.nn.ConvTranspose2d(k // 16, k // 32, kernel_size=4, stride=2, padding=1),
#             torch.nn.ReLU(True),
#             # Upsample to 64x64
#             torch.nn.ConvTranspose2d(k // 32, k // 64, kernel_size=4, stride=2, padding=1),
#             torch.nn.ReLU(True),
#             #torch.nn.ReLU(True),
#             # Upsample to 128x128
#             torch.nn.ConvTranspose2d(k // 64, k // 64, kernel_size=4, stride=2, padding=1),
#             torch.nn.ReLU(True),
#
#             # Upsample to 256x256
#             torch.nn.ConvTranspose2d(k // 64, 1, kernel_size=4, stride=2, padding=1),
#             torch.nn.Tanh()  # Normalize the output to [-1, 1]
#         )
#
#     def forward(self, z):
#         z = z.squeeze()
#         x = self.model(z)
#         return x



def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        #torch.nn.init.normal_(m.weight, mean=0.0, std=5.0)
        #torch.nn.init.uniform(m.weight,0,10)
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

#############################################################################################





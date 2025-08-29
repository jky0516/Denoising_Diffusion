import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class SimpleRegressor(nn.Module):
    def __init__(self, out_dim=4, in_channels=1, hidden_dims=None, time_emb_dim=32):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=hidden_dims[0],
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU()
        )
        
        self.time_proj = nn.Linear(time_emb_dim, hidden_dims[0])

        in_channels = hidden_dims[0]
        self.remaining_convs = nn.ModuleList()
        for h_dim in hidden_dims[1:]:
            self.remaining_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x, t): 
        t_emb = self.time_mlp(t)
        
        h = self.conv1(x)
        
        # Time embedding
        time_cond = self.time_proj(t_emb)[:, :, None, None] 
        h = h + time_cond 
        
        for conv_block in self.remaining_convs:
            h = conv_block(h)
            
        h = self.flatten(h)

        return self.fc_layers(h)
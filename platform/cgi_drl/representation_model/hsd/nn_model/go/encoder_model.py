import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        x_input = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x_input + x)
        return x

class EncoderModel(nn.Module):
    def __init__(self, network_settings):
        super(EncoderModel, self).__init__()
        self.hidden_size = network_settings["hidden_dimension"]
        
        # VQ Embeddings
        # H0
        self.h0_vq_embedding_size = network_settings["vq_embedding_sizes"][0]
        self.h0_vq_embedding = nn.Embedding(self.h0_vq_embedding_size[0], self.h0_vq_embedding_size[1])
        nn.init.trunc_normal_(self.h0_vq_embedding.weight, mean=0, std=0.02)
        # H1
        self.h1_vq_embedding_size = network_settings["vq_embedding_sizes"][1]
        self.h1_vq_embedding = nn.Embedding(self.h1_vq_embedding_size[0], self.h1_vq_embedding_size[1])
        nn.init.trunc_normal_(self.h1_vq_embedding.weight, mean=0, std=0.02)
        # H2
        self.h2_vq_embedding_size = network_settings["vq_embedding_sizes"][2]
        self.h2_vq_embedding = nn.Embedding(self.h2_vq_embedding_size[0], self.h2_vq_embedding_size[1])
        nn.init.trunc_normal_(self.h2_vq_embedding.weight, mean=0, std=0.02)

        # H0 encoder
        self.h0_conv = nn.Conv2d(18, self.hidden_size, kernel_size=3, padding=1)
        self.h0_bn = nn.BatchNorm2d(self.hidden_size)

        num_blocks = 3
        self.h0_residual_blocks = nn.ModuleList([ResidualBlock(self.hidden_size) for _ in range(num_blocks)])

        self.h0_conv2 = nn.Conv2d(self.hidden_size, 32, kernel_size=3, padding=1)
        self.h0_bn2 = nn.BatchNorm2d(32)

        # H1 encoder
        self.h1_fc = nn.Linear(self.h0_vq_embedding_size[1]*19*19, 256)
        self.h1_fc2 = nn.Linear(256, self.h1_vq_embedding_size[1]*8)


        # H2 encoder
        self.h2_fc = nn.Linear(self.h1_vq_embedding_size[1]*8, 128)
        self.h2_fc2 = nn.Linear(128, self.h2_vq_embedding_size[1]*8)
    
        # H2 decoder
        self.h2_fc3 = nn.Linear(self.h2_vq_embedding_size[1]*8, 128)
        self.h2_fc4 = nn.Linear(128, self.h1_vq_embedding_size[1]*8)

        # H1 decoder
        self.h1_fc3 = nn.Linear(self.h1_vq_embedding_size[1]*8, 256)
        self.h1_fc4 = nn.Linear(256, self.h0_vq_embedding_size[1]*19*19)

        # H0 decoder
        self.policy_fc = nn.Linear(32*19*19, 362)
        self.value_fc = nn.Linear(32*19*19, 1)

    def forward(self, observations, hierarchy_usages):
        z_e = []
        z_q = []
        z_q_latent = []
        embedding_ks = []
        embedding_mean = []

        def vq(latent_point, embedding_points, embedding_size):
            input_shape = latent_point.shape
            latent_point = latent_point.view(-1, embedding_size)
            # (a-b)^2 = a^2 + b^2 - 2ab
            vq_distance = (torch.sum(latent_point**2, dim=-1, keepdim=True) + torch.sum(embedding_points**2, dim=-1) - 2 * torch.matmul(latent_point, embedding_points.t()))
            k = torch.argmin(vq_distance, dim=-1)
            z_decoder = torch.index_select(embedding_points, 0, k)
            z_gradient_copy = latent_point + (z_decoder - latent_point).detach()
            return k.view(input_shape[0], -1), z_decoder.view(input_shape), z_gradient_copy.view(input_shape)

        # base
        x = observations
        x = self.h0_conv(x)
        x = self.h0_bn(x)
        x = F.relu(x)

        for residual_block in self.h0_residual_blocks:
            x = residual_block(x)
        x = self.h0_conv2(x)
        x = self.h0_bn2(x)
        x = torch.tanh(x)

        z_e.append(x)
        embedding_k, embedding_decoder_latent, decoder_latent = vq(x, self.h0_vq_embedding.weight, self.h0_vq_embedding_size[1])
        z_q.append(embedding_decoder_latent)
        z_q_latent.append(decoder_latent)
        embedding_ks.append(embedding_k)

        h0_embedding_mean = torch.mean(self.h0_vq_embedding.weight, dim=0)
        h0_embedding_mean = h0_embedding_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(x.shape[0], -1, 19, 19)
        embedding_mean.append(h0_embedding_mean)

        # H1
        x = x.detach()
        x = x.view(-1, self.h0_vq_embedding_size[1]*19*19)

        x = self.h1_fc(x)
        x = F.relu(x)
        x = self.h1_fc2(x)
        x = torch.tanh(x)
        x = x.view(-1, self.h1_vq_embedding_size[1], 8)

        z_e.append(x)
        embedding_k, embedding_decoder_latent, decoder_latent = vq(x, self.h1_vq_embedding.weight, self.h1_vq_embedding_size[1])
        z_q.append(embedding_decoder_latent)
        z_q_latent.append(decoder_latent)
        embedding_ks.append(embedding_k)

        h1_embedding_mean = torch.mean(self.h1_vq_embedding.weight, dim=0)
        h1_embedding_mean = h1_embedding_mean.unsqueeze(0).unsqueeze(-1).expand(x.shape[0], -1, 8)
        embedding_mean.append(h1_embedding_mean)

        # H2
        x = x.detach()
        x = x.view(-1, self.h1_vq_embedding_size[1]*8)

        x = self.h2_fc(x)
        x = F.relu(x)
        x = self.h2_fc2(x)
        x = torch.tanh(x)
        x = x.view(-1, self.h2_vq_embedding_size[1], 8)

        z_e.append(x)
        embedding_k, embedding_decoder_latent, decoder_latent = vq(x, self.h2_vq_embedding.weight, self.h2_vq_embedding_size[1])
        z_q.append(embedding_decoder_latent)
        z_q_latent.append(decoder_latent)
        embedding_ks.append(embedding_k)

        h2_embedding_mean = torch.mean(self.h2_vq_embedding.weight, dim=0)
        h2_embedding_mean = h2_embedding_mean.unsqueeze(0).unsqueeze(-1).expand(x.shape[0], -1, 8)
        embedding_mean.append(h2_embedding_mean)

        # decode
        x = z_q_latent[2]

        # H2
        x = x.view(-1, self.h2_vq_embedding_size[1]*8)
        x = self.h2_fc3(x)
        x = F.relu(x)
        x = self.h2_fc4(x)
        x = torch.tanh(x)
        x = x.view(-1, self.h1_vq_embedding_size[1], 8)

        h1_high_level = x
        x = hierarchy_usages[:,1].view(-1, 1, 1) * z_q_latent[1] + (1 - hierarchy_usages[:,1].view(-1, 1, 1)) * h1_high_level

        # H1
        x = x.view(-1, self.h1_vq_embedding_size[1]*8)
        x = self.h1_fc3(x)
        x = F.relu(x)
        x = self.h1_fc4(x)
        x = torch.tanh(x)
        x = x.view(-1, self.h0_vq_embedding_size[1], 19, 19)

        h0_high_level = x
        x = hierarchy_usages[:,0].view(-1, 1, 1, 1) * z_q_latent[0] + (1 - hierarchy_usages[:,0].view(-1, 1, 1, 1)) * h0_high_level

        x = x.view(-1, 32*19*19)

        policy = self.policy_fc(x)

        value = self.value_fc(x)
        value = torch.tanh(value)

        return {
            "policy": policy, 
            "value": value, 
            "z_e": z_e, 
            "z_q": z_q, 
            "k": embedding_ks, 
            "z_q_latent": z_q_latent,
            "embedding_mean": embedding_mean,
        }

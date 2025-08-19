import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EncoderModel(nn.Module):
    def __init__(self, network_settings):
        super(EncoderModel, self).__init__()

        # Base Encoder
        self.base_encoder_hidden_layers = nn.ModuleList([
            nn.Conv2d(12 + 4, 32, kernel_size=3, padding=0, stride=1), # [32, 6, 6]
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=1) # [64, 4, 4]
        ])
        self.base_encoder_output_layer = nn.Conv2d(64, 128, kernel_size=3, padding=0, stride=1) # [128, 2, 2]


        # Base Decoder
        self.base_decoder_hidden_layers = nn.ModuleList([
            nn.ConvTranspose2d(128,  64, kernel_size=3, stride=1),
            nn.ConvTranspose2d(64,  32, kernel_size=3, stride=1)
        ])
        self.base_decoder_output_layer = nn.ConvTranspose2d(32,  12 + 4, kernel_size=3, stride=1)

        # Policy
        self.policy_hidden_layer = nn.Linear(128*2*2, 128)
        self.policy_output_layer = nn.Linear(128, 6)
        
        # Hierarchical Encoder
        self.hierarchical_encoder_hidden_layer = nn.Linear(128*2*2, 128)  
        self.hierarchical_encoder_output_layer = nn.Linear(128, 16 * 8)
        
        # Hierarchical Decoder
        self.hierarchical_decoder_hidden_layer = nn.Linear(16 * 8, 128)
        self.hierarchical_decoder_output_layer = nn.Linear(128, 128*2*2)
    
        # VQ Embeddings
        self.vq_embedding_sizes = network_settings["vq_embedding_sizes"]
        self.vq_embeddings = nn.ModuleList([ 
            nn.Embedding(vq_embedding_size[0], vq_embedding_size[1])
            for vq_embedding_size in self.vq_embedding_sizes
        ])
        for embedding in self.vq_embeddings:
            nn.init.trunc_normal_(embedding.weight, mean=0, std=0.02)

    def forward(self, observations, hierarchy_usages):
        x_visual = observations["observation_visual"]
        x_vector = observations["observation_vector"]
        x_vector = x_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)
        x = torch.cat([x_visual, x_vector], dim=1)

        for layer in self.base_encoder_hidden_layers:
            x = F.relu(layer(x))
        x = torch.tanh(self.base_encoder_output_layer(x))

        z_e0 = x

        def vq_2d(latent_point, embedding_points):
            # (a-b)^2 = a^2 + b^2 - 2ab
            flatten_latent_point = latent_point.permute(0, 2, 3, 1).contiguous() # [N,H,W,C]
            input_shape = flatten_latent_point.shape
            flatten_latent_point = flatten_latent_point.view(-1, embedding_points.shape[-1]) # [N*H*W,C]
            vq_distance = (torch.sum(flatten_latent_point**2, dim=-1, keepdim=True) + torch.sum(embedding_points**2, dim=-1) - 2 * torch.matmul(flatten_latent_point, embedding_points.t()))
            k = torch.argmin(vq_distance, dim=-1)
            z_decoder = torch.index_select(embedding_points, 0, k).view(input_shape).permute(0, 3, 1, 2).contiguous() # [N,C,H,W]
            z_gradient_copy = latent_point + (z_decoder - latent_point).detach()
            return k.view(input_shape[:-1]), z_decoder, z_gradient_copy

        def vq_1d(latent_point, embedding_points):
            # (a-b)^2 = a^2 + b^2 - 2ab
            flatten_latent_point = latent_point.permute(0, 2, 1).contiguous() # [N,W,C]
            input_shape = flatten_latent_point.shape
            flatten_latent_point = flatten_latent_point.view(-1, embedding_points.shape[-1]) # [N*W,C]
            vq_distance = (torch.sum(flatten_latent_point**2, dim=-1, keepdim=True) + torch.sum(embedding_points**2, dim=-1) - 2 * torch.matmul(flatten_latent_point, embedding_points.t()))
            k = torch.argmin(vq_distance, dim=-1)
            z_decoder = torch.index_select(embedding_points, 0, k).view(input_shape).permute(0, 2, 1).contiguous() # [N,C,W]
            z_gradient_copy = latent_point + (z_decoder - latent_point).detach()
            return k.view(input_shape[:-1]), z_decoder, z_gradient_copy

        k0, embedding_0, z_q0_low_level = vq_2d(z_e0, self.vq_embeddings[0].weight)

        # hierarchical
        x = z_e0.detach().view(-1, 128 * 2 * 2)
        x = F.relu(self.hierarchical_encoder_hidden_layer(x))
        x = torch.tanh(self.hierarchical_encoder_output_layer(x))
        fc_z_e1 = x.view(-1, 16, 8)
        k1, fc_embedding_1, fc_z_q1 = vq_1d(fc_z_e1, self.vq_embeddings[1].weight)

        x = fc_z_q1.view(-1, 16 * 8)
        x = F.relu(self.hierarchical_decoder_hidden_layer(x))
        x = self.hierarchical_decoder_output_layer(x)
        z_q0_high_level = x.view(-1, 128, 2, 2)

        # cnn decoder
        combined_x = hierarchy_usages[:,0].view(-1, 1, 1, 1) * z_q0_low_level + (1 - hierarchy_usages[:,0].view(-1, 1, 1, 1)) * z_q0_high_level
        
        x = combined_x
        for layer in self.base_decoder_hidden_layers:
            x = F.relu(layer(x))
        x = torch.sigmoid(self.base_decoder_output_layer(x))

        compressed_observation = {
            "observation_visual" : x[:, :12, :, :],
            "observation_vector" : F.adaptive_avg_pool2d(x[:, 12:, :, :], (1, 1)).squeeze(-1).squeeze(-1)
        }

        # policy decoder
        x = combined_x.view(-1, 128 * 2 * 2)
        x = F.relu(self.policy_hidden_layer(x))
        x = self.policy_output_layer(x)
        policy = x

        return [(z_e0, embedding_0), (fc_z_e1, fc_embedding_1)], [k0, k1], compressed_observation, policy

    def state_forward(self, observations):
        x_visual = observations["observation_visual"]
        x_vector = observations["observation_vector"]
        x_vector = x_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)
        x = torch.cat([x_visual, x_vector], dim=1)

        for layer in self.base_encoder_hidden_layers:
            x = F.relu(layer(x))
        x = torch.tanh(self.base_encoder_output_layer(x))

        z_e0 = x

        def vq_2d(latent_point, embedding_points):
            # (a-b)^2 = a^2 + b^2 - 2ab
            flatten_latent_point = latent_point.permute(0, 2, 3, 1).contiguous() # [N,H,W,C]
            input_shape = flatten_latent_point.shape
            flatten_latent_point = flatten_latent_point.view(-1, embedding_points.shape[-1]) # [N*H*W,C]
            vq_distance = (torch.sum(flatten_latent_point**2, dim=-1, keepdim=True) + torch.sum(embedding_points**2, dim=-1) - 2 * torch.matmul(flatten_latent_point, embedding_points.t()))
            k = torch.argmin(vq_distance, dim=-1)
            z_decoder = torch.index_select(embedding_points, 0, k).view(input_shape).permute(0, 3, 1, 2).contiguous() # [N,C,H,W]
            z_gradient_copy = latent_point + (z_decoder - latent_point).detach()
            return k.view(input_shape[:-1]), z_decoder, z_gradient_copy

        def vq_1d(latent_point, embedding_points):
            # (a-b)^2 = a^2 + b^2 - 2ab
            flatten_latent_point = latent_point.permute(0, 2, 1).contiguous() # [N,W,C]
            input_shape = flatten_latent_point.shape
            flatten_latent_point = flatten_latent_point.view(-1, embedding_points.shape[-1]) # [N*W,C]
            vq_distance = (torch.sum(flatten_latent_point**2, dim=-1, keepdim=True) + torch.sum(embedding_points**2, dim=-1) - 2 * torch.matmul(flatten_latent_point, embedding_points.t()))
            k = torch.argmin(vq_distance, dim=-1)
            z_decoder = torch.index_select(embedding_points, 0, k).view(input_shape).permute(0, 2, 1).contiguous() # [N,C,W]
            z_gradient_copy = latent_point + (z_decoder - latent_point).detach()
            return k.view(input_shape[:-1]), z_decoder, z_gradient_copy

        k0, embedding_0, z_q0_low_level = vq_2d(z_e0, self.vq_embeddings[0].weight)

        # hierarchical
        x = z_e0.detach().view(-1, 128 * 2 * 2)
        x = F.relu(self.hierarchical_encoder_hidden_layer(x))
        x = torch.tanh(self.hierarchical_encoder_output_layer(x))
        fc_z_e1 = x.view(-1, 16, 8)
        k1, fc_embedding_1, fc_z_q1 = vq_1d(fc_z_e1, self.vq_embeddings[1].weight)

        return [k0, k1]


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EncoderModel(nn.Module):
    def __init__(self, network_settings):
        super(EncoderModel, self).__init__()
        self.input_dimensions = network_settings["input_dimensions"]
        self.hidden_dimensions = network_settings["hidden_dimensions"]
        self.cnn_output_dimensions = network_settings["cnn_cnn_output_dimensions"]
        self.policy_output_dimensions = network_settings["policy_output_dimensions"]

        # Visual Encoder
        self.cnn_encoder_hidden_layers = nn.ModuleList([
            nn.Conv2d(in_channels=self.input_dimensions[0][0], out_channels=self.input_dimensions[1][0], kernel_size=8, stride=4),
            nn.Conv2d(in_channels=self.input_dimensions[1][0], out_channels=self.input_dimensions[2][0], kernel_size=4, stride=2),
        ])
        # tensorflow weight initialization
        for layer in self.cnn_encoder_hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.cnn_encoder_output_layers = nn.Conv2d(in_channels=self.input_dimensions[2][0], out_channels=self.input_dimensions[3][0], kernel_size=3, stride=1)
        # tensorflow weight initialization
        nn.init.xavier_uniform_(self.cnn_encoder_output_layers.weight)
        nn.init.zeros_(self.cnn_encoder_output_layers.bias)

        # Visual Decoder
        self.cnn_decoder_hidden_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=self.cnn_output_dimensions[0][0],  out_channels=self.cnn_output_dimensions[1][0], kernel_size=3, stride=1),
            nn.ConvTranspose2d(in_channels=self.cnn_output_dimensions[1][0],  out_channels=self.cnn_output_dimensions[2][0], kernel_size=4, stride=2),
        ])
        # tensorflow weight initialization
        for layer in self.cnn_decoder_hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.cnn_decoder_output_layers = nn.ConvTranspose2d(in_channels=self.cnn_output_dimensions[2][0],  out_channels=self.cnn_output_dimensions[3][0], kernel_size=8, stride=4)
        # tensorflow weight initialization
        nn.init.xavier_uniform_(self.cnn_decoder_output_layers.weight)
        nn.init.zeros_(self.cnn_decoder_output_layers.bias)

        # Policy
        self.policy_hidden_layer = nn.Linear(np.prod(self.policy_output_dimensions[0]), np.prod(self.policy_output_dimensions[1]))
        # tensorflow weight initialization
        nn.init.xavier_uniform_(self.policy_hidden_layer.weight)
        nn.init.zeros_(self.policy_hidden_layer.bias)
        self.policy_output_layer = nn.Linear(np.prod(self.policy_output_dimensions[1]), np.prod(self.policy_output_dimensions[2]))
        # tensorflow weight initialization
        nn.init.xavier_uniform_(self.policy_output_layer.weight)
        nn.init.zeros_(self.policy_output_layer.bias)

        # Hierarchical Encoder
        self.hierarchical_encoder_hidden_layer = nn.Linear(np.prod(self.hidden_dimensions[0]), np.prod(self.hidden_dimensions[1]))
        # tensorflow weight initialization
        nn.init.xavier_uniform_(self.hierarchical_encoder_hidden_layer.weight)
        nn.init.zeros_(self.hierarchical_encoder_hidden_layer.bias)
        self.hierarchical_encoder_output_layer = nn.Linear(np.prod(self.hidden_dimensions[1]), np.prod(self.hidden_dimensions[2]))
        # tensorflow weight initialization
        nn.init.xavier_uniform_(self.hierarchical_encoder_output_layer.weight)
        nn.init.zeros_(self.hierarchical_encoder_output_layer.bias)
        
        # Hierarchical Decoder
        self.hierarchical_decoder_hidden_layer = nn.Linear(np.prod(self.hidden_dimensions[2]), np.prod(self.hidden_dimensions[1]))
        # tensorflow weight initialization
        nn.init.xavier_uniform_(self.hierarchical_decoder_hidden_layer.weight)
        nn.init.zeros_(self.hierarchical_decoder_hidden_layer.bias)
        self.hierarchical_decoder_output_layer = nn.Linear(np.prod(self.hidden_dimensions[1]), np.prod(self.hidden_dimensions[0]))
        # tensorflow weight initialization
        nn.init.xavier_uniform_(self.hierarchical_decoder_output_layer.weight)
        nn.init.zeros_(self.hierarchical_decoder_output_layer.bias)
    
        # VQ Embeddings
        self.vq_embedding_sizes = network_settings["vq_embedding_sizes"]
        self.vq_embeddings = nn.ModuleList([ 
            nn.Embedding(vq_embedding_size[0], vq_embedding_size[1])
            for vq_embedding_size in self.vq_embedding_sizes
        ])
        for embedding in self.vq_embeddings:
            nn.init.trunc_normal_(embedding.weight, mean=0, std=0.02)

    def forward(self, observations, hierarchy_usages):
        x = observations["observation_2d"]

        for layer in self.cnn_encoder_hidden_layers:
            x = F.leaky_relu(layer(x), negative_slope=0.01)
        x = torch.tanh(self.cnn_encoder_output_layers(x))

        cnn_z_e0 = x

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

        k0, cnn_embedding_0, cnn_z_q0_low_level = vq_2d(cnn_z_e0, self.vq_embeddings[0].weight)

        # hierarchical
        x = cnn_z_e0.detach().view(-1, np.prod(self.hidden_dimensions[0]))
        x = F.leaky_relu(self.hierarchical_encoder_hidden_layer(x), negative_slope=0.01)
        x = torch.tanh(self.hierarchical_encoder_output_layer(x))
        fc_z_e1 = x.view(-1, *self.hidden_dimensions[2])
        k1, fc_embedding_1, fc_z_q1 = vq_1d(fc_z_e1, self.vq_embeddings[1].weight)

        x = fc_z_q1.view(-1, np.prod(self.hidden_dimensions[2]))
        x = F.leaky_relu(self.hierarchical_decoder_hidden_layer(x), negative_slope=0.01)
        x = self.hierarchical_decoder_output_layer(x)
        cnn_z_q0_high_level = x.view(-1, *self.hidden_dimensions[0])

        # cnn decoder
        cnn_z_e0 = hierarchy_usages[:,0].view(-1, 1, 1, 1) * cnn_z_q0_low_level + (1 - hierarchy_usages[:,0].view(-1, 1, 1, 1)) * cnn_z_q0_high_level
        x = cnn_z_e0
        
        for layer in self.cnn_decoder_hidden_layers:
            x = F.leaky_relu(layer(x), negative_slope=0.01)
        x = torch.sigmoid(self.cnn_decoder_output_layers(x))
        reconstructed_observation = x
        reconstructed_observation = {"observation_2d" : reconstructed_observation}

        # policy decoder
        x = cnn_z_e0.view(-1, np.prod(self.hidden_dimensions[0]))
        x = F.leaky_relu(self.policy_hidden_layer(x), negative_slope=0.01)
        x = self.policy_output_layer(x)
        policy = x

        return [(cnn_z_e0, cnn_embedding_0), (fc_z_e1, fc_embedding_1)], [k0, k1], reconstructed_observation, policy

    def state_forward(self, observations):
        x = observations["observation_2d"]

        for layer in self.cnn_encoder_hidden_layers:
            x = F.leaky_relu(layer(x), negative_slope=0.01)
        x = self.cnn_encoder_output_layers(x)

        cnn_z_e0 = x

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

        k0, cnn_embedding_0, cnn_z_q0_low_level = vq_2d(cnn_z_e0, self.vq_embeddings[0].weight)

        # hierarchical
        x = cnn_z_e0.detach().view(-1, np.prod(self.hidden_dimensions[0]))
        x = F.leaky_relu(self.hierarchical_encoder_hidden_layer(x), negative_slope=0.01)
        x = self.hierarchical_encoder_output_layer(x)
        fc_z_e1 = x.view(-1, *self.hidden_dimensions[2])
        k1, fc_embedding_1, fc_z_q1 = vq_1d(fc_z_e1, self.vq_embeddings[1].weight)

        return [k0, k1]


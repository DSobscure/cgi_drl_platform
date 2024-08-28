import torch
import torch.nn as nn
import torch.nn.functional as F

def leaky_relu(x):
    return F.leaky_relu(x, negative_slope=0.01)

class PredictionModel(nn.Module):
    def __init__(self, network_settings):
        super(PredictionModel, self).__init__()
        self.input_dimension = network_settings["input_dimension"]
        self.hidden_dimension = network_settings["hidden_dimension"]

        self.category_encoder_hidden_layers = nn.ModuleList([
            nn.Linear(self.input_dimension, self.hidden_dimension),
            nn.Linear(self.hidden_dimension, self.hidden_dimension),
        ])
        # tensorflow weight initialization
        for layer in self.category_encoder_hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.category_encoder_output_layer = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        # tensorflow weight initialization
        nn.init.xavier_uniform_(self.category_encoder_output_layer.weight)
        nn.init.zeros_(self.category_encoder_output_layer.bias)

        self.residual_winvalue_predictor_hidden_layers = nn.ModuleList([
            nn.Linear(self.hidden_dimension * 2, self.hidden_dimension),
            nn.Linear(self.hidden_dimension, self.hidden_dimension),
            nn.Linear(self.hidden_dimension, self.hidden_dimension),
            nn.Linear(self.hidden_dimension, self.hidden_dimension),
        ])
        # tensorflow weight initialization
        for layer in self.residual_winvalue_predictor_hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.residual_winvalue_predictor_output_layer = nn.Linear(self.hidden_dimension, 1)
        # tensorflow weight initialization
        nn.init.xavier_uniform_(self.residual_winvalue_predictor_output_layer.weight)
        nn.init.zeros_(self.residual_winvalue_predictor_output_layer.bias)

        self.embedding_count = network_settings["vq_embedding_size"][0]
        self.embedding_dimension = network_settings["vq_embedding_size"][1]
        self.vq_embeddings = nn.Embedding(self.embedding_count, self.embedding_dimension)
        nn.init.trunc_normal_(self.vq_embeddings.weight, mean=0, std=0.02)
        # self.vq_embeddings.weight.data.uniform_(-1/self.embedding_count, 1/self.embedding_count)

    def forward(self, observations):
        x_player1_combo = observations["player1_combo_input"]
        x_player2_combo = observations["player2_combo_input"]

        for layer in self.category_encoder_hidden_layers:
            x_player1_combo = leaky_relu(layer(x_player1_combo))
            x_player2_combo = leaky_relu(layer(x_player2_combo))
        
        x_player1_combo = self.category_encoder_output_layer(x_player1_combo)
        x_player2_combo = self.category_encoder_output_layer(x_player2_combo)

        def vq(latent_point, embedding_points):
            # (a-b)^2 = a^2 + b^2 - 2ab
            vq_distance = (torch.sum(latent_point**2, dim=-1, keepdim=True) + torch.sum(embedding_points**2, dim=-1) - 2 * torch.matmul(latent_point, embedding_points.t()))
            k = torch.argmin(vq_distance, dim=-1)
            z_decoder = torch.index_select(embedding_points, 0, k)
            z_gradient_copy = latent_point + (z_decoder - latent_point).detach()
            return k, z_decoder, z_gradient_copy

        player1_embedding_k, player1_embedding_decoder_latent, player1_decoder_latent = vq(x_player1_combo, self.vq_embeddings.weight)
        player2_embedding_k, player2_embedding_decoder_latent, player2_decoder_latent = vq(x_player2_combo, self.vq_embeddings.weight)

        x1 = torch.cat([player1_decoder_latent, player2_decoder_latent], dim=-1)
        x2 = torch.cat([player2_decoder_latent, player1_decoder_latent], dim=-1)

        for layer in self.residual_winvalue_predictor_hidden_layers:
            x1 = leaky_relu(layer(x1))
            x2 = leaky_relu(layer(x2))
            
        x1 = torch.tanh(self.residual_winvalue_predictor_output_layer(x1))
        x2 = torch.tanh(self.residual_winvalue_predictor_output_layer(x2))

        return (x_player1_combo, player1_embedding_decoder_latent), (x_player2_combo, player2_embedding_decoder_latent), player1_embedding_k, player2_embedding_k, (x1 - x2) / 2

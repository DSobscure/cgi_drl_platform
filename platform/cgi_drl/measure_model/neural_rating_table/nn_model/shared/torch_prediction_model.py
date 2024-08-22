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
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.input_dimension, self.hidden_dimension),
            nn.Linear(self.hidden_dimension, self.hidden_dimension),
            nn.Linear(self.hidden_dimension, self.hidden_dimension),
            nn.Linear(self.hidden_dimension, self.hidden_dimension)
        ])
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.output_layer = nn.Linear(self.hidden_dimension, 1)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, observations):
        x_player1_combo = observations["player1_combo_input"]
        x_player2_combo = observations["player2_combo_input"]
        
        for layer in self.hidden_layers:
            x_player1_combo = leaky_relu(layer(x_player1_combo))
            x_player2_combo = leaky_relu(layer(x_player2_combo))
        
        x_player1_combo = torch.exp(self.output_layer(x_player1_combo))
        x_player2_combo = torch.exp(self.output_layer(x_player2_combo))
        
        return x_player1_combo, x_player2_combo

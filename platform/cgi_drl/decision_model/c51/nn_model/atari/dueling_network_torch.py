import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyModel(nn.Module):
    def __init__(self, network_settings):
        super(PolicyModel, self).__init__()

        self.action_space = network_settings["action_space"]
        self.value_head_count = network_settings["value_head_count"]
        self.value_atom_count = network_settings["value_atom_count"]
        self.value_atom_scale_supports = network_settings["value_atom_scale_supports"]

        self.conv = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)

        self.value_hidden_heads = nn.ModuleList()
        self.value_output_heads = nn.ModuleList()
        self.advantage_hidden_heads = nn.ModuleList()
        self.advantage_output_heads = nn.ModuleList()
        
        for i_head in range(self.value_head_count):
            self.value_hidden_heads.append(nn.Linear(7*7*64, 512))
            self.value_output_heads.append(nn.Linear(512, self.value_atom_count))
            same_dimension_hidden_advantages = nn.ModuleList()
            same_dimension_output_advantages = nn.ModuleList()
            for space in self.action_space:
                same_dimension_hidden_advantages.append(nn.Linear(7*7*64, 512))
                same_dimension_output_advantages.append(nn.Linear(512, space * self.value_atom_count))
            self.advantage_hidden_heads.append(same_dimension_hidden_advantages)
            self.advantage_output_heads.append(same_dimension_output_advantages)

    def forward(self, observations):
        x = observations["observation_2d"]

        x = self.conv(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x_shared = x.view(-1, 7*7*64)

        q_values = []
        q_distribution_logits = []
        q_distributions = []

        for i_head in range(self.value_head_count):
            x_v = self.value_hidden_heads[i_head](x_shared)
            x_v = F.relu(x_v)
            x_v = self.value_output_heads[i_head](x_v)
            x_v = x_v.view(-1, 1, self.value_atom_count)

            same_dimension_q_values = []
            same_dimension_q_distribution_logits = []
            same_dimension_q_distributions = []
            for i_space in range(len(self.action_space)):
                x_a = self.advantage_hidden_heads[i_head][i_space](x_shared)
                x_a = F.relu(x_a)
                x_a = self.advantage_output_heads[i_head][i_space](x_a)
                x_a = x_a.view(-1, self.action_space[i_space], self.value_atom_count)
                
                x_logits = x_v + x_a - x_a.mean(dim=-2, keepdim=True)
                
                x_probabilities = F.softmax(x_logits, dim=-1)
                x_q_values = torch.sum(self.value_atom_scale_supports * x_probabilities, dim=-1)

                same_dimension_q_values.append(x_q_values)
                same_dimension_q_distribution_logits.append(x_logits)
                same_dimension_q_distributions.append(x_probabilities)
            q_values.append(same_dimension_q_values)
            q_distribution_logits.append(same_dimension_q_distribution_logits)
            q_distributions.append(same_dimension_q_distributions)

        return {
            "q_value" : q_values,
            "q_distribution_logit" : q_distribution_logits,
            "q_distribution" : q_distributions,
        }
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyModel(nn.Module):
    def __init__(self, network_settings):
        super(PolicyModel, self).__init__()

        self.action_space = network_settings["action_space"]
        self.value_head_count = network_settings["value_head_count"]

        self.conv = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)

        self.value_hidden_heads = nn.ModuleList()
        self.value_output_heads = nn.ModuleList()
        self.advantage_hidden_heads = nn.ModuleList()
        self.advantage_output_heads = nn.ModuleList()
        
        for i_head in range(self.value_head_count):
            self.value_hidden_heads.append(nn.Linear(7*7*64, 512))
            self.value_output_heads.append(nn.Linear(512, 1))
            same_dimension_hidden_advantages = nn.ModuleList()
            same_dimension_output_advantages = nn.ModuleList()
            for space in self.action_space:
                same_dimension_hidden_advantages.append(nn.Linear(7*7*64, 512))
                same_dimension_output_advantages.append(nn.Linear(512, space))
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

        for i_head in range(self.value_head_count):
            x_v = self.value_hidden_heads[i_head](x_shared)
            x_v = F.relu(x_v)
            x_v = self.value_output_heads[i_head](x_v)

            same_dimension_q_values = []
            for i_space in range(len(self.action_space)):
                x_a = self.advantage_hidden_heads[i_head][i_space](x_shared)
                x_a = F.relu(x_a)
                x_a = self.advantage_output_heads[i_head][i_space](x_a)

                x_q = x_v + x_a - torch.mean(x_a, dim=-1, keepdim=True)
                same_dimension_q_values.append(x_q)
            q_values.append(same_dimension_q_values)

        return q_values
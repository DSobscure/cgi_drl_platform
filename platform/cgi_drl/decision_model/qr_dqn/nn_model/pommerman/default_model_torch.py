import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyModel(nn.Module):
    def __init__(self, network_settings):
        super(PolicyModel, self).__init__()

        self.action_space = network_settings["action_space"]
        self.value_head_count = network_settings["value_head_count"]
        self.value_quantile_count  = network_settings["value_quantile_count"]

        # [12*4, 8, 8]
        self.conv = nn.Conv2d(12*4, 64, kernel_size=3, padding=0, stride=1)
        # [64, 6, 6]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=0, stride=1)
        # [128, 4, 4]
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=1)
        # [256, 2, 2]
        self.conv4 = nn.Conv2d(256, 256, kernel_size=2, padding=0, stride=1)
        # [256, 1, 1]
        
        self.fc = nn.Linear(256 + 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)

        self.value_hidden_heads = nn.ModuleList()
        self.value_output_heads = nn.ModuleList()
        self.advantage_hidden_heads = nn.ModuleList()
        self.advantage_output_heads = nn.ModuleList()
        
        for i_head in range(self.value_head_count):
            self.value_hidden_heads.append(nn.Linear(128, 128))
            self.value_output_heads.append(nn.Linear(128, self.value_quantile_count))
            same_dimension_hidden_advantages = nn.ModuleList()
            same_dimension_output_advantages = nn.ModuleList()
            for space in self.action_space:
                same_dimension_hidden_advantages.append(nn.Linear(128, 128))
                same_dimension_output_advantages.append(nn.Linear(128, space * self.value_quantile_count))
            self.advantage_hidden_heads.append(same_dimension_hidden_advantages)
            self.advantage_output_heads.append(same_dimension_output_advantages)

    def forward(self, observations):
        x_visual = observations["observation_visual"]
        x_vector = observations["observation_vector"]

        x_visual = self.conv(x_visual)
        x_visual = F.relu(x_visual)

        x_visual = self.conv2(x_visual)
        x_visual = F.relu(x_visual)

        x_visual = self.conv3(x_visual)
        x_visual = F.relu(x_visual)
        
        x_visual = self.conv4(x_visual)
        x_visual = F.relu(x_visual)

        x_visual = x_visual.view(-1, 256)
        x = torch.concat([x_visual, x_vector], dim=1)
        
        x = self.fc(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x_shared = F.relu(x)

        q_values = []
        q_quantile_values = []

        for i_head in range(self.value_head_count):
            x_v = self.value_hidden_heads[i_head](x_shared)
            x_v = F.relu(x_v)
            x_v = self.value_output_heads[i_head](x_v)
            x_v = x_v.view(-1, 1, self.value_quantile_count)

            same_dimension_q_values = []
            same_dimension_q_quantile_values = []
            for i_space in range(len(self.action_space)):
                x_a = self.advantage_hidden_heads[i_head][i_space](x_shared)
                x_a = F.relu(x_a)
                x_a = self.advantage_output_heads[i_head][i_space](x_a)
                x_a = x_a.view(-1, self.action_space[i_space], self.value_quantile_count)
                
                x_quantile_values = x_v + x_a - x_a.mean(dim=-2, keepdim=True)
                x_q_values = torch.mean(x_quantile_values, dim=-1)

                same_dimension_q_values.append(x_q_values)
                same_dimension_q_quantile_values.append(x_quantile_values)
            q_values.append(same_dimension_q_values)
            q_quantile_values.append(same_dimension_q_quantile_values)

        return {
            "q_value" : q_values,
            "q_quantile_value" : q_quantile_values,
        }
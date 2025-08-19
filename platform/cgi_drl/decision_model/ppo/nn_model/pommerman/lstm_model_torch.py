import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np

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

class PolicyModel(nn.Module):
    def __init__(self, network_settings):
        super(PolicyModel, self).__init__()

        self.action_space = network_settings["action_space"]
        self.value_head_count = network_settings["value_head_count"]
        self.memory_size = network_settings["memory_size"]

        # [12, 8, 8]
        self.conv = nn.Conv2d(12, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        # [32, 8, 8]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        # [64, 8, 8]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        num_blocks = 2
        self.residual_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(num_blocks)])
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.lstm = nn.LSTM(128 + 4  + np.sum(self.action_space) + self.value_head_count, self.memory_size // 2)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.value_hiddens = nn.ModuleList()
        self.value_heads = nn.ModuleList()
        for i_head in range(self.value_head_count):
            fc = nn.Linear(self.memory_size // 2, 128)
            self.value_hiddens.append(fc)
            fc = nn.Linear(128, 1)
            self.value_heads.append(fc)
        
        self.policy_hiddens = nn.ModuleList()
        self.policy_heads = nn.ModuleList()
        for i_space in self.action_space:
            fc = nn.Linear(self.memory_size // 2, 128)
            self.policy_hiddens.append(fc)
            fc = nn.Linear(128, i_space)
            self.policy_heads.append(fc)

    def forward(self, observations, rnn_sequence_length, rnn_burn_in_length):
        x_visual = observations["observation_visual"]
        x_vector = observations["observation_vector"]
        x_memory = observations["observation_memory"]
        x_previous_reward = observations["observation_previous_reward"].view(-1, self.value_head_count)
        x_previous_action = observations["observation_previous_action"]

        x = self.conv(x_visual)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        for residual_block in self.residual_blocks:
            x = residual_block(x)
            
        x = self.global_avg_pool(x)
        
        x = x.view(-1, 128)
        
        feature_list = [x, x_vector, x_previous_reward]
        for i_space in range(len(self.action_space)):
            one_hot_action = F.one_hot(x_previous_action[:, i_space], num_classes=self.action_space[i_space]).float()
            feature_list.append(one_hot_action)
        x = torch.concat(feature_list, dim=-1)

        # LSTM
        sequence_count = x.size(0) // rnn_sequence_length
        x = x.view(sequence_count, rnn_sequence_length, -1)

        h0, c0 = x_memory.view(sequence_count, rnn_sequence_length, -1)[:,0,:].chunk(2, dim=-1)
        h0 = h0.view(self.lstm.num_layers, sequence_count, self.memory_size // 2)
        c0 = c0.view(self.lstm.num_layers, sequence_count, self.memory_size // 2)
        lstm_state = (h0, c0)

        x = x.transpose(0, 1).contiguous()

        burn_in_output = []
        if rnn_burn_in_length > 0:
            with torch.no_grad():
                burn_in_x = x[:rnn_burn_in_length]
                for t in range(rnn_burn_in_length):
                    lstm_state = (lstm_state[0].contiguous(), lstm_state[1].contiguous())
                    out, lstm_state = self.lstm(burn_in_x[t].unsqueeze(0), lstm_state)
                    burn_in_output.append(out)

        main_x = x[rnn_burn_in_length:]
        main_output = []
        for t in range(main_x.size(0)):
            lstm_state = (lstm_state[0].contiguous(), lstm_state[1].contiguous())
            out, lstm_state = self.lstm(main_x[t].unsqueeze(0), lstm_state)
            main_output.append(out)

        full_output = burn_in_output + main_output
        full_output = torch.cat(full_output, dim=0)
        x = full_output.transpose(0, 1).contiguous().view(-1, self.memory_size // 2)

        next_memory = torch.cat((lstm_state[0].view(-1, self.memory_size // 2), lstm_state[1].view(-1, self.memory_size // 2)), dim=-1)

        value_heads = []
        for head in self.value_heads:
            x_value = self.value_hiddens[0](x)
            x_value = F.relu(x_value)
            x_value = head(x_value)
            value_heads.append(x_value)

        policy_heads = []
        policy_probability = []
        sampled_actions = []
        max_actions = []
        entropy = 0

        for head in self.policy_heads:
            x_policy = self.policy_hiddens[0](x)
            x_policy = F.relu(x_policy)
            x_policy = head(x_policy)
            policy_distribution = Categorical(logits=x_policy)
            action = policy_distribution.sample()
            max_action = torch.argmax(x_policy, dim=-1)

            policy_heads.append(x_policy)
            policy_probability.append(policy_distribution)
            sampled_actions.append(action)
            max_actions.append(max_action)
            entropy += policy_distribution.entropy().mean()

        return {
            "sample_action": sampled_actions,
            "max_action": max_actions,
            "value": value_heads,
            "entropy": entropy,
            "policy_distribution": policy_probability,
            "next_memory": next_memory
        }
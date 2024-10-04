import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyModel(nn.Module):
    def __init__(self, network_settings):
        super(PolicyModel, self).__init__()
        self.invertible_value_function = network_settings["invertible_value_function"]

        self.action_space = network_settings["action_space"]
        self.value_head_count = network_settings["value_head_count"]
        self.value_quantile_count  = network_settings["value_quantile_count"]
        self.memory_size = network_settings["memory_size"]

        self.conv = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.lstm = nn.LSTM(64*7*7 + np.sum(self.action_space) + self.value_head_count, self.memory_size // 2)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.value_hidden_heads = nn.ModuleList()
        self.value_output_heads = nn.ModuleList()
        self.advantage_hidden_heads = nn.ModuleList()
        self.advantage_output_heads = nn.ModuleList()
        
        for i_head in range(self.value_head_count):
            self.value_hidden_heads.append(nn.Linear(self.memory_size // 2, 512))
            self.value_output_heads.append(nn.Linear(512, self.value_quantile_count))
            same_dimension_hidden_advantages = nn.ModuleList()
            same_dimension_output_advantages = nn.ModuleList()
            for space in self.action_space:
                same_dimension_hidden_advantages.append(nn.Linear(self.memory_size // 2, 512))
                same_dimension_output_advantages.append(nn.Linear(512, space * self.value_quantile_count))
            self.advantage_hidden_heads.append(same_dimension_hidden_advantages)
            self.advantage_output_heads.append(same_dimension_output_advantages)

    def forward(self, observations, rnn_sequence_length, rnn_burn_in_length):
        x = observations["observation_2d"]
        x_memory = observations["observation_memory"]
        x_previous_reward = self.invertible_value_function(observations["observation_previous_reward"].view(-1, self.value_head_count))
        x_previous_action = observations["observation_previous_action"]

        x = self.conv(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.view(-1, 64*7*7)
        
        feature_list = [x, x_previous_reward]
        for i_space in range(len(self.action_space)):
            one_hot_action = F.one_hot(x_previous_action[:, i_space], num_classes=self.action_space[i_space]).float()
            feature_list.append(one_hot_action)
        x = torch.concat(feature_list, dim=-1)
        
        sequence_count = x.size(0) // rnn_sequence_length
        x = x.view(sequence_count, rnn_sequence_length, -1)
        x = x.transpose(0, 1).contiguous()

        h0, c0 = x_memory.view(sequence_count, rnn_sequence_length, -1)[:,0,:].chunk(2, dim=-1)
        h0 = h0.view(self.lstm.num_layers, sequence_count, self.memory_size // 2)
        c0 = c0.view(self.lstm.num_layers, sequence_count, self.memory_size // 2)
        lstm_state = (h0, c0)

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
        
        x_shared = x

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
            "next_memory": next_memory,
        }
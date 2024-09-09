import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class PolicyModel(nn.Module):
    def __init__(self, network_settings):
        super(PolicyModel, self).__init__()

        self.action_space = network_settings["action_space"]
        self.value_head_count = network_settings["value_head_count"]
        self.memory_size = network_settings["memory_size"]

        self.conv = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc = nn.Linear(64*7*7, 512)
        self.lstm = nn.LSTM(512, self.memory_size // 2)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        
        self.value_heads = nn.ModuleList()
        for i_head in range(self.value_head_count):
            fc = nn.Linear(self.memory_size // 2, 1)
            self.value_heads.append(fc)
        
        self.policy_heads = nn.ModuleList()
        for i_space in self.action_space:
            fc = nn.Linear(self.memory_size // 2, i_space)
            self.policy_heads.append(fc)

    def forward(self, observations, rnn_sequence_length, rnn_burn_in_length):
        x = observations["observation_2d"]
        x_memory = observations["observation_memory"]

        x = self.conv(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 64*7*7)
        x = self.fc(x)
        x = F.relu(x)     
        
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
            x_value = head(x)
            value_heads.append(x_value)

        policy_heads = []
        policy_probability = []
        sampled_actions = []
        max_actions = []
        entropy = 0

        for head in self.policy_heads:
            x_policy = head(x)
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
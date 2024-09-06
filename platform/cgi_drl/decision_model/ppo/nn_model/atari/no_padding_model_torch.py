import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class PolicyModel(nn.Module):
    def __init__(self, network_settings):
        super(PolicyModel, self).__init__()

        self.action_space = network_settings["action_space"]
        self.value_head_count = network_settings["value_head_count"]

        self.conv = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc = nn.Linear(64*7*7, 512)
        
        self.value_heads = nn.ModuleList()
        for i_head in range(self.value_head_count):
            fc = nn.Linear(512, 1)
            self.value_heads.append(fc)
        
        self.policy_heads = nn.ModuleList()
        for i_space in self.action_space:
            fc = nn.Linear(512, i_space)
            self.policy_heads.append(fc)

    def forward(self, observations):
        x = observations["observation_2d"]

        x = self.conv(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 64*7*7)
        x = self.fc(x)
        x = F.relu(x)     

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
        }
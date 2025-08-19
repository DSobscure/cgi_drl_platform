import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class PolicyModel(nn.Module):
    def __init__(self, network_settings):
        super(PolicyModel, self).__init__()

        self.action_space = network_settings["action_space"]
        self.value_head_count = network_settings["value_head_count"]

        # [12*4, 8, 8]
        self.conv = nn.Conv2d(12*4, 32, kernel_size=3, padding=1, stride=1)
        # [32, 8, 8]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        # [64, 8, 8]
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        # [64, 8, 8]
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        # [64, 8, 8]
        
        self.fc = nn.Linear(64*8*8 + 4 * 4, 256)

        self.value_heads = nn.ModuleList()
        for i_head in range(self.value_head_count):
            fc = nn.Linear(256, 1)
            self.value_heads.append(fc)
        
        self.policy_heads = nn.ModuleList()
        for i_space in self.action_space:
            fc = nn.Linear(256, i_space)
            self.policy_heads.append(fc)

    def forward(self, observations):
        x_visual = observations["observation_visual"]
        x_vector = observations["observation_vector"]

        x_visual = self.conv(x_visual)
        x_visual = F.leaky_relu(x_visual)

        x_visual = self.conv2(x_visual)
        x_visual = F.leaky_relu(x_visual)

        x_visual = self.conv3(x_visual)
        x_visual = F.leaky_relu(x_visual)
        
        x_visual = self.conv4(x_visual)
        x_visual = F.leaky_relu(x_visual)

        x_visual = x_visual.view(-1, 64*8*8)
        x = torch.concat([x_visual, x_vector], dim=1)
        
        x = self.fc(x)
        x = F.leaky_relu(x)

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
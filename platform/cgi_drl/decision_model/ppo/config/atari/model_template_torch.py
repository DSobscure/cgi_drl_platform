import torch
import numpy as np

class DefaultTemplate(dict):
    def __init__(self, config):
        self["value_head_count"] = config.get("value_head_count", 1)

        self["use_rnn"] = config.get("use_rnn", False)
        
        def observation_prodiver(observations, device):
            return {
                "observation_2d" : torch.tensor(np.asarray(observations["observation_2d"]), dtype=torch.float32).view(-1, 4, 84, 84).to(device),
            }
        def action_prodiver(actions, device):
            return torch.tensor(actions, dtype=torch.float32).to(device)
        def return_prodiver(returns, device):
            return torch.tensor(returns, dtype=torch.float32).to(device)
        def advantage_prodiver(advantages, device):
            return torch.tensor(advantages, dtype=torch.float32).to(device)
        self["observation_prodiver"] = observation_prodiver
        self["action_prodiver"] = action_prodiver
        self["return_prodiver"] = return_prodiver
        self["advantage_prodiver"] = advantage_prodiver

        def invertible_value_function(x, is_inverse):
            return x
        self["invertible_value_functions"] = config.get("invertible_value_functions", [invertible_value_function] * self["value_head_count"])

        super().__init__(config)
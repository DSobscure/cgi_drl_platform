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
            return torch.tensor(actions, dtype=torch.long).to(device)
        def target_q_values_prodiver(target_q_values, device):
            return torch.tensor(target_q_values, dtype=torch.float32).to(device)
        self["observation_prodiver"] = observation_prodiver
        self["action_prodiver"] = action_prodiver
        self["target_q_values_prodiver"] = target_q_values_prodiver
        
        self["model_define_path"] = config.get("model_define_path", "decision_model.dqn.nn_model.atari.dueling_networkl_torch")

        def invertible_value_function(x, is_inverse):
            epsilon = 0.001
            if is_inverse:
                return torch.sign(x) * (torch.square((torch.sqrt(1 + 4 * epsilon * (torch.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon)) - 1)
            else:
                return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x
        self["invertible_value_function"] = config.get("invertible_value_function", invertible_value_function)

        super().__init__(config)
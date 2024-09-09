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
        def target_q_distributions_prodiver(target_q_distributions, device):
            return torch.tensor(target_q_distributions, dtype=torch.float32).to(device)
        self["observation_prodiver"] = observation_prodiver
        self["action_prodiver"] = action_prodiver
        self["target_q_distributions_prodiver"] = target_q_distributions_prodiver
        
        self["model_define_path"] = config.get("model_define_path", "decision_model.c51.nn_model.atari.dueling_networkl_torch")

        # It is complex in combining categorical DQN Bellman operator with transformed Bellman operator.
        # We do not implement invertible_value_function on C51, but directly on QR-DQN
        
        self["value_atom_count"] = config.get("value_atom_count", 51)
        self["value_V_min"] = config.get("value_V_min", -10)
        self["value_V_max"] = config.get("value_V_max", 10)
        self["value_zero_index"] = config.get("value_zero_index", 25)

        super().__init__(config)
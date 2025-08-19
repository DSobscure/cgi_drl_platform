import torch
import numpy as np
import torch.nn.functional as F

class TorchTemplate(dict):
    def __init__(self, config):
        def observation_prodiver(observations, device):
            return {
                "observation_visual" : torch.tensor(np.asarray(observations["observation_visual"]), dtype=torch.float32).view(-1, 1, 8, 8, 12).permute(0, 1, 4, 2, 3).reshape(-1, 12, 8, 8).to(device),
                "observation_vector" : torch.tensor(np.asarray(observations["observation_vector"]), dtype=torch.float32).view(-1, 1 * 4).to(device),
            }
        def compressed_observation_prodiver(observations, device):
            return {
                "observation_visual" : torch.tensor(np.asarray(observations["observation_visual"]), dtype=torch.float32).view(-1, 1, 8, 8, 12).permute(0, 1, 4, 2, 3).reshape(-1, 12, 8, 8).to(device),
                "observation_vector" : torch.tensor(np.asarray(observations["observation_vector"]), dtype=torch.float32).view(-1, 1 * 4).to(device),
            }
        def hierarchy_usage_provider(hierarchy_usages, device):
            return torch.tensor(hierarchy_usages, dtype=torch.float32).to(device)
        def action_prodiver(actions, device):
            return F.one_hot(torch.tensor(actions, dtype=torch.long).view(-1), num_classes=6).float().to(device)

        self["observation_prodiver"] = observation_prodiver
        self["compressed_observation_prodiver"] = compressed_observation_prodiver
        self["hierarchy_usage_provider"] = hierarchy_usage_provider
        self["action_prodiver"] = action_prodiver

        self["vq_beta"] = config.get("vq_beta", 0.25)
        self["vq_mean_beta"] = config.get("vq_mean_beta", 0.25)

        self["model_define_path"] = config.get("model_define_path", "representation_model.hsd.nn_model.pommerman.torch_encoder_model")
        super().__init__(config)
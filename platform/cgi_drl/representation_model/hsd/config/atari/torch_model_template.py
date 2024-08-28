import torch

class TorchTemplate(dict):
    def __init__(self, config):
        def observation_prodiver(observations, device):
            return {
                "observation_2d" : torch.tensor(observations["observation_2d"], dtype=torch.float32).view(-1, 4, 84, 84).to(device),
            }
        def compressed_observation_prodiver(observations, device):
            return {
                "observation_2d" : torch.tensor(observations["observation_2d"], dtype=torch.float32).view(-1, 4, 84, 84).to(device),
            }
        def hierarchy_usage_provider(hierarchy_usages, device):
            return torch.tensor(hierarchy_usages, dtype=torch.float32).to(device)
        def action_prodiver(actions, device):
            return torch.tensor(actions, dtype=torch.float32).to(device)

        self["observation_prodiver"] = observation_prodiver
        self["compressed_observation_prodiver"] = compressed_observation_prodiver
        self["hierarchy_usage_provider"] = hierarchy_usage_provider
        self["action_prodiver"] = action_prodiver

        self["vq_beta"] = config.get("vq_beta", 0.25)
        self["vq_mean_beta"] = config.get("vq_mean_beta", 0.25)

        self["model_define_path"] = config.get("model_define_path", "measure_model.neural_counter_table.nn_model.shared.torch_prediction_model")
        super().__init__(config)

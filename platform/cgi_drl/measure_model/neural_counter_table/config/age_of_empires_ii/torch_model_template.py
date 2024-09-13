import torch

class DefaultTemplate(dict):
    def __init__(self, config):
        def observation_prodiver(observations, device):
            return {
                "player1_combo_input" : torch.tensor(observations["player1_combo"], dtype=torch.float32).to(device),
                "player2_combo_input" : torch.tensor(observations["player2_combo"], dtype=torch.float32).to(device)
            }

        def match_result_prodiver(match_results, device):
            return torch.tensor(match_results, dtype=torch.float32).view(-1, 1).to(device)

        def bt_prediction_prodiver(bt_predictions, device):
            return torch.tensor(bt_predictions, dtype=torch.float32).view(-1, 1).to(device)

        self["observation_prodiver"] = observation_prodiver
        self["match_result_prodiver"] = match_result_prodiver
        self["bt_prediction_prodiver"] = bt_prediction_prodiver

        self["vq_beta"] = config.get("vq_beta", 0.01)

        self["model_define_path"] = config.get("model_define_path", "measure_model.neural_counter_table.nn_model.shared.torch_prediction_model")
        super().__init__(config)
        self["network_settings"]["input_dimension"] = 45
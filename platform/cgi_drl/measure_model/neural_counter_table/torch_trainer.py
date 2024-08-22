import torch
import importlib
import numpy as np

class NeuralCounterTableTrainer():
    def __init__(self, config):
        PredictionModel = getattr(importlib.import_module(config["model_define_path"]), "PredictionModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vq_embedding_size = config["vq_embedding_size"]
        network_settings = config.get("network_settings", {})
        network_settings["vq_embedding_size"] = self.vq_embedding_size
        self.prediction_model = PredictionModel(network_settings).to(self.device)
        self.observation_prodiver = config["observation_prodiver"]
        self.match_result_prodiver = config["match_result_prodiver"]
        self.bt_prediction_prodiver = config["bt_prediction_prodiver"]
        self.optimizer = torch.optim.Adam(self.prediction_model.parameters())
        self.loss_fn = torch.nn.MSELoss()
        self.vq_beta = config["vq_beta"]
        self.vq_mean_beta = config.get("vq_mean_beta", 0.25)

    def update(self, observations, match_results, bt_predictions, extra_settings = None):         
        self.prediction_model.train()
        observation_inputs = self.observation_prodiver(observations, self.device)
        match_result_inputs = self.match_result_prodiver(match_results, self.device)
        bt_prediction_inputs = self.bt_prediction_prodiver(bt_predictions, self.device)

        for i in range(len(self.optimizer.param_groups)): 
            self.optimizer.param_groups[i]['lr'] = extra_settings["learning_rate"]

        self.optimizer.zero_grad()
        vq_embedding_mean = torch.mean(self.prediction_model.vq_embeddings.weight, dim=0) 
        vq_embedding_mean = vq_embedding_mean.unsqueeze(0).expand(bt_prediction_inputs.size(0), -1)
        x_player1_combo_latent_pair, x_player2_combo_latent_pair, player1_embedding_k, player2_embedding_k, residual_winvalue_prediction = self.prediction_model(observation_inputs)

        winvalue_prediction_loss = self.loss_fn(residual_winvalue_prediction, match_result_inputs - bt_prediction_inputs)

        player1_vq_loss = self.loss_fn(x_player1_combo_latent_pair[1], x_player1_combo_latent_pair[0].detach())
        player1_commit_loss = self.loss_fn(x_player1_combo_latent_pair[1].detach(), x_player1_combo_latent_pair[0])
        player1_vq_mean_loss = self.loss_fn(vq_embedding_mean, x_player1_combo_latent_pair[0].detach())

        player2_vq_loss = self.loss_fn(x_player2_combo_latent_pair[1], x_player2_combo_latent_pair[0].detach())
        player2_commit_loss = self.loss_fn(x_player2_combo_latent_pair[1].detach(), x_player2_combo_latent_pair[0])
        player2_vq_mean_loss = self.loss_fn(vq_embedding_mean, x_player2_combo_latent_pair[0].detach())

        vq_loss = (player1_vq_loss + player2_vq_loss) / 2
        commit_loss = (player1_commit_loss + player2_commit_loss) / 2
        vq_mean_loss = (player1_vq_mean_loss + player2_vq_mean_loss) / 2

        loss = winvalue_prediction_loss + vq_loss + self.vq_beta * commit_loss + self.vq_mean_beta * vq_mean_loss
        loss.backward()
        self.optimizer.step()

        return winvalue_prediction_loss.item(), vq_loss.item(), vq_mean_loss.item(), player1_embedding_k.tolist()

    def get_predictions(self, observations, bt_predictions, extra_settings = None):     
        self.prediction_model.eval()
        observation_inputs = self.observation_prodiver(observations, self.device)
        bt_prediction_inputs = self.bt_prediction_prodiver(bt_predictions, self.device)

        _, _, player1_embedding_k, player2_embedding_k, residual_winvalue_prediction = self.prediction_model(observation_inputs)
        winvalue_prediction = bt_prediction_inputs + residual_winvalue_prediction

        return winvalue_prediction.squeeze(1).cpu().detach().numpy(), player1_embedding_k.tolist(), player2_embedding_k.tolist()

    def save(self, path, time_step):
        '''save NN model (give a directory name for the model) '''
        torch.save(self.prediction_model.state_dict(), f"{path}/model_{time_step}.ckpt")
        print(f"Model saved in file: {path}/model_{time_step}.ckpt")

    def load(self, path):
        '''load NN model (give a directory name for the model) '''
        self.prediction_model.load_state_dict(torch.load(path))
        print("Model restored.")

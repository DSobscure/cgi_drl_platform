import torch
import importlib

class NeuralRatingTableTrainer():
    def __init__(self, config):
        PredictionModel = getattr(importlib.import_module(config["model_define_path"]), "PredictionModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_model = PredictionModel(config.get("network_settings", {})).to(self.device)
        self.observation_prodiver = config["observation_prodiver"]
        self.match_result_prodiver = config["match_result_prodiver"]
        self.optimizer = torch.optim.Adam(self.prediction_model.parameters())
        self.loss_fn = torch.nn.MSELoss()

    def update(self, observations, match_results, extra_settings = None):         
        self.prediction_model.train()
        observation_inputs = self.observation_prodiver(observations, self.device)
        match_result_inputs = self.match_result_prodiver(match_results, self.device)

        for i in range(len(self.optimizer.param_groups)): 
            self.optimizer.param_groups[i]['lr'] = extra_settings["learning_rate"]
        
        self.optimizer.zero_grad()
        player1_combo_rating, player2_combo_rating = self.prediction_model(observation_inputs)
        bradley_terry_winvalue_prediction = player1_combo_rating / (player1_combo_rating + player2_combo_rating)
        winvalue_prediction_loss = self.loss_fn(bradley_terry_winvalue_prediction, match_result_inputs)
        winvalue_prediction_loss.backward()
        self.optimizer.step()

        return winvalue_prediction_loss.item()

    def get_predictions(self, observations, extra_settings = None):        
        self.prediction_model.eval()
        observation_inputs = self.observation_prodiver(observations, self.device)

        player1_combo_rating, player2_combo_rating = self.prediction_model(observation_inputs)
        bradley_terry_winvalue_prediction = player1_combo_rating / (player1_combo_rating + player2_combo_rating)

        return bradley_terry_winvalue_prediction.cpu().detach().numpy()

    def get_rating_and_predictions(self, observations, extra_settings = None):            
        self.prediction_model.eval()
        observation_inputs = self.observation_prodiver(observations, self.device)

        player1_combo_rating, player2_combo_rating = self.prediction_model(observation_inputs)
        bradley_terry_winvalue_prediction = player1_combo_rating / (player1_combo_rating + player2_combo_rating)

        return player1_combo_rating.cpu().detach().numpy(), player2_combo_rating.cpu().detach().numpy(), bradley_terry_winvalue_prediction.cpu().detach().numpy()

    def save(self, path, time_step):
        '''save NN model (give a directory name for the model) '''
        torch.save(self.prediction_model.state_dict(), f"{path}/model_{time_step}.ckpt")
        print(f"Model saved in file: {path}/model_{time_step}.ckpt")

    def load(self, path):
        '''load NN model (give a directory name for the model) '''
        self.prediction_model.load_state_dict(torch.load(path))
        print("Model restored.")

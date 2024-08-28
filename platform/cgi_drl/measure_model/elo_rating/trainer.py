import tensorflow as tf
import importlib

class EloRatingTrainer():
    def __init__(self, config):
        self.rating_table = dict()
        self.initial_rating = config["initial_rating"]
        self.update_k = config["update_k"]

    def set_session(self, session):
        pass

    def update(self, observations, match_results, extra_settings = None):        
        loss = 0 
        for i_sample in range(len(match_results)):
            combo1 = observations["player1_combo"][i_sample]
            combo2 = observations["player2_combo"][i_sample]

            combo1_rating = self.rating_table.get(combo1, self.initial_rating)
            combo2_rating = self.rating_table.get(combo2, self.initial_rating)

            win_value = 1 / (1 + 10 ** ((combo2_rating - combo1_rating) / 400))
            match_result = match_results[i_sample]

            loss += (match_result - win_value) ** 2

            self.rating_table[combo1] = combo1_rating + self.update_k * (match_result - win_value)
            self.rating_table[combo2] = combo2_rating + self.update_k * ((1 - match_result) - (1 - win_value))
            
        return loss / len(match_results)


    def get_predictions(self, observations, extra_settings = None):            
        predictions = []
        for i_sample in range(len(observations["player1_combo"])):
            combo1_rating = self.rating_table.get(observations["player1_combo"][i_sample], self.initial_rating)
            combo2_rating = self.rating_table.get(observations["player2_combo"][i_sample], self.initial_rating)
            win_value = 1 / (1 + 10 ** ((combo2_rating - combo1_rating)/400))
            predictions.append(win_value)

        return predictions

    def save(self, path, time_step):
        pass

    def load(self, path):
        pass

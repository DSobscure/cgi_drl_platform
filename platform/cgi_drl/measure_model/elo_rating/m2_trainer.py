import numpy as np
import importlib

def sigmoid(x):
    return 1 / (1 + np.exp(min(-x, 100)))

class EloRatingTrainer():
    def __init__(self, config):
        self.rating_table = dict()
        self.c_table = dict()
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

            if combo1 not in self.c_table:
                self.c_table[combo1] = np.random.uniform(-10, 10, 2)
            combo1_c = self.c_table[combo1]
            if combo2 not in self.c_table:
                self.c_table[combo2] = np.random.uniform(-10, 10, 2)
            combo2_c = self.c_table[combo2]

            win_value = sigmoid(combo1_rating - combo2_rating + combo1_c[0] * combo2_c[1] - combo2_c[0] * combo1_c[1])
            match_result = match_results[i_sample]

            loss += (match_result - win_value) ** 2

            self.rating_table[combo1] = combo1_rating + self.update_k * (match_result - win_value)
            self.rating_table[combo2] = combo2_rating + self.update_k * ((1 - match_result) - (1 - win_value))

            self.c_table[combo1] = combo1_c + np.asarray([(match_result - win_value) * combo2_c[1], -(match_result - win_value) * combo1_c[1]])
            self.c_table[combo2] = combo2_c + np.asarray([-(match_result - win_value) * combo2_c[0], (match_result - win_value) * combo1_c[0]])

            self.c_table[combo1] = np.clip(self.c_table[combo1], -1000, 1000)
            self.c_table[combo2] = np.clip(self.c_table[combo2], -1000, 1000)
            
        return loss / len(match_results)


    def get_predictions(self, observations, extra_settings = None):            
        predictions = []
        for i_sample in range(len(observations["player1_combo"])):
            combo1 = observations["player1_combo"][i_sample]
            combo2 = observations["player2_combo"][i_sample]

            combo1_rating = self.rating_table.get(combo1, self.initial_rating)
            combo2_rating = self.rating_table.get(combo2, self.initial_rating)
            if combo1 not in self.c_table:
                self.c_table[combo1] = np.random.uniform(-10, 10, 2)
            combo1_c = self.c_table[combo1]
            if combo2 not in self.c_table:
                self.c_table[combo2] = np.random.uniform(-10, 10, 2)
            combo2_c = self.c_table[combo2]

            win_value = sigmoid(combo1_rating - combo2_rating + combo1_c[0] * combo2_c[1] - combo2_c[0] * combo1_c[1])
            predictions.append(win_value)

        return predictions

    def save(self, path, time_step):
        pass

    def load(self, path):
        pass

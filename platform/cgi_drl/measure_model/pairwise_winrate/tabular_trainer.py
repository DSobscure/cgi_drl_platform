import tensorflow as tf
import importlib

class PairwiseWinrateTrainer():
    def __init__(self, config):
        self.game_counts = dict()
        self.win_counts = dict()

    def set_session(self, session):
        pass

    def update(self, observations, match_results, extra_settings = None):         
        for i_sample in range(len(match_results)):
            combo1 = observations["player1_combo"][i_sample]
            combo2 = observations["player2_combo"][i_sample]

            combo = f"{combo1}|{combo2}"
            if combo not in self.game_counts:
                combo = f"{combo2}|{combo1}"
                if combo not in self.game_counts:
                    self.game_counts[combo] = 1
                    self.win_counts[combo] = 1 - match_results[i_sample]
                else:
                    self.game_counts[combo] += 1
                    self.win_counts[combo] += 1 - match_results[i_sample]
            else:
                self.game_counts[combo] += 1
                self.win_counts[combo] += match_results[i_sample]

        return 0

    def get_predictions(self, observations, extra_settings = None):            
        predictions = []
        for i_sample in range(len(observations["player1_combo"])):
            combo1 = observations["player1_combo"][i_sample]
            combo2 = observations["player2_combo"][i_sample]

            combo = f"{combo1}|{combo2}"
            if combo in self.game_counts:
                predictions.append(self.win_counts[combo]/self.game_counts[combo])
            else:
                combo = f"{combo2}|{combo1}"
                if combo in self.game_counts:
                    predictions.append(1 - self.win_counts[combo]/self.game_counts[combo])
                else:
                    predictions.append(-1)

        return predictions

    def save(self, path, time_step):
        pass

    def load(self, path):
        pass

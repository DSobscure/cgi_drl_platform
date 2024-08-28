class SimpleWinrateTrainer():
    def __init__(self, config):
        self.game_counts = dict()
        self.win_counts = dict()

    def set_session(self, session):
        pass

    def update(self, observations, match_results, extra_settings = None):    
        for i_sample in range(len(match_results)):
            if observations["player1_combo"][i_sample] not in self.game_counts:
                self.game_counts[observations["player1_combo"][i_sample]] = 0
                self.win_counts[observations["player1_combo"][i_sample]] = 0
            self.game_counts[observations["player1_combo"][i_sample]] += 1
            self.win_counts[observations["player1_combo"][i_sample]] += match_results[i_sample]
            if observations["player2_combo"][i_sample] not in self.game_counts:
                self.game_counts[observations["player2_combo"][i_sample]] = 0
                self.win_counts[observations["player2_combo"][i_sample]] = 0
            self.game_counts[observations["player2_combo"][i_sample]] += 1
            self.win_counts[observations["player2_combo"][i_sample]] += 1 - match_results[i_sample]

        return 0

    def get_predictions(self, observations, extra_settings = None):            
        predictions = []
        for combo in observations["player1_combo"]:
            if combo in self.game_counts:
                predictions.append(self.win_counts[combo]/self.game_counts[combo])
            else:
                predictions.append(-1)
        for combo in observations["player2_combo"]:
            if combo in self.game_counts:
                predictions.append(self.win_counts[combo]/self.game_counts[combo])
            else:
                predictions.append(-1)

        return predictions

    def save(self, path, time_step):
        pass

    def load(self, path):
        pass

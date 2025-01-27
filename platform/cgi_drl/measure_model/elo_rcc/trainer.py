import importlib
import numpy as np

def softmax(logits):   
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

class EloRccTrainer():
    def __init__(self, config):
        self.rating_table = dict()
        self.category_table = dict()
        self.expected_residual_table = dict()
        self.initial_rating = config["initial_rating"]
        self.update_k = config["update_k"]
        self.residual_learning_rate = config["residual_learning_rate"]
        self.category_learning_rate = config["category_learning_rate"]
        self.category_count = config["category_count"]
        self.counter_table = np.zeros([self.category_count, self.category_count])

    def set_session(self, session):
        pass

    def update(self, observations, match_results, extra_settings = None):        
        prediction_loss, residual_loss, cross_entropy = 0, 0, 0
        for i_sample in range(len(match_results)):
            combo1 = observations["player1_combo"][i_sample]
            combo2 = observations["player2_combo"][i_sample]
            match_result = match_results[i_sample]
            
            combo1_rating = self.rating_table.get(combo1, self.initial_rating)
            combo2_rating = self.rating_table.get(combo2, self.initial_rating)

            win_value = 1 / (1 + 10 ** ((combo2_rating - combo1_rating) / 400))
            prediction_loss += (match_result - win_value) ** 2
            
            # Elo rating update
            self.rating_table[combo1] = combo1_rating + self.update_k * (match_result - win_value)
            self.rating_table[combo2] = combo2_rating + self.update_k * ((1 - match_result) - (1 - win_value))
            
            combo1_logits = self.category_table.get(combo1, np.zeros(self.category_count))
            combo2_logits = self.category_table.get(combo2, np.zeros(self.category_count))
            
            combo1_rating = self.rating_table.get(combo1, self.initial_rating)
            combo2_rating = self.rating_table.get(combo2, self.initial_rating)

            win_value = 1 / (1 + 10 ** ((combo2_rating - combo1_rating) / 400))
            residual_win_value = np.clip(match_result - win_value, -1, 1)
            
            combo1_category = np.random.choice(self.category_count, p=softmax(combo1_logits))
            combo2_category = np.random.choice(self.category_count, p=softmax(combo2_logits))
                       
            residual_delta = residual_win_value - self.counter_table[combo1_category,combo2_category]
            self.counter_table[combo1_category,combo2_category] += self.residual_learning_rate * residual_delta
            if combo1_category == combo2_category:
                self.counter_table[combo1_category,combo2_category] = 0
            self.counter_table[combo2_category,combo1_category] = -self.counter_table[combo1_category,combo2_category]
            
            combo1_expected_residual = self.expected_residual_table.get(combo1, np.zeros(self.category_count))
            combo1_expected_residual[combo2_category] += self.residual_learning_rate * (residual_win_value - combo1_expected_residual[combo2_category])
            self.expected_residual_table[combo1] = combo1_expected_residual
            
            combo2_expected_residual = self.expected_residual_table.get(combo2, np.zeros(self.category_count))
            combo2_expected_residual[combo1_category] += self.residual_learning_rate * (-residual_win_value - combo2_expected_residual[combo1_category])
            self.expected_residual_table[combo2] = combo2_expected_residual
        
            # EM algorithm
            combo1_distances = np.abs(self.counter_table - self.expected_residual_table[combo1])
            combo2_distances = np.abs(self.counter_table - self.expected_residual_table[combo2])
            combo1_best_category = combo1_distances.sum(axis=1).argmin()
            combo2_best_category = combo2_distances.sum(axis=1).argmin()
            category_weight = 1

            combo1_label, combo2_label = np.zeros(self.category_count), np.zeros(self.category_count)
            combo1_label[combo1_best_category] = 1
            combo2_label[combo2_best_category] = 1

            cross_entropy1 = -np.sum(combo1_label * np.log(softmax(combo1_logits) + 1e-9))
            cross_entropy2 = -np.sum(combo2_label * np.log(softmax(combo2_logits) + 1e-9))

            self.category_table[combo1] = combo1_logits + self.category_learning_rate * category_weight * (combo1_label - softmax(combo1_logits))
            self.category_table[combo2] = combo2_logits + self.category_learning_rate * category_weight * (combo2_label - softmax(combo2_logits))

            residual_loss += residual_delta ** 2
            cross_entropy += cross_entropy1 + cross_entropy2
            
            
        return prediction_loss / len(match_results), residual_loss / len(match_results), cross_entropy / len(match_results)


    def get_predictions(self, observations, extra_settings = None):            
        predictions = []
        for i_sample in range(len(observations["player1_combo"])):
            combo1_rating = self.rating_table.get(observations["player1_combo"][i_sample], self.initial_rating)
            combo2_rating = self.rating_table.get(observations["player2_combo"][i_sample], self.initial_rating)
            win_value = 1 / (1 + 10 ** ((combo2_rating - combo1_rating)/400))
            combo1_logits = self.category_table.get(observations["player1_combo"][i_sample], np.zeros(self.category_count))
            combo1_category = np.argmax(combo1_logits)
            combo2_logits = self.category_table.get(observations["player2_combo"][i_sample], np.zeros(self.category_count))
            combo2_category = np.argmax(combo2_logits)
            win_value += self.counter_table[combo1_category, combo2_category]
            win_value = np.clip(win_value, 0, 1)
            predictions.append(win_value)

        return predictions

    def save(self, path, time_step):
        pass

    def load(self, path):
        pass

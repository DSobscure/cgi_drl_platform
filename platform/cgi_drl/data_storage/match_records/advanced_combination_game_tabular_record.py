import numpy as np
import pandas as pd

class AdvancedCombinationGameRecordGenerator(object):
    def __init__(self):
        self.element_count = 20
        self.combo_element_count = 3
        self.winner_bonus = 60

    def generate_match(self):
        combo1 = list(np.random.choice(self.element_count, self.combo_element_count, replace=False))
        combo2 = list(np.random.choice(self.element_count, self.combo_element_count, replace=False))

        combo1_score = np.sum(combo1) + 3
        combo2_score = np.sum(combo2) + 3
        combo1_category = combo1_score % 3
        combo2_category = combo2_score % 3

        # Determine the rock-paper-scissors result
        if combo1_category == combo2_category:
            rps_match_result = 0.5  # Tie
        elif combo1_category == 0 and combo2_category == 2:
            rps_match_result = 1  # Player 1 wins (Rock vs Scissors)
        elif combo1_category == 2 and combo2_category == 1:
            rps_match_result = 1  # Player 1 wins (Scissors vs Paper)
        elif combo1_category == 1 and combo2_category == 0:
            rps_match_result = 1  # Player 1 wins (Paper vs Rock)
        else:
            rps_match_result = 0  # Player 2 wins

        if rps_match_result == 1:
            combo1_score += self.winner_bonus
        elif rps_match_result == 0:
            combo2_score += self.winner_bonus
        else:
            pass
        combo1_score = combo1_score ** 2
        combo2_score = combo2_score ** 2
        winrate = combo1_score / (combo1_score + combo2_score)
        match_result = np.random.binomial(1, winrate)

        return combo1 + [20 + combo1_category], combo2 + [20 + combo2_category], match_result
    
class AdvancedCombinationGameMatchResultBuffer(object):
    def __init__(self, config):
        self.element_count = 20
        self.dataset_path = config["dataset_path"]

        self._combo1_buffer = []
        self._combo2_buffer = []
        self._result_buffer = []

        dataset = pd.read_csv(self.dataset_path)

        combo1_1 = dataset["combo1_1"].tolist()
        combo1_2 = dataset["combo1_2"].tolist()
        combo1_3 = dataset["combo1_3"].tolist()
        combo1_4 = dataset["combo1_4"].tolist()

        combo2_1 = dataset["combo2_1"].tolist()
        combo2_2 = dataset["combo2_2"].tolist()
        combo2_3 = dataset["combo2_3"].tolist()
        combo2_4 = dataset["combo2_4"].tolist()

        win_teams = dataset["result"].tolist()

        for i_index in range(len(combo1_1)):
            
            player1_combo = sorted([combo1_1[i_index], combo1_2[i_index], combo1_3[i_index], combo1_4[i_index]])
            player2_combo = sorted([combo2_1[i_index], combo2_2[i_index], combo2_3[i_index], combo2_4[i_index]])
            
            self._combo1_buffer.append(player1_combo)
            self._combo2_buffer.append(player2_combo)
            if win_teams[i_index] == 1:
                self._result_buffer.append(1)
            else:
                self._result_buffer.append(0)

        self.sample_size = len(self._result_buffer)

        self._win_counter = {}
        self._game_counter = {}
        self._average_result_buffer = []

        for i in range(self.sample_size):
            combo1_string = ",".join([str(c) for c in self._combo1_buffer[i]])
            combo2_string = ",".join([str(c) for c in self._combo2_buffer[i]])
            if combo1_string + "|" + combo2_string in self._win_counter:
                self._win_counter[combo1_string + "|" + combo2_string] += self._result_buffer[i]
                self._game_counter[combo1_string + "|" + combo2_string] += 1
            elif combo2_string + "|" + combo1_string in self._win_counter:
                self._win_counter[combo2_string + "|" + combo1_string] += 1 - self._result_buffer[i]
                self._game_counter[combo2_string + "|" + combo1_string] += 1
            else:
                self._win_counter[combo1_string + "|" + combo2_string] = self._result_buffer[i]
                self._game_counter[combo1_string + "|" + combo2_string] = 1

        for i in range(self.sample_size):
            combo1_string = ",".join([str(c) for c in self._combo1_buffer[i]])
            combo2_string = ",".join([str(c) for c in self._combo2_buffer[i]])
            if combo1_string + "|" + combo2_string in self._win_counter:
                self._average_result_buffer.append(self._win_counter[combo1_string + "|" + combo2_string] / self._game_counter[combo1_string + "|" + combo2_string])
            else:
                self._average_result_buffer.append(1 - self._win_counter[combo2_string + "|" + combo1_string] / self._game_counter[combo2_string + "|" + combo1_string])
            

    def encode_combo(self, combo):
        return ','.join(str(x) for x in combo)

    def size(self):
        return self.sample_size

    def __len__(self):
        return self.sample_size

    def random_sample_all_batch(self, batch_size):
        idx = np.random.permutation(self.sample_size)
        for i in range(0, self.sample_size, batch_size):
            if i + batch_size >= self.sample_size:
                batch_size = self.sample_size - i
            player1_combos = []
            player2_combos = []
            match_results = []
            in_reverse = np.random.choice(2, batch_size)
            for j in range(i, i + batch_size, 1):
                if in_reverse[j - i] == 0:
                    player1_combos.append(self.encode_combo(self._combo1_buffer[idx[j]]))
                    player2_combos.append(self.encode_combo(self._combo2_buffer[idx[j]]))
                    match_results.append(self._result_buffer[idx[j]])
                else:
                    player1_combos.append(self.encode_combo(self._combo2_buffer[idx[j]]))
                    player2_combos.append(self.encode_combo(self._combo1_buffer[idx[j]]))
                    match_results.append(1 - self._result_buffer[idx[j]])
            yield {
                "player1_combos": player1_combos,
                "player2_combos": player2_combos,
                "match_results": np.array(match_results, dtype=np.float32),
            }
            

    def sample_all_batch(self, batch_size):
        for i in range(0, self.sample_size, batch_size):
            if i + batch_size >= self.sample_size:
                batch_size = self.sample_size - i
            yield {
                "player1_combos": [self.encode_combo(self._combo1_buffer[j]) for j in range(i, i + batch_size, 1)],
                "player2_combos": [self.encode_combo(self._combo2_buffer[j]) for j in range(i, i + batch_size, 1)],
                "match_results": np.array([self._average_result_buffer[j] for j in range(i, i + batch_size, 1)], dtype=np.float32),
            }

    def sample_all_combos(self, batch_size):
        for i in range(0, self.sample_size, batch_size):
            if i + batch_size >= self.sample_size:
                batch_size = self.sample_size - i
            yield {
                "player1_combos": [self.encode_combo(self._combo1_buffer[j]) for j in range(i, i + batch_size, 1)],
                "player2_combos": [self.encode_combo(self._combo2_buffer[j]) for j in range(i, i + batch_size, 1)],
                "player1_raw_combos": [self._combo1_buffer[j] for j in range(i, i + batch_size, 1)],
                "player2_raw_combos": [self._combo2_buffer[j] for j in range(i, i + batch_size, 1)],
            }

if __name__ == '__main__':
    g = AdvancedCombinationGameRecordGenerator()
    record_count = 100000

    import csv
    with open('advanced_combination_game_records.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['combo1_1','combo1_2','combo1_3','combo1_4', 'combo2_1', 'combo2_2', 'combo2_3', 'combo2_4', 'result'])
        for i in range(record_count):
            print("{}/{}".format(i, record_count), end='\r')
            combo1, combo2, match_result = g.generate_match()
            writer.writerow([combo1[0], combo1[1], combo1[2], combo1[3], combo2[0], combo2[1], combo2[2], combo2[3], match_result])


import numpy as np
import pandas as pd

class AgeOfEmpiresIIMatchResultBuffer(object):
    def __init__(self, config):
        self.civilization_set = ['armenians', 'aztecs', 'bengalis', 'berbers', 'bohemians', 'britons', 'bulgarians', 'burgundians', 'burmese', 'byzantines', 'celts', 'chinese', 'cumans', 'dravidians', 'ethiopians', 'franks', 'georgians', 'goths', 'gurjaras', 'hindustanis', 'huns', 'incas', 'italians', 'japanese', 'khmer', 'koreans', 'lithuanians', 'magyars', 'malay', 'malians', 'mayans', 'mongols', 'persians', 'poles', 'portuguese', 'romans', 'saracens', 'sicilians', 'slavs', 'spanish', 'tatars', 'teutons', 'turks', 'vietnamese', 'vikings'] 
        self.dataset_path = config["dataset_path"]

        self._combo1_buffer = []
        self._combo2_buffer = []
        self._result_buffer = []

        dataset = pd.read_csv(self.dataset_path)

        self._combo1_buffer = dataset["Civilization 1"].tolist()
        self._combo2_buffer = dataset["Civilization 2"].tolist()
        self._result_buffer = dataset["Result"].tolist()
        self.sample_size = len(self._result_buffer)

        self._win_counter = {}
        self._game_counter = {}
        self._average_result_buffer = []

        for i in range(self.sample_size):
            combo1_string = str(self._combo1_buffer[i])
            combo2_string = str(self._combo2_buffer[i])
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
            combo1_string = str(self._combo1_buffer[i])
            combo2_string = str(self._combo2_buffer[i])
            if combo1_string + "|" + combo2_string in self._win_counter:
                self._average_result_buffer.append(self._win_counter[combo1_string + "|" + combo2_string] / self._game_counter[combo1_string + "|" + combo2_string])
            else:
                self._average_result_buffer.append(1 - self._win_counter[combo2_string + "|" + combo1_string] / self._game_counter[combo2_string + "|" + combo1_string])

    def encode_combo(self, combo):
        return combo

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
                "match_results": np.array([self._average_result_buffer[j] for j in range(i, i + batch_size, 1)], dtype=np.float32),
                "player1_combos": [self._combo1_buffer[j] for j in range(i, i + batch_size, 1)],
                "player2_combos": [self._combo2_buffer[j] for j in range(i, i + batch_size, 1)],
            }

    def sample_all_combos(self, batch_size):
        for i in range(0, self.sample_size, batch_size):
            if i + batch_size >= self.sample_size:
                batch_size = self.sample_size - i
            yield {
                "player1_raw_combos": [self._combo1_buffer[j] for j in range(i, i + batch_size, 1)],
                "player2_raw_combos": [self._combo2_buffer[j] for j in range(i, i + batch_size, 1)],
                "player1_combos": [self._combo1_buffer[j] for j in range(i, i + batch_size, 1)],
                "player2_combos": [self._combo2_buffer[j] for j in range(i, i + batch_size, 1)],
            }

if __name__ == '__main__':
    replay = AgeOfEmpiresIIMatchResultBuffer({
        "dataset_path" : "~/balance/AoE2/aoestats_1v1_random_map_elo_all.csv",
    })
    print(replay.size()) # 1261288
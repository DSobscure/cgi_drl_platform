import numpy as np
import pandas as pd

class LoLMatchResultBuffer(object):
    def __init__(self, config):
        self.dataset_path = config["dataset_path"]
        self.hero_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 48, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 68, 69, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 89, 90, 91, 92, 96, 98, 99, 101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114, 115, 117, 119, 120, 121, 122, 126, 127, 131, 133, 134, 136, 143, 150, 154, 157, 161, 163, 164, 201, 202, 203, 222, 223, 236, 238, 240, 245, 254, 266, 267, 268, 412, 420, 421, 427, 429, 432, 497, 498]

        self._combo1_buffer = []
        self._combo2_buffer = []
        self._result_buffer = []

        dataset = pd.read_csv(self.dataset_path)

        heros1_1 = dataset["win_member1"].tolist()
        heros1_2 = dataset["win_member2"].tolist()
        heros1_3 = dataset["win_member3"].tolist()
        heros1_4 = dataset["win_member4"].tolist()
        heros1_5 = dataset["win_member5"].tolist()

        heros2_1 = dataset["loss_member1"].tolist()
        heros2_2 = dataset["loss_member2"].tolist()
        heros2_3 = dataset["loss_member3"].tolist()
        heros2_4 = dataset["loss_member4"].tolist()
        heros2_5 = dataset["loss_member5"].tolist()

        for i_index in range(len(heros1_1)):
            player1_combo = sorted([heros1_1[i_index], heros1_2[i_index], heros1_3[i_index], heros1_4[i_index], heros1_5[i_index]])
            player2_combo = sorted([heros2_1[i_index], heros2_2[i_index], heros2_3[i_index], heros2_4[i_index], heros2_5[i_index]])
            
            self._combo1_buffer.append(player1_combo)
            self._combo2_buffer.append(player2_combo)
            self._result_buffer.append(1)

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
                "player1_raw_combos": [",".join([str(c) for c in self._combo1_buffer[j]]) for j in range(i, i + batch_size, 1)],
                "player2_raw_combos": [",".join([str(c) for c in self._combo2_buffer[j]]) for j in range(i, i + batch_size, 1)],
            }

if __name__ == '__main__':
    replay = LoLMatchResultBuffer({
        "dataset_path" : "~/balance/LoL/league_of_legends.csv",
    })
    combo_set = set()
    for combos_dict in replay.sample_all_combos(1024):
        for c in combos_dict["player1_raw_combos"]:
            combo_set.add(c)
        for c in combos_dict["player2_raw_combos"]:
            combo_set.add(c)
    print(f"Total {len(combo_set)} comps")
    
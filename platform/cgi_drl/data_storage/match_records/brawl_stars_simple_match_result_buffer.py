import numpy as np
import pandas as pd

class BrawlStarsMatchResultBuffer(object):
    def __init__(self, config):
        self.dataset_path = config["dataset_path"]

        self._combo1_buffer = []
        self._combo2_buffer = []
        self._result_buffer = []

        dataset = pd.read_csv(self.dataset_path)

        heros1_1 = dataset["teamA_member1"].tolist()
        heros1_2 = dataset["teamA_member2"].tolist()
        heros1_3 = dataset["teamA_member3"].tolist()

        heros2_1 = dataset["teamB_member1"].tolist()
        heros2_2 = dataset["teamB_member2"].tolist()
        heros2_3 = dataset["teamB_member3"].tolist()

        win_teams = dataset["winning_team"].tolist()

        for i_index in range(len(heros1_1)):
            
            _player1_combo = sorted([heros1_1[i_index], heros1_2[i_index], heros1_3[i_index]])
            _player2_combo = sorted([heros2_1[i_index], heros2_2[i_index], heros2_3[i_index]])

            for i_player1_subset in range(3):
                if i_player1_subset == 0:
                    player1_combo = [_player1_combo[0], _player1_combo[1]]
                elif i_player1_subset == 1:
                    player1_combo = [_player1_combo[0], _player1_combo[2]]
                else:
                    player1_combo = [_player1_combo[1], _player1_combo[2]]
                for i_player2_subset in range(3):
                    if i_player2_subset == 0:
                        player2_combo = [_player2_combo[0], _player2_combo[1]]
                    elif i_player2_subset == 1:
                        player2_combo = [_player2_combo[0], _player2_combo[2]]
                    else:
                        player2_combo = [_player2_combo[1], _player2_combo[2]]
            
                    self.encode_combo(player1_combo)
                    self.encode_combo(player2_combo)
                    self._combo1_buffer.append(player1_combo)
                    self._combo2_buffer.append(player2_combo)
                    if win_teams[i_index] == "teamA":
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
        result = np.zeros(64)
        for c in combo:
            result[c] += 1
        return result

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
                "player1_combos": np.asarray(player1_combos, dtype=np.float32),
                "player2_combos": np.asarray(player2_combos, dtype=np.float32),
                "match_results": np.array(match_results, dtype=np.float32),
            }
            

    def sample_all_batch(self, batch_size):
        for i in range(0, self.sample_size, batch_size):
            if i + batch_size >= self.sample_size:
                batch_size = self.sample_size - i
            yield {
                "player1_combos": np.asarray([self.encode_combo(self._combo1_buffer[j]) for j in range(i, i + batch_size, 1)], dtype=np.float32),
                "player2_combos": np.asarray([self.encode_combo(self._combo2_buffer[j]) for j in range(i, i + batch_size, 1)], dtype=np.float32),
                "match_results": np.array([self._average_result_buffer[j] for j in range(i, i + batch_size, 1)], dtype=np.float32),
                "player1_raw_combos": [",".join([str(c) for c in self._combo1_buffer[j]]) for j in range(i, i + batch_size, 1)],
                "player2_raw_combos": [",".join([str(c) for c in self._combo2_buffer[j]]) for j in range(i, i + batch_size, 1)],
            }

    def sample_all_combos(self, batch_size):
        for i in range(0, self.sample_size, batch_size):
            if i + batch_size >= self.sample_size:
                batch_size = self.sample_size - i
            yield {
                "player1_combos": np.asarray([self.encode_combo(self._combo1_buffer[j]) for j in range(i, i + batch_size, 1)], dtype=np.float32),
                "player2_combos": np.asarray([self.encode_combo(self._combo2_buffer[j]) for j in range(i, i + batch_size, 1)], dtype=np.float32),
                "player1_raw_combos": [",".join([str(c) for c in self._combo1_buffer[j]]) for j in range(i, i + batch_size, 1)],
                "player2_raw_combos": [",".join([str(c) for c in self._combo2_buffer[j]]) for j in range(i, i + batch_size, 1)],
            }

if __name__ == '__main__':
    replay = BrawlStarsMatchResultBuffer({
        "dataset_path" : "~/balance/BrawlStars/brawl_stars_2023.csv",
    })
    combo_set = set()
    for combos_dict in replay.sample_all_combos(1024):
        for c in combos_dict["player1_raw_combos"]:
            combo_set.add(c)
        for c in combos_dict["player2_raw_combos"]:
            combo_set.add(c)
    print(f"Total {len(combo_set)} comps")
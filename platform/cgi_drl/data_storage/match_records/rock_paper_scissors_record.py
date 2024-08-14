import numpy as np
import pandas as pd

class RockPaperScissorsRecordGenerator(object):
    def generate_match(self):
        combo1 = np.random.choice(3)
        combo2 = np.random.choice(3)
        
        # Determine the match result
        if combo1 == combo2:
            match_result = 0.5  # Tie
        elif combo1 == 0 and combo2 == 2:
            match_result = 1  # Player 1 wins (Rock vs Scissors)
        elif combo1 == 2 and combo2 == 1:
            match_result = 1  # Player 1 wins (Scissors vs Paper)
        elif combo1 == 1 and combo2 == 0:
            match_result = 1  # Player 1 wins (Paper vs Rock)
        else:
            match_result = 0  # Player 2 wins

        return combo1, combo2, match_result

import unittest

class TestRockPaperScissorsRecordGenerator(unittest.TestCase):

    def test_sample_mini_batch(self):
        generator = RockPaperScissorsRecordGenerator()
        batch_size = 10
        result = generator.sample_mini_batch(batch_size)

        # Check if the output structure is correct
        self.assertIn('player1_combos', result)
        self.assertIn('player2_combos', result)
        self.assertIn('match_results', result)

        # Check if the output types are correct
        self.assertIsInstance(result['player1_combos'], np.ndarray)
        self.assertIsInstance(result['player2_combos'], np.ndarray)
        self.assertIsInstance(result['match_results'], np.ndarray)

        # Check if the batch sizes are correct
        self.assertEqual(result['player1_combos'].shape[0], batch_size)
        self.assertEqual(result['player2_combos'].shape[0], batch_size)
        self.assertEqual(result['match_results'].shape[0], batch_size)

        # Check if the values in the arrays are valid
        for i in range(batch_size):
            self.assertTrue(np.array_equal(result['player1_combos'][i], [1, 0, 0]) or
                            np.array_equal(result['player1_combos'][i], [0, 1, 0]) or
                            np.array_equal(result['player1_combos'][i], [0, 0, 1]))
            self.assertTrue(np.array_equal(result['player2_combos'][i], [1, 0, 0]) or
                            np.array_equal(result['player2_combos'][i], [0, 1, 0]) or
                            np.array_equal(result['player2_combos'][i], [0, 0, 1]))
            self.assertIn(result['match_results'][i], [0, 0.5, 1])
            print(result['player1_combos'][i], result['player2_combos'][i], result['match_results'][i])

class RockPaperScissorsMatchResultBuffer(object):
    def __init__(self, config):
        self.element_count = 3
        self.dataset_path = config["dataset_path"]

        self._combo1_buffer = []
        self._combo2_buffer = []
        self._result_buffer = []

        dataset = pd.read_csv(self.dataset_path)

        combo1 = dataset["combo1"].tolist()
        combo2 = dataset["combo2"].tolist()
        win_teams = dataset["result"].tolist()

        for i_index in range(len(combo1)):
            
            player1_combo = combo1[i_index]
            player2_combo = combo2[i_index]
            
            self._combo1_buffer.append(player1_combo)
            self._combo2_buffer.append(player2_combo)
            self._result_buffer.append(win_teams[i_index])

        self.sample_size = len(self._result_buffer)

    def encode_combo(self, combo):
        result = np.zeros(self.element_count)
        result[combo] = 1
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
                "match_results": np.array([self._result_buffer[j] for j in range(i, i + batch_size, 1)], dtype=np.float32),
            }

    def sample_all_combos(self, batch_size):
        for i in range(0, self.sample_size, batch_size):
            if i + batch_size >= self.sample_size:
                batch_size = self.sample_size - i
            yield {
                "player1_combos": np.asarray([self.encode_combo(self._combo1_buffer[j]) for j in range(i, i + batch_size, 1)], dtype=np.float32),
                "player2_combos": np.asarray([self.encode_combo(self._combo2_buffer[j]) for j in range(i, i + batch_size, 1)], dtype=np.float32),
                "player1_raw_combos": [self._combo1_buffer[j] for j in range(i, i + batch_size, 1)],
                "player2_raw_combos": [self._combo2_buffer[j] for j in range(i, i + batch_size, 1)],
            }

if __name__ == '__main__':
    # unittest.main()
    g = RockPaperScissorsRecordGenerator()
    record_count = 100000

    import csv
    with open('rock_paper_scissors_records.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['combo1', 'combo2', 'result'])
        for i in range(record_count):
            print("{}/{}".format(i, record_count), end='\r')
            combo1, combo2, match_result = g.generate_match()
            writer.writerow([combo1, combo2, match_result])


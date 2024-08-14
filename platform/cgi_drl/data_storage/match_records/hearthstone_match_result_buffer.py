import numpy as np
import pandas as pd
import json

class HearthstoneMatchResultBuffer(object):
    def __init__(self, config):
        self.dataset_path = config["dataset_path"]
        # 91 decks
        self.deck_set = ['Aggro Demon Hunter', 'Aggro Paladin', 'Arcane Hunter', 'Automaton Priest', 'Big Demon Hunter', 'Big Druid', 'Big Rogue', 'Big Shaman', 'Blood Death Knight', 'Breakfast Hunter', 'Burn Mage', 'Casino Mage', 'Cleave Hunter', 'Combo Rogue', 'Control Mage', 'Control Priest', 'Control Warlock', 'Control Warrior', 'Curse Warlock', 'Dancing Paladin', 'Deathrattle Druid', 'Dragon Druid', 'Drum Druid', 'Earthen Paladin', 'Elemental Mage', 'Elemental Shaman', 'Enrage Warrior', 'Evolve Shaman', 'Face Hunter', 'Frost Aggro Death Knight', 'Frost Death Knight', 'Highlander Blood Death Knight', 'Highlander Demon Hunter', 'Highlander Druid', 'Highlander Hunter', 'Highlander Mage', 'Highlander Paladin', 'Highlander Priest', 'Highlander Rogue', 'Highlander Shaman', 'Highlander Warlock', 'Highlander Warrior', 'Hound Hunter', 'Imp Warlock', 'Lightshow Mage', 'Mech Mage', 'Mech Paladin', 'Mech Rogue', 'Menagerie Warrior', 'Mill Druid', 'Mining Mage', 'Mining Rogue', 'Mining Warlock', 'Mining Warrior', 'Miracle Priest', 'Miracle Rogue', 'Moonbeam Druid', 'Murloc Warlock', 'Naga Demon Hunter', 'Naga Priest', 'Nature Druid', 'Nature Shaman', 'Ogre Priest', 'Ogre Rogue', 'Outcast Demon Hunter', 'Overload Shaman', 'Plague Death Knight', 'Pure Paladin', 'Rainbow Death Knight', 'Rainbow Mage', 'Ramp Druid', 'Relic Demon Hunter', "Rock 'n' Roll Warrior", 'Secret Hunter', 'Secret Mage', 'Secret Rogue', 'Showdown Paladin', 'Silver Hand Paladin', 'Sludge Warlock', 'Spell Demon Hunter', 'Spooky Mage', 'Taunt Warrior', 'Thaddius Druid', 'Thaddius Warlock', 'Totem Shaman', 'Treant Druid', 'Undead Priest', 'Unholy Death Knight', 'Weapon Rogue', 'Wishing Rogue', 'Yogg Shaman']

        self._combo1_buffer = []
        self._combo2_buffer = []
        self._result_buffer = []

        dataset = pd.read_csv(self.dataset_path)

        self._combo1_buffer = dataset["Deck"].tolist()
        self._combo2_buffer = dataset["Opponent Deck"].tolist()
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
        result = np.zeros(91)
        result[self.deck_set.index(combo)] += 1
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
            }

    # def sample_all_combos(self, batch_size):
    #     for i in range(0, self.sample_size, batch_size):
    #         if i + batch_size >= self.sample_size:
    #             batch_size = self.sample_size - i
    #         yield {
    #             "player1_combos": np.asarray([self.encode_combo(self._combo1_buffer[j]) for j in range(i, i + batch_size, 1)], dtype=np.float32),
    #             "player2_combos": np.asarray([self.encode_combo(self._combo2_buffer[j]) for j in range(i, i + batch_size, 1)], dtype=np.float32),
    #             "player1_raw_combos": [self._combo1_buffer[j] for j in range(i, i + batch_size, 1)],
    #             "player2_raw_combos": [self._combo2_buffer[j] for j in range(i, i + batch_size, 1)],
    #         }
    def sample_all_combos(self, batch_size):
        match_buffer = []
        for deck1 in self.deck_set:
            for deck2 in self.deck_set:
                match_buffer.append((deck1, deck2))
        for i in range(0, 91 * 91, batch_size):
            if i + batch_size >= 91 * 91:
                batch_size = 91 * 91 - i
            yield {
                "player1_combos": np.asarray([self.encode_combo(match_buffer[j][0]) for j in range(i, i + batch_size, 1)], dtype=np.float32),
                "player2_combos": np.asarray([self.encode_combo(match_buffer[j][1]) for j in range(i, i + batch_size, 1)], dtype=np.float32),
                "player1_raw_combos": [match_buffer[j][0] for j in range(i, i + batch_size, 1)],
                "player2_raw_combos": [match_buffer[j][1] for j in range(i, i + batch_size, 1)],
            }

if __name__ == '__main__':
    replay = HearthstoneMatchResultBuffer({
        "dataset_path" : "~/balance/Hearthstone/hearthstone_ranking_gold.csv",
    })
    print(replay.size()) # 10154929
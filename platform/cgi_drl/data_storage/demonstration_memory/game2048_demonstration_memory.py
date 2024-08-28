import numpy as np
import json

class Game2048DemonstrationMemory(object):
    def __init__(self, config):
        self.file_paths = config["file_paths"]
        self.datasets = []
        for file_path in self.file_paths:
            dataset = []
            with open(file_path, 'r', encoding='utf-8') as f:
                game2048_data = json.load(f)
                self.datasets.append(game2048_data)


    def collect_episodes(self, dataset_index, episode_index):
        return self.datasets[dataset_index][episode_index]["moves"]

    def encode_state(self, board_string, code_level):
        if code_level == -2:
            return "none"
        elif code_level == -1:
            return board_string
        else:
            return "none"


if __name__ == '__main__':
    replay = Game2048DemonstrationMemory({
        "file_paths" : ["/root/playstyle_similarity_tmlr/game2048/model_1m.json"],
    })
    print(replay.collect_episodes(0, 1))
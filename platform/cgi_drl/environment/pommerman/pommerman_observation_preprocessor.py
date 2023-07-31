import numpy as np
from cgi_drl.environment.observation_preprocessor import ObservationPreprocessor
import copy
import pommerman

class PommermanObservationPreprocessor(ObservationPreprocessor):
    def __init__(self, config):
        self.environment_id = config["environment_id"]
        self.frame_stack_size = 3
        self.observation_aggregator = config["observation_aggregator"]
        if self.environment_id.startswith("One"):
            self.observation_space = {
                "observation_visual": (8, 8, 12 * self.frame_stack_size),
                "observation_vector": (3 * self.frame_stack_size,),
            }
        elif self.environment_id.startswith("PommeFFA"):
            self.observation_space = {
                "observation_visual": (11, 11, 12 * self.frame_stack_size),
                "observation_vector": (3 * self.frame_stack_size,),
            }
        elif self.environment_id.startswith("PommeTeam"):
            self.observation_space = {
                "observation_visual": (11, 11, 13 * self.frame_stack_size),
                "observation_vector": (3 * self.frame_stack_size,),
            }
        else:
            raise NotImplementedError("Unknown observation_space.")

    def process(self, observations, process_settings = None):    
        processed_observations = {}  
        if type(observations) is dict:
            features_2d, features_1d = self.featurize(observations)
            processed_observations["observation_visual"] = np.asarray(features_2d, dtype=np.float32)
            processed_observations["observation_vector"] = np.asarray(features_1d, dtype=np.float32)
        else:
            observation_visual_list = []
            observation_vector_list = []
            for o in observations:
                features_2d, features_1d = self.featurize(o)
                observation_visual_list.append(features_2d)
                observation_vector_list.append(features_1d)
            processed_observations["observation_visual"] = np.asarray(observation_visual_list, dtype=np.float32)
            processed_observations["observation_vector"] = np.asarray(observation_vector_list, dtype=np.float32)
        return processed_observations

    def featurize(self, observation):
        board = observation["board"]
        features_2d = []
        # Binary features
        board_items = [
            pommerman.constants.Item.Passage,
            pommerman.constants.Item.Wood,
            pommerman.constants.Item.Rigid,
            pommerman.constants.Item.ExtraBomb,
            pommerman.constants.Item.IncrRange,
            pommerman.constants.Item.Kick,
        ]

        for item in board_items:
            features_2d.append(board == item.value)

        # Set walkable feature plan for extrabomb, incrange, kick and bomb if can kick
        for i in range(board.shape[0]):
            for j in range(board.shape[0]):
                if board[i, j] in [
                    pommerman.constants.Item.ExtraBomb.value,
                    pommerman.constants.Item.IncrRange.value,
                    pommerman.constants.Item.Kick.value,
                ]:
                    features_2d[0][i, j] = 1

        position = np.zeros(board.shape)
        position[observation["position"]] = 1
        features_2d.append(position)

        # ! For teammate
        if self.environment_id.startswith("PommeTeam"):
            features_2d.append(
                board == observation["teammate"].value
                if observation["teammate"].value in observation["alive"]
                else np.zeros_like(board)
            )

        enemies = np.zeros(board.shape)
        for enemy in observation["enemies"]:
            enemies[(board == enemy.value)] = 1
        features_2d.append(enemies)

        features_2d.append(observation["bomb_moving_direction"] / 4.0)

        # Normal features_2d
        for feature, max_value in zip(
            ["bomb_life", "bomb_blast_strength", "flame_life"], [9, 20, 3]
        ):
            features_2d.append(observation[feature] / max_value)

        # ! Optional choice that concatenate scalar features
        features_1d = np.array([
            observation["ammo"] / 20, 
            observation["blast_strength"] / 20, 
            1 if observation["can_kick"] else 0,
            observation["time_step"] / 400],
            dtype=np.float,
        )
        features_2d = np.array(features_2d, dtype=np.float)
        # (C, H, W) => (H, W, C)
        features_2d = np.moveaxis(features_2d, 0, -1)

        return features_2d, features_1d

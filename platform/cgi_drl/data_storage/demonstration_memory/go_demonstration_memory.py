import numpy as np
import base64
import copy
from io import StringIO
import queue

class GoDemonstrationMemory(object):
    def __init__(self, config, encoder_function):
        self.replay_file_path = config["replay_file_path"]
        self.pre_build_record_units = [1, 5, 10, 25, 50, 75, 100]
        self.player_records = {}
        self.pre_build_player_records = {}
        self.pre_build_style_infos = {}
        self.first_n_moves = 1000
        self.code_levels = [0, 1, 2]

        inference_queue = []
        state_queues = []
        for i_level in range(len(self.code_levels)):
            state_queues.append([])

        batch_size= 1024

        with open(self.replay_file_path, "r") as replay_file:
            sample_count = 0
            inference_count = 0
            player_set = set()

            while True:
                position_info_string = replay_file.readline()
                if len(position_info_string) == 0:
                    break
                
                position_infos = position_info_string.split()
                player_name = position_infos[0]

                if int(position_infos[2]) > self.first_n_moves:
                    continue

                if int(position_infos[2]) < 2:
                    previous_boards = np.zeros([18, 19, 19])

                previous_boards[0,:,:] = np.array([int(bit) for bit in position_infos[3]], dtype=np.float32).reshape((19, 19))
                previous_boards[1,:,:] = np.array([int(bit) for bit in position_infos[4]], dtype=np.float32).reshape((19, 19))
                previous_boards[2,:,:] = np.array([int(bit) for bit in position_infos[5]], dtype=np.float32).reshape((19, 19))
                previous_boards[3,:,:] = np.array([int(bit) for bit in position_infos[6]], dtype=np.float32).reshape((19, 19))

                if int(position_infos[7]) == 1: # black
                    previous_boards[16,:,:] = np.ones([19,19], dtype=np.float32)
                    previous_boards[17,:,:] = np.zeros([19,19], dtype=np.float32)
                else: # white
                    previous_boards[16,:,:] = np.zeros([19,19], dtype=np.float32)
                    previous_boards[17,:,:] = np.ones([19,19], dtype=np.float32)
                inference_queue.append(np.array(previous_boards[:,:,:]))
                sample_count += 1
                
                if sample_count >= batch_size:
                    states = encoder_function(np.asarray(inference_queue))
                    for i_level in range(len(self.code_levels)):
                        if self.code_levels[i_level] == -1:
                            state_queues[i_level] += ["x"] * len(inference_queue)
                        else:
                            state_queues[i_level] += states[self.code_levels[i_level]]
                    inference_queue = []
                    inference_count += sample_count
                    sample_count = 0
                    print("Inference count: {}".format(inference_count), end='\r')
                    
                previous_boards[[4,5,6,7,8,9,10,11,12,13,14,15],:,:] = previous_boards[[0,1,2,3,4,5,6,7,8,9,10,11],:,:]
            if sample_count > 0:
                states = encoder_function(np.asarray(inference_queue))
                for i_level in range(len(self.code_levels)):
                    if self.code_levels[i_level] == -1:
                        state_queues[i_level] += ["x"] * len(inference_queue)
                    else:
                        state_queues[i_level] += states[self.code_levels[i_level]]
                inference_queue = []
                inference_count += sample_count
                sample_count = 0
                print("Inference count: {}".format(inference_count))

        state_queue_index = 0
        with open(self.replay_file_path, "r") as replay_file:
            record_count = 0
            state_sets = []
            for i_level in range(len(self.code_levels)):
                state_sets.append(set())
            action_set = set()
            player_set = set()
            player_state_set = {}
            while True:
                position_info_string = replay_file.readline()
                if len(position_info_string) == 0:
                    break
                position_infos = position_info_string.split()

                player_name = position_infos[0]
                
                if player_name not in self.player_records:
                    self.player_records[player_name] = {}
                    player_state_set[player_name] = set()
                    
                game_index = int(position_infos[1])

                if int(position_infos[2]) > self.first_n_moves:
                    continue

                if game_index not in self.player_records[player_name]:
                    self.player_records[player_name][game_index] = []

                hierarcy_state = []
                for i_level in range(len(self.code_levels)):
                    state = state_queues[i_level][state_queue_index]
                    state_sets[i_level].add(state)
                    hierarcy_state.append(state)
                    player_state_set[player_name].add(state)
                action = int(position_infos[8])

                action_set.add(action)
                self.player_records[player_name][game_index].append((hierarcy_state, action))
                state_queue_index += 1
                record_count += 1
                log_string = "Currently {} moves, player count: {}, state space: ".format(record_count, len(self.player_records))
                for i_level in range(len(self.code_levels)):
                    log_string += str(len(state_sets[i_level])) + " "
                log_string += ", action space: {}".format(len(action_set))
                print(log_string, end='\r')
        print()
        print("Totaly {} moves".format(record_count))

        for player_name in self.get_player_names():
            records = []
            for i_game in range(100):
                records.extend(self.player_records[player_name][i_game])
                if i_game + 1 in self.pre_build_record_units:
                    if player_name not in self.pre_build_player_records:
                        self.pre_build_player_records[player_name] = {}
                    self.pre_build_player_records[player_name][i_game + 1] = copy.deepcopy(records)

    def get_player_names(self):
        return self.player_records.keys()

    def get_records(self, player_name, game_count):
        return self.pre_build_player_records[player_name][game_count]


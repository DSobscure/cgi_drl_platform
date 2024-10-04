import numpy as np
import gzip
import pickle

class RecurrentExperienceReplay(object):
    '''This object maintains agent's sequence transitions (recurrent specific transition, (observation, action, reward, terminal, next_observation))
        and provide sample mini batch function. The default compression method is gzip.
    '''
    class Path(object):
        '''This object maintains those unfinished trajectories and pack those finished trajectories for the experience replay.
            Each agent has one path for simultaneously maintaining several unfinished trajectories.
        '''
        def __init__(self, sequence_length, adjacent_offset, use_initial_padding):
            self.sequence_length = sequence_length
            self.adjacent_offset = adjacent_offset
            self.use_initial_padding = use_initial_padding
            self.trajectories = [RecurrentExperienceReplay.Trajectory(self.sequence_length, self.adjacent_offset, self.use_initial_padding)]
            
        def __len__(self):
            return len(self.trajectories)

        def append(self, sample):
            self.trajectories[-1].append(sample) 
            if self.trajectories[-1].transitions["done"][-1]:
                self.trajectories.append(RecurrentExperienceReplay.Trajectory(self.sequence_length, self.adjacent_offset, self.use_initial_padding))
            
        def pop_first_trajectory(self):
            return self.trajectories.pop(0)
                
        def get_keys(self):
            return self.trajectories[0].get_keys()

        def get_observation_keys(self):
            return self.trajectories[0].get_observation_keys()

    class Trajectory(object):
        def __init__(self, sequence_length, adjacent_offset, use_initial_padding):
            self.sequence_length = sequence_length
            self.adjacent_offset = adjacent_offset
            self.transitions = {
                "observation": {},
                "action": [],
                "reward": [],
                "done": [],
                "next_observation": {},
            }
            self.use_initial_padding = use_initial_padding

        def __len__(self):
            length = len(self.transitions["action"])
            sequence_count = 0
            if self.use_initial_padding and (length % self.sequence_length) != 0:
                sequence_count += length // self.sequence_length + 1
            else:
                sequence_count += length // self.sequence_length
            remained_sequence = self.sequence_length - self.adjacent_offset
            while remained_sequence > 0:
                length -= self.adjacent_offset
                sequence_count += length // self.sequence_length
                remained_sequence -= self.adjacent_offset
            return sequence_count

        def append(self, sample):
            transition = {
                "observation" : sample[0],
                "action" : sample[1], 
                "reward" : sample[2], 
                "done" : sample[3], 
                "next_observation" : sample[4],
            }
            for key in transition:
                if key == "observation" or key == "next_observation":
                    for obs_key in transition[key]:
                        if obs_key not in self.transitions[key]:
                            self.transitions[key][obs_key] = []
                        self.transitions[key][obs_key].append(transition[key][obs_key])
                else:
                    self.transitions[key].append(transition[key])

        def get_keys(self):
            return ["observation", "action", "reward", "done", "next_observation"]

        def get_observation_keys(self):
            return self.transitions["observation"].keys()

        def merge(self, key):
            trajectory_length = len(self.transitions["action"])
            cut_off_length = trajectory_length % self.sequence_length
            
            if self.use_initial_padding and cut_off_length != 0:
                merged_results = [s for s in self.transitions[key]]
                merged_results = [merged_results[0]] * (self.sequence_length - cut_off_length) + merged_results
            else:
                merged_results = [s for s in self.transitions[key]][cut_off_length:]
                
            sequences = []
            
            length = len(self.transitions["action"])
            if self.use_initial_padding and (length % self.sequence_length) != 0:
                sequence_count = length // self.sequence_length + 1
            else:
                sequence_count = length // self.sequence_length
            for i_sequence in range(sequence_count):
                sequences.append(merged_results[i_sequence * self.sequence_length : i_sequence * self.sequence_length + self.sequence_length])
                
            remained_sequence = self.sequence_length - self.adjacent_offset
            while remained_sequence > 0:
                length -= self.adjacent_offset
                sequence_count = length // self.sequence_length
                for i_sequence in range(sequence_count):
                    sequences.append(merged_results[remained_sequence + i_sequence * self.sequence_length : remained_sequence + i_sequence * self.sequence_length + self.sequence_length])
                remained_sequence -= self.adjacent_offset
        
            return sequences

        def merge_observations(self, key):
            trajectory_length = len(self.transitions["action"])
            cut_off_length = trajectory_length % self.sequence_length

            if self.use_initial_padding and cut_off_length != 0:
                merged_results = [s for s in self.transitions["observation"][key]]
                merged_results = [merged_results[0]] * (self.sequence_length - cut_off_length) + merged_results
            else:
                merged_results = [s for s in self.transitions["observation"][key]][cut_off_length:]
                
            sequences = []
            
            length = len(self.transitions["action"])
            if self.use_initial_padding and (length % self.sequence_length) != 0:
                sequence_count = length // self.sequence_length + 1
            else:
                sequence_count = length // self.sequence_length
            for i_sequence in range(sequence_count):
                sequences.append(merged_results[i_sequence * self.sequence_length : i_sequence * self.sequence_length + self.sequence_length])
                
            remained_sequence = self.sequence_length - self.adjacent_offset
            while remained_sequence > 0:
                length -= self.adjacent_offset
                sequence_count = length // self.sequence_length
                for i_sequence in range(sequence_count):
                    sequences.append(merged_results[remained_sequence + i_sequence * self.sequence_length : remained_sequence + i_sequence * self.sequence_length + self.sequence_length])
                remained_sequence -= self.adjacent_offset
                
            return sequences

        def merge_next_observations(self, key):
            trajectory_length = len(self.transitions["action"])
            cut_off_length = trajectory_length % self.sequence_length

            if self.use_initial_padding and cut_off_length != 0:
                merged_results = [s for s in self.transitions["next_observation"][key]]
                merged_results = [merged_results[0]] * (self.sequence_length - cut_off_length) + merged_results
            else:
                merged_results = [s for s in self.transitions["next_observation"][key]][cut_off_length:]
                
            sequences = []
            
            length = len(self.transitions["action"])
            if self.use_initial_padding and (length % self.sequence_length) != 0:
                sequence_count = length // self.sequence_length + 1
            else:
                sequence_count = length // self.sequence_length
            for i_sequence in range(sequence_count):
                sequences.append(merged_results[i_sequence * self.sequence_length : i_sequence * self.sequence_length + self.sequence_length])
                
            remained_sequence = self.sequence_length - self.adjacent_offset
            while remained_sequence > 0:
                length -= self.adjacent_offset
                sequence_count = length // self.sequence_length
                for i_sequence in range(sequence_count):
                    sequences.append(merged_results[remained_sequence + i_sequence * self.sequence_length : remained_sequence + i_sequence * self.sequence_length + self.sequence_length])
                remained_sequence -= self.adjacent_offset
                
            return sequences
        
    def __init__(self, config):
        self.sequence_length = config["sequence_length"]
        self.max_capacity = config['max_capacity'] // self.sequence_length
        self.use_compression = config.get("use_compression", True)
        self.use_initial_padding = config.get("use_initial_padding", True)
        self.adjacent_offset = config.get("adjacent_offset", self.sequence_length // 2)
        if self.adjacent_offset == 0:
            self.adjacent_offset = self.sequence_length
        self.paths = []
        for _ in range(config["agent_count"]):
            self.paths.append(RecurrentExperienceReplay.Path(self.sequence_length, self.adjacent_offset, self.use_initial_padding))
        self._size = 0
        self._ring_buffer_head = 0
        self._buffer = [None for _ in range(self.max_capacity)]
    
    def __len__(self):
        return self._size * self.sequence_length

    def size(self):
        '''return the size of current buffer'''
        return self._size * self.sequence_length

    def clear(self):
        '''clear all saved transitions'''
        self.paths = []
        for _ in range(self.config["agent_count"]):
            self.paths.append(RecurrentExperienceReplay.Path(self.sequence_length, self.adjacent_offset, self.use_initial_padding))
        self._size = 0
        self._ring_buffer_head = 0
        self._buffer = [None for _ in range(self.max_capacity)]

    def append(self, index, sample):
        self.paths[index].append(sample)
        if len(self.paths[index]) > 1: # A new finished trajectory.
            trajectory = self.paths[index].pop_first_trajectory()
            
            sequences = {}
            for key in trajectory.get_keys():
                if key == "observation":
                    observation_sequences = {}
                    for obs_key in trajectory.get_observation_keys():
                        observation_sequences[obs_key] = trajectory.merge_observations(obs_key)
                    sequences[key] = observation_sequences
                elif key == "next_observation":
                    next_observation_sequences = {}
                    for obs_key in trajectory.get_observation_keys():
                        next_observation_sequences[obs_key] = trajectory.merge_next_observations(obs_key)
                    sequences[key] = next_observation_sequences
                else:
                    sequences[key] = trajectory.merge(key)
            
            for i_sequence in range(len(trajectory)):
                sequence = []
                for key in trajectory.get_keys():
                    if key == "observation":
                        observation_sequence = {}
                        for obs_key in trajectory.get_observation_keys():
                            observation_sequence[obs_key] = sequences[key][obs_key][i_sequence]
                        sequence.append(observation_sequence)
                    elif key == "next_observation":
                        next_observation_sequence = {}
                        for obs_key in trajectory.get_observation_keys():
                            next_observation_sequence[obs_key] = sequences[key][obs_key][i_sequence]
                        sequence.append(next_observation_sequence)
                    else:
                        sequence.append(sequences[key][i_sequence])
                sequence = tuple(sequence)
                if self.use_compression:
                    self._buffer[self._ring_buffer_head] = gzip.compress(pickle.dumps(sequence))
                else:
                    self._buffer[self._ring_buffer_head] = sequence
                    
                if self._size < self.max_capacity:
                    self._size += 1
                self._ring_buffer_head = (self._ring_buffer_head + 1) % self.max_capacity

    def sample_mini_batch(self, batch_size):
        random_indexes = np.random.choice(self._size, batch_size // self.sequence_length, replace=False)

        if self.use_compression:
            transitions = [pickle.loads(gzip.decompress(self._buffer[index])) for index in random_indexes]
        else:
            transitions = [self._buffer[index] for index in random_indexes]

        batch = []
        for i in range(len(transitions[0])):
            if type(transitions[0][i]) is dict:
                result = {}
                for key in transitions[0][i]:
                    result[key] = np.concatenate([t[i][key] for t in transitions])
                batch.append(result)
            else:
                batch.append(np.concatenate([t[i] for t in transitions]))

        return tuple(batch)

if __name__ == '__main__':
    replay = RecurrentExperienceReplay({
        "max_capacity" : 100,
        "use_compression" : True,
        "sequence_length" : 2,
        "agent_count" : 3,
        "adjacent_offset" : 1,
    })

    for i in range(20):
        replay.append(0, ({"state":str(i)},"action","reward",False,{"state":str(i+1)}))
        replay.append(1, ({"state":str(i)},"action","reward",False,{"state":str(i+1)}))
        replay.append(2, ({"state":str(i)},"action","reward",False,{"state":str(i+1)}))
    replay.append(0, ({"state":str(i)},"action","reward",True,{"state":str(i+1)}))
    replay.append(1, ({"state":str(i)},"action","reward",True,{"state":str(i+1)}))
    replay.append(2, ({"state":str(i)},"action","reward",True,{"state":str(i+1)}))
    replay.append(0, ({"state":str(i)},"action","reward",False,{"state":str(i+1)}))
    replay.append(1, ({"state":str(i)},"action","reward",False,{"state":str(i+1)}))
    replay.append(2, ({"state":str(i)},"action","reward",False,{"state":str(i+1)}))

    state_batch, action_batch, reward_batch, done_batch, next_state_batch = replay.sample_mini_batch(4)
    print(state_batch, next_state_batch)
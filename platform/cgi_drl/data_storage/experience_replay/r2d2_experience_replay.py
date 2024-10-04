import numpy as np
import gzip
import pickle

class R2d2ExperienceReplay(object):
    '''This object maintains agent's sequence transitions (R2D2 specific transition, (observation, action, reward, terminal, next_observation))
        and provide sample mini batch function. The default compression method is gzip.
    '''
    class Path(object):
        '''This object maintains those unfinished trajectories and pack those finished trajectories for the prioritized experience replay.
            Each agent has one path for simultaneously maintaining several unfinished trajectories.
        '''
        def __init__(self, sequence_length, adjacent_offset, use_initial_padding):
            self.sequence_length = sequence_length
            self.adjacent_offset = adjacent_offset
            self.use_initial_padding = use_initial_padding
            self.trajectories = [R2d2ExperienceReplay.Trajectory(self.sequence_length, self.adjacent_offset, self.use_initial_padding)]
            
        def __len__(self):
            return len(self.trajectories)

        def append(self, sample):
            self.trajectories[-1].append(sample) 
            if self.trajectories[-1].transitions["done"][-1]:
                self.trajectories.append(R2d2ExperienceReplay.Trajectory(self.sequence_length, self.adjacent_offset, self.use_initial_padding))
            
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
        self.priority_alpha = config['priority_alpha']
        self.initial_priority = config.get("initial_priority", 100)
        self.priority_mixture_eta = config.get("priority_mixture_eta", 0.9)
        self.edge_case_value_epsilon = config.get("edge_case_value_epsilon", 1e-6)
        self.use_initial_padding = config.get("use_initial_padding", True)
        self.adjacent_offset = config.get("adjacent_offset", self.sequence_length // 2)
        if self.adjacent_offset == 0:
            self.adjacent_offset = self.sequence_length
        self.paths = []
        for _ in range(config["agent_count"]):
            self.paths.append(R2d2ExperienceReplay.Path(self.sequence_length, self.adjacent_offset, self.use_initial_padding))
        self._size = 0
        self._ring_buffer_head = 0
        self._buffer = [None for _ in range(self.max_capacity)]
        self._sum_tree = R2d2ExperienceReplay.SumTree(self.max_capacity, self.edge_case_value_epsilon)
    
    def __len__(self):
        return self._size * self.sequence_length

    def size(self):
        '''return the size of current buffer'''
        return self._size * self.sequence_length

    def clear(self):
        '''clear all saved transitions'''
        self.paths = []
        for _ in range(self.config["agent_count"]):
            self.paths.append(R2d2ExperienceReplay.Path(self.sequence_length, self.adjacent_offset, self.use_initial_padding))
        self._size = 0
        self._ring_buffer_head = 0
        self._buffer = [None for _ in range(self.max_capacity)]
        self._sum_tree = R2d2ExperienceReplay.SumTree(self.max_capacity, self.edge_case_value_epsilon)

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
                    
                self._sum_tree.add(self.initial_priority ** self.priority_alpha)
                if self._size < self.max_capacity:
                    self._size += 1
                self._ring_buffer_head = (self._ring_buffer_head + 1) % self.max_capacity

    def sample_mini_batch(self, batch_size, importance_sampling_beta):
        random_indexes = []
        importance_sampling_weights = np.zeros(batch_size // self.sequence_length)
        p = np.random.uniform(0, 1, batch_size // self.sequence_length)
        for i in range(batch_size // self.sequence_length):
            index, probability = self._sum_tree.access_leaf_index(p[i])
            if index == -1:
                index = self._size - 1
            random_indexes.append(index)
            importance_sampling_weights[i] = (self.max_capacity * probability) ** -importance_sampling_beta
        importance_sampling_weights /= (np.max(importance_sampling_weights)+1e-8)

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
        batch.append(random_indexes)
        batch.append(importance_sampling_weights)

        return tuple(batch)
    
    def update_batch(self, batch_indexes, priority_values):
        priority_values = np.reshape(priority_values, [-1, self.sequence_length])
        for i in range(len(batch_indexes)):
            p = self.priority_mixture_eta * np.max(priority_values[i]) + (1 - self.priority_mixture_eta) * np.mean(priority_values[i])
            self._sum_tree.update(batch_indexes[i], p ** self.priority_alpha)
    
    class SumTree(object):
        def __init__(self, max_capacity, edge_case_value_epsilon):
            self.max_capacity = max_capacity
            self.edge_case_value_epsilon = edge_case_value_epsilon
            self.node_offset = int(2 ** np.ceil(np.log2(self.max_capacity)) - 1)
            self.heap_size = self.node_offset + self.max_capacity
            self.binary_heap = np.zeros(self.heap_size)
            self.insertion_offset = 0

        def add(self, value):
            self.update(self.insertion_offset, value)
            self.insertion_offset = (self.insertion_offset + 1) % self.max_capacity

        def update(self, leaf_index, value):
            value += self.edge_case_value_epsilon
            binary_heap_index = self.node_offset + leaf_index
            value_delta = value - self.binary_heap[binary_heap_index]

            while binary_heap_index >= 0:
                self.binary_heap[binary_heap_index] += value_delta
                parent_index = (binary_heap_index - 1) // 2
                binary_heap_index = parent_index

        def access_leaf_index(self, proportional_value):
            value = self.get_total_priority() * proportional_value
            current_index = 0
            while True:
                left_child_index = current_index * 2 + 1
                right_child_index = left_child_index + 1

                if left_child_index > self.heap_size - 1:
                    break
                else:
                    if value <= self.binary_heap[left_child_index]:
                        current_index = left_child_index
                    else:
                        value -= self.binary_heap[left_child_index]
                        current_index = right_child_index
            probability = self.binary_heap[current_index] / self.binary_heap[0]
            return current_index - self.node_offset, probability

        def get_total_priority(self):
            return self.binary_heap[0]

if __name__ == '__main__':
    replay = R2d2ExperienceReplay({
        "max_capacity" : 100,
        "use_compression" : True,
        "sequence_length" : 3,
        "priority_alpha" : 1,
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

    state_batch, action_batch, reward_batch, done_batch, next_state_batch, random_indexes, importance_sampling_weights = replay.sample_mini_batch(4, 1)
    print(state_batch, next_state_batch)
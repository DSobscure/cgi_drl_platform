import numpy as np
import gzip
import pickle

class PrioritizedExperienceReplay(object):
    '''This object maintain agent's transitions (customized transition, like (state, action, reward, terminal, next_state, ...))
        and provide sample mini batch function. The default compression method is gzip.
    '''
    def __init__(self, config):
        self.max_capacity = config['max_capacity']
        self.use_compression = config.get("use_compression", True)
        self.priority_alpha = config['priority_alpha']
        self.initial_priority = config.get("initial_priority", 100)
        self.edge_case_value_epsilon = config.get("edge_case_value_epsilon", 1e-6)
        self._buffer = [None for _ in range(self.max_capacity)]
        self._size = 0
        self._sum_tree = PrioritizedExperienceReplay.SumTree(self.max_capacity, self.edge_case_value_epsilon)
        self._ring_buffer_head = 0
    
    def __len__(self):
        return self._size

    def size(self):
        '''return the size of current buffer'''
        return self._size

    def clear(self):
        '''clear all saved transitions'''
        self._buffer = [None for _ in range(self.max_capacity)]
        self._sum_tree = PrioritizedExperienceReplay.SumTree(self.max_capacity, self.edge_case_value_epsilon)
        self._size = 0
        self._ring_buffer_head = 0

    def append(self, *transition, priority_value=None):
        if priority_value == None:
            priority_value = self.initial_priority

        if self.use_compression:
            self._buffer[self._ring_buffer_head] = gzip.compress(pickle.dumps(transition))
        else:
            self._buffer[self._ring_buffer_head] = transition

        self._sum_tree.add(priority_value ** self.priority_alpha)

        if self._size < self.max_capacity:
            self._size += 1
        self._ring_buffer_head = (self._ring_buffer_head + 1) % self.max_capacity

    def sample_mini_batch(self, batch_size, importance_sampling_beta):
        random_indexes = []
        importance_sampling_weights = np.zeros(batch_size)
        p = np.random.uniform(0, 1, batch_size)
        for i in range(batch_size):
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
                    result[key] = [t[i][key] for t in transitions]
                batch.append(result)
            else:
                batch.append([t[i] for t in transitions])
        batch.append(random_indexes)
        batch.append(importance_sampling_weights)

        return tuple(batch)

    def update_batch(self, batch_indexes, priority_values):
        for i in range(len(batch_indexes)):
            self._sum_tree.update(batch_indexes[i], priority_values[i] ** self.priority_alpha)

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
    mem = PrioritizedExperienceReplay({
        "max_capacity" : 10,
        "priority_alpha" : 1,
    })
    for i in range(20):
        mem.append({"x":[[[i]]]}, 0, 0, False, {"x":[[[i+1]]]}, priority_value=1)
    mem.append({"x":[[[10000000]]]}, 0, 1, True, {"x":[[[10000000]]]}, priority_value=10)


    state_batch, action_batch, reward_batch, done_batch, next_state_batch, random_indexes, importance_sampling_weights = mem.sample_mini_batch(10, 1)
    for i in range(len(action_batch)):
        print("{} {} {}".format(state_batch["x"][i], random_indexes[i], importance_sampling_weights[i]))
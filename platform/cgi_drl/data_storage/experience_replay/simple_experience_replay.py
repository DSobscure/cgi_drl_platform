import numpy as np
import gzip
import pickle

class SimpleExperienceReplay(object):
    '''This object maintain agent's transitions (customized transition, like (state, action, reward, terminal, next_state, ...))
        and provide sample mini batch function. The default compression method is gzip.
    '''
    def __init__(self, config):
        self.max_capacity = config['max_capacity']
        self.use_compression = config.get("use_compression", True)
        self._buffer = [None for _ in range(self.max_capacity)]
        self._size = 0
        self._ring_buffer_head = 0
    
    def __len__(self):
        return self._size

    def size(self):
        '''return the size of current buffer'''
        return self._size

    def clear(self):
        '''clear all saved transitions'''
        self._buffer = [None for _ in range(self.max_capacity)]
        self._size = 0
        self._ring_buffer_head = 0

    def append(self, *transition):
        if self.use_compression:
            self._buffer[self._ring_buffer_head] = gzip.compress(pickle.dumps(transition))
        else:
            self._buffer[self._ring_buffer_head] = transition
        if self._size < self.max_capacity:
            self._size += 1
        self._ring_buffer_head = (self._ring_buffer_head + 1) % self.max_capacity

    def sample_mini_batch(self, batch_size):
        random_indexes = np.random.choice(self._size, batch_size, replace=False)

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

        return tuple(batch)

if __name__ == '__main__':
    replay = SimpleExperienceReplay({
        "max_capacity" : 10,
        "use_compression" : True
    })

    for i in range(20):
        replay.append({"state":str(i)},"action","reward","done","next_state"+str(i+1))

    state_batch, action_batch, reward_batch, done_batch, next_state_batch = replay.sample_mini_batch(4)
    print(state_batch, next_state_batch)
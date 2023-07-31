from collections import deque
import numpy as np
from cgi_drl.data_storage.demonstration_memory.demonstration_tool import load_npz


class AtariDemonstrationMemory(object):
    '''This object maintain demonstration state-action pairs
        and provide sample mini batch function.
    '''

    def __init__(self, config):
        np.random.seed(1234)
        self.npz_folder = config["npz_folder"]
        self.visual_observation_frame_count = config["visual_observation_frame_count"]
        self.action_count = config["action_count"]

        visual_obs, actions, rewards, done = load_npz(self.npz_folder, use_visual_obs=True, nd_array=False)

        self._visual_obs_buffer = deque(visual_obs)
        self._actions_obs_buffer = deque(actions)
        self._done_obs_buffer = deque(done)
        self.sample_size = len(self._visual_obs_buffer)
        self._one_hot_actions_obs_buffer = np.zeros((self.sample_size, self.action_count))
        self._convert_action_to_one_hot()
        

    def size(self):
        '''return the size of inner deque size'''
        return self.sample_size

    def __len__(self):
        return self.sample_size

    def clear(self):
        '''clear all demonstration transitions'''
        self._visual_obs_buffer.clear()
        self._actions_obs_buffer.clear()
        self._done_obs_buffer.clear()
        self.sample_size = len(self._visual_obs_buffer)

    def save_experience(self, state, action, done):
        '''save a state-action pair into memory'''
        self._visual_obs_buffer.append(state)
        self._actions_obs_buffer.append(action)
        self._done_obs_buffer.append(done)
        self.sample_size += 1

    def _sample_observation_with_window(self, index, window_size):
        observation = []
        done = False
        last_non_done_observation = None
        for i in range(window_size):
            if done:
                observation.append(last_non_done_observation)
            else:
                observation.append(self._visual_obs_buffer[index - i])
                last_non_done_observation = self._visual_obs_buffer[index - i]
                if index - i - 1 < 0 or self._done_obs_buffer[index - i - 1]:
                    done = True
        return observation

    def sample_mini_batch(self, batch_size):
        '''uniform random sample state-action pairs from demonstration memory'''

        assert self.sample_size >= batch_size, "buffer size < batch size"
        idx = [np.random.randint(self.sample_size) for i in range(batch_size)]
        visual_obs_batch = [self._sample_observation_with_window(i, self.visual_observation_frame_count) for i in idx] 
        action_batch = [self._one_hot_actions_obs_buffer[i] for i in idx]

        return np.asarray(visual_obs_batch, dtype=np.float32) / 255.0, np.array(action_batch)

    def sample_all_batch(self, batch_size):
        '''Yield all batch with all experience sequentially'''
        visual_obs_batch = [
            np.asarray(self._sample_observation_with_window(i, self.visual_observation_frame_count), dtype=np.float32) / 255.0
            for i in range(self.sample_size)]
        action_batch = [self._one_hot_actions_obs_buffer[i] for i in range(self.sample_size)]
        for i in range(0, self.sample_size, batch_size):
            yield [
                np.array(visual_obs_batch[i:i + batch_size]),
                np.array(action_batch[i:i + batch_size])
            ]

    def _convert_action_to_one_hot(self):
        for i in range(self.sample_size):
            self._one_hot_actions_obs_buffer[i][self._actions_obs_buffer[i]] = 1



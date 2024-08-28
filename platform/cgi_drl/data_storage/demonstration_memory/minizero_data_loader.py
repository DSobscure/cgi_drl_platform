import torch
import numpy as np

class MinizeroDadaLoader(object):
    def __init__(self, config):
        minizero_root_path = config["minizero_root_path"]
        import sys
        sys.path.append(minizero_root_path)
        self._py = __import__('build.go', globals(), locals(), ['minizero_py'], 0).minizero_py
        minizero_config_path = config["minizero_config_path"]

        self.data_loader = self._py.DataLoader(minizero_config_path)
        self.data_loader.initialize()
        self.data_list = []

        # allocate memory
        self.sampled_index = np.zeros(self._py.get_batch_size() * 2, dtype=np.int32)
        self.features = np.zeros(self._py.get_batch_size() * self._py.get_nn_num_input_channels() * self._py.get_nn_input_channel_height() * self._py.get_nn_input_channel_width(), dtype=np.float32)
        self.loss_scale = np.zeros(self._py.get_batch_size(), dtype=np.float32)
        self.value_accumulator = np.ones(1) if self._py.get_nn_discrete_value_size() == 1 else np.arange(-int(self._py.get_nn_discrete_value_size() / 2), int(self._py.get_nn_discrete_value_size() / 2) + 1)
        if self._py.get_nn_type_name() == "alphazero":
            self.action_features = None
            self.policy = np.zeros(self._py.get_batch_size() * self._py.get_nn_action_size(), dtype=np.float32)
            self.value = np.zeros(self._py.get_batch_size() * self._py.get_nn_discrete_value_size(), dtype=np.float32)
            self.reward = None
        else:
            self.action_features = np.zeros(self._py.get_batch_size() * self._py.get_muzero_unrolling_step() * self._py.get_nn_num_action_feature_channels()
                                            * self._py.get_nn_hidden_channel_height() * self._py.get_nn_hidden_channel_width(), dtype=np.float32)
            self.policy = np.zeros(self._py.get_batch_size() * (self._py.get_muzero_unrolling_step() + 1) * self._py.get_nn_action_size(), dtype=np.float32)
            self.value = np.zeros(self._py.get_batch_size() * (self._py.get_muzero_unrolling_step() + 1) * self._py.get_nn_discrete_value_size(), dtype=np.float32)
            self.reward = np.zeros(self._py.get_batch_size() * self._py.get_muzero_unrolling_step() * self._py.get_nn_discrete_value_size(), dtype=np.float32)

        for data_path in config["data_paths"]:
            self.load_data(data_path)

    def load_data(self, file_name):
        self.data_loader.load_data_from_file(file_name)

    def sample_data(self, device='cpu'):
        self.data_loader.sample_data(self.features, self.action_features, self.policy, self.value, self.reward, self.loss_scale, self.sampled_index)
        features = torch.FloatTensor(self.features).view(self._py.get_batch_size(), self._py.get_nn_num_input_channels(), self._py.get_nn_input_channel_height(), self._py.get_nn_input_channel_width()).to(device)
        action_features = None if self.action_features is None else torch.FloatTensor(self.action_features).view(self._py.get_batch_size(),
                                                                                                                 -1,
                                                                                                                 self._py.get_nn_num_action_feature_channels(),
                                                                                                                 self._py.get_nn_hidden_channel_height(),
                                                                                                                 self._py.get_nn_hidden_channel_width()).to(device)
        policy = torch.FloatTensor(self.policy).view(self._py.get_batch_size(), -1, self._py.get_nn_action_size()).to(device)
        value = torch.FloatTensor(self.value).view(self._py.get_batch_size(), -1, self._py.get_nn_discrete_value_size()).to(device)
        reward = None if self.reward is None else torch.FloatTensor(self.reward).view(self._py.get_batch_size(), -1, self._py.get_nn_discrete_value_size()).to(device)
        loss_scale = torch.FloatTensor(self.loss_scale / np.amax(self.loss_scale)).to(device)
        sampled_index = self.sampled_index

        return features, action_features, policy, value, reward, loss_scale, sampled_index

    def update_priority(self, sampled_index, batch_values):
        batch_values = (batch_values * self.value_accumulator).sum(axis=1)
        self.data_loader.update_priority(sampled_index, batch_values)
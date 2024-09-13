import torch
import torch.nn.functional as F
import importlib
import numpy as np

class PolicyTrainer():
    def __init__(self, config):
        self.action_space = config["action_space"]
        if isinstance(self.action_space, int):
            self.action_space = [self.action_space]
        self.value_head_count = config.get("value_head_count", 1)

        self.use_rnn = config.get("use_rnn", False)
        self.invertible_value_function = config.get("invertible_value_function", lambda x : x)

        PolicyModel = getattr(importlib.import_module(config["model_define_path"]), "PolicyModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        network_settings = config.get("network_settings", {})
        network_settings["action_space"] = self.action_space
        network_settings["value_head_count"] = self.value_head_count
        if self.use_rnn:
            network_settings["memory_size"] = config["memory_size"]
            self.memory_size = config["memory_size"]
            self.rnn_sequence_length = config["rnn_sequence_length"]
            self.rnn_burn_in_length = config["rnn_burn_in_length"]
        
        self.behavior_network = PolicyModel(network_settings).to(self.device)
        self.target_network = PolicyModel(network_settings).to(self.device)

        self.observation_prodiver = config["observation_prodiver"]
        self.action_prodiver = config["action_prodiver"]
        self.target_q_values_prodiver = config["target_q_values_prodiver"]
        self.optimizer = torch.optim.Adam(self.behavior_network.parameters())

    def update(self, transitions, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}
        observations = self.observation_prodiver(transitions["observations"], self.device)
        actions = self.action_prodiver(transitions["actions"], self.device)
        q_values = self.target_q_values_prodiver(transitions["q_values"], self.device)
        
        for i in range(len(self.optimizer.param_groups)): 
            self.optimizer.param_groups[i]['lr'] = extra_settings["learning_rate"]

        batch_size = len(actions)
        if "loss_weights" in extra_settings:
            loss_weights = torch.tensor(extra_settings["loss_weights"], dtype=torch.float32).to(self.device).view(-1, 1)
        else:
            loss_weights = torch.tensor(np.ones(batch_size), dtype=torch.float32).to(self.device).view(-1, 1)

        self.behavior_network.train()
        self.optimizer.zero_grad()

        if self.use_rnn:
            network_output = self.behavior_network(observations, self.rnn_sequence_length, self.rnn_burn_in_length)
            bevahior_q_values = network_output["q_value"]
        else:
            bevahior_q_values = self.behavior_network(observations)

        q_losses = []
        q_loss = 0
        for i_head in range(self.value_head_count):
            head_q_loss = 0
            for i_space in range(len(self.action_space)):
                one_hot_action = F.one_hot(actions[:, i_space], num_classes=self.action_space[i_space]).float()
                behavior_q = torch.sum(one_hot_action * bevahior_q_values[i_head][i_space], dim=-1).view(-1, 1)
                head_q_loss += torch.mean(loss_weights * torch.nn.functional.huber_loss(behavior_q, self.invertible_value_function(q_values[:, i_head], is_inverse=False).view(-1, 1), reduction='none'))
            q_losses.append(head_q_loss)
            q_loss += head_q_loss
        q_loss /= self.value_head_count * len(self.action_space)
        
        q_loss.backward()
        self.optimizer.step()
        
        return q_loss, q_losses
    
    def _q_list_to_cpu(self, q_list):
        _q_list = []
        for i_head in range(len(q_list)):
            _q_branch = []
            for i_space in range(len(q_list[i_head])):
                _q_branch.append(self.invertible_value_function(q_list[i_head][i_space], is_inverse=True).to('cpu').detach().numpy())
            _q_list.append(_q_branch)
        return _q_list

    def get_behavior_q_values(self, observations, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}

        self.behavior_network.eval()
        observations = self.observation_prodiver(observations, self.device)
        
        if self.use_rnn:
            network_output = self.behavior_network(observations, 1, 0)
            return self._q_list_to_cpu(network_output["q_value"]), network_output["next_memory"].to('cpu').detach().numpy()
        else:
            q_values = self.behavior_network(observations)
            return self._q_list_to_cpu(q_values)

    def get_target_q_values(self, observations, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}
        
        self.target_network.eval()
        observations = self.observation_prodiver(observations, self.device)

        if self.use_rnn:
            network_output = self.target_network(observations, 1, 0)
            return self._q_list_to_cpu(network_output["q_value"]), network_output["next_memory"].to('cpu').detach().numpy()
        else:
            q_values = self.target_network(observations)
            return self._q_list_to_cpu(q_values)

    def update_target_policy(self):
        print("update_target_policy")
        self.target_network.load_state_dict(self.behavior_network.state_dict())

    def save(self, path, time_step):
        '''save NN model (give a directory name for the model) '''
        torch.save(self.behavior_network.state_dict(), f"{path}/model_{time_step}.ckpt")
        print(f"Model saved in file: {path}/model_{time_step}.ckpt")

    def load(self, path):
        '''load NN model (give a directory name for the model) '''
        self.behavior_network.load_state_dict(torch.load(path))
        print("Model restored.")

    def save_to_agent_pool(self, agent_pool_path, time_step="latest"):
        self.save(agent_pool_path, time_step)
        print(f"Model saved to an agent pool: {agent_pool_path}")
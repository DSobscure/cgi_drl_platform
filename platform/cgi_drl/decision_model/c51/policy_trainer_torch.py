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
        self.value_atom_count = config["value_atom_count"]
        self.value_V_min = config["value_V_min"]
        self.value_V_max = config["value_V_max"]
        self.value_zero_index = config["value_zero_index"]
        self.value_atom_scale_support_delta_z = (self.value_V_max - self.value_V_min) / (self.value_atom_count - 1)
        self.value_atom_scale_supports = self.value_V_min + np.arange(self.value_atom_count) * self.value_atom_scale_support_delta_z
        self.cross_entropy_minimal_epsilon = config.get("cross_entropy_minimal_epsilon", 1e-10)

        self.use_rnn = config.get("use_rnn", False)

        PolicyModel = getattr(importlib.import_module(config["model_define_path"]), "PolicyModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        network_settings = config.get("network_settings", {})
        network_settings["action_space"] = self.action_space
        network_settings["value_head_count"] = self.value_head_count
        network_settings["value_atom_count"] = self.value_atom_count
        network_settings["value_atom_scale_supports"] = torch.tensor(self.value_atom_scale_supports, dtype=torch.float32).to(self.device)
        if self.use_rnn:
            network_settings["memory_size"] = config["memory_size"]
            self.memory_size = config["memory_size"]
            self.rnn_sequence_length = config["rnn_sequence_length"]
            self.rnn_burn_in_length = config["rnn_burn_in_length"]
        
        self.behavior_network = PolicyModel(network_settings).to(self.device)
        self.target_network = PolicyModel(network_settings).to(self.device)

        self.observation_prodiver = config["observation_prodiver"]
        self.action_prodiver = config["action_prodiver"]
        self.target_q_distributions_prodiver = config["target_q_distributions_prodiver"]
        self.optimizer = config.get("optimizer_function", lambda x: torch.optim.Adam(x))(self.behavior_network.parameters())

    def update(self, transitions, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}
        observations = self.observation_prodiver(transitions["observations"], self.device)
        actions = self.action_prodiver(transitions["actions"], self.device)
        q_distributions = self.target_q_distributions_prodiver(transitions["q_distributions"], self.device)
        
        for i in range(len(self.optimizer.param_groups)): 
            self.optimizer.param_groups[i]['lr'] = extra_settings["learning_rate"]

        batch_size = len(actions)
        if "loss_weights" in extra_settings:
            loss_weights = torch.tensor(extra_settings["loss_weights"], dtype=torch.float32).to(self.device).view(-1, 1)
            if self.use_rnn:
                loss_weights = loss_weights.repeat_interleave(self.rnn_sequence_length).view(-1, 1)
        else:
            loss_weights = torch.tensor(np.ones(batch_size), dtype=torch.float32).to(self.device).view(-1, 1) 

        self.behavior_network.train()
        self.optimizer.zero_grad()

        if self.use_rnn:
            behavior_q_distribution_logits = self.behavior_network(observations, self.rnn_sequence_length, self.rnn_burn_in_length)["q_distribution_logit"]
        else:
            behavior_q_distribution_logits = self.behavior_network(observations)["q_distribution_logit"]

        q_losses = 0
        q_loss = 0
        for i_head in range(self.value_head_count):
            for i_space in range(len(self.action_space)):
                one_hot_action = F.one_hot(actions[:, i_space], num_classes=self.action_space[i_space]).float()
                one_hot_action = one_hot_action.unsqueeze(-2)
                behavior_q_distribution_logits = behavior_q_distribution_logits[i_head][i_space].permute(0, 2, 1).contiguous()
                behavior_q_distribution_logits = torch.sum(one_hot_action * behavior_q_distribution_logits, dim=-1)
                target_distribution = q_distributions[:, i_head, :]           
                branch_q_loss = -torch.sum(
                    (target_distribution + self.cross_entropy_minimal_epsilon) *
                    torch.log(F.softmax(behavior_q_distribution_logits, dim=-1) + self.cross_entropy_minimal_epsilon),
                    dim=-1
                ).view(-1, 1)            
                q_losses += branch_q_loss
                q_loss += torch.mean(loss_weights * branch_q_loss)
        q_losses /= self.value_head_count * len(self.action_space)
        q_loss /= self.value_head_count * len(self.action_space)
        
        q_loss.backward()
        self.optimizer.step()
        
        return q_loss, q_losses.squeeze(-1).to('cpu').detach().numpy()
    
    def _q_list_to_cpu(self, q_list):
        _q_list = []
        for i_head in range(len(q_list)):
            _q_branch = []
            for i_space in range(len(q_list[i_head])):
                _q_branch.append(q_list[i_head][i_space].to('cpu').detach().numpy())
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
            q_values = self.behavior_network(observations)["q_value"]
            return self._q_list_to_cpu(q_values)
        
    def get_behavior_q_distributions(self, observations, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}
            
        self.behavior_network.eval()
        observations = self.observation_prodiver(observations, self.device)
        
        if self.use_rnn:
            network_output = self.behavior_network(observations, 1, 0)
            return self._q_list_to_cpu(network_output["q_distribution"]), network_output["next_memory"].to('cpu').detach().numpy()
        else:
            q_distributions = self.behavior_network(observations)["q_distribution"]
            return self._q_list_to_cpu(q_distributions)

    def get_target_q_values_and_distributions(self, observations, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}
        
        self.target_network.eval()
        observations = self.observation_prodiver(observations, self.device)

        if self.use_rnn:
            network_output = self.target_network(observations, 1, 0)
            return self._q_list_to_cpu(network_output["q_value"]), self._q_list_to_cpu(network_output["q_distribution"]), network_output["next_memory"].to('cpu').detach().numpy()
        else:
            network_output = self.target_network(observations)
            return self._q_list_to_cpu(network_output["q_value"]), self._q_list_to_cpu(network_output["q_distribution"])

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
    
    def distributional_bellman_operator(self, next_q_distribution, gamma, reward):
        T_z = np.clip(reward[:, None] + gamma * self.value_atom_scale_supports, self.value_V_min, self.value_V_max)
        b = (T_z - self.value_V_min) / self.value_atom_scale_support_delta_z
        l = np.floor(b).astype(np.int32)
        u = np.ceil(b).astype(np.int32)
        l_int = np.clip(l, 0, self.value_atom_count - 1)
        u_int = np.clip(u, 0, self.value_atom_count - 1)
        
        mask = (l == u).astype(np.float32)
        l_weight = (u - b) * (1 - mask) + mask
        u_weight = (b - l) * (1 - mask) 

        l_distribution = next_q_distribution * l_weight
        u_distribution = next_q_distribution * u_weight

        final_q_distribution = np.zeros_like(next_q_distribution, dtype=np.float32)
        
        batch_size, atom_count = next_q_distribution.shape
        batch_indices = np.repeat(np.arange(batch_size), atom_count)
        
        l_int_flat = l_int.flatten()
        u_int_flat = u_int.flatten()
        l_distribution_flat = l_distribution.flatten()
        u_distribution_flat = u_distribution.flatten()
        
        np.add.at(final_q_distribution, (batch_indices, l_int_flat), l_distribution_flat)
        np.add.at(final_q_distribution, (batch_indices, u_int_flat), u_distribution_flat)

        return final_q_distribution
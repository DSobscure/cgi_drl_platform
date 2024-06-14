import torch
import importlib
import numpy as np
   
class PolicyTrainer():
    def __init__(self, config):
        self.action_space = config["action_space"]
        if isinstance(self.action_space, int):
            self.action_space = [self.action_space]
        self.value_head_count = config.get("value_head_count", 1)

        self.use_rnn = config.get("use_rnn", False)
        self.max_gradient_norm = config.get("max_gradient_norm", 10)
        self.invertible_value_functions = config.get("invertible_value_functions", lambda x : x)

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
        
        self.policy_model = PolicyModel(network_settings).to(self.device)
        self.old_policy_model = PolicyModel(network_settings).to(self.device)

        self.observation_prodiver = config["observation_prodiver"]
        self.action_prodiver = config["action_prodiver"]
        self.return_prodiver = config["return_prodiver"]
        self.advantage_prodiver = config["advantage_prodiver"]

        self.optimizer = torch.optim.Adam(self.policy_model.parameters())

    def update(self, transitions, extra_settings = None):       
        if extra_settings == None:
            extra_settings = {}
        observations = self.observation_prodiver(transitions["observations"], self.device)
        actions = self.action_prodiver(transitions["actions"], self.device)
        returns = self.return_prodiver(transitions["returns"], self.device)
        advantages = self.advantage_prodiver(transitions["advantages"], self.device)
        
        for i in range(len(self.optimizer.param_groups)): 
            self.optimizer.param_groups[i]['lr'] = extra_settings["learning_rate"]

        clip_epsilon = extra_settings["clip_epsilon"]
        entropy_coefficient = extra_settings["entropy_coefficient"]
        value_coefficient = extra_settings["value_coefficient"]
        value_clip_range = extra_settings["value_clip_range"]

        self.policy_model.train()

        self.optimizer.zero_grad()

        network_output = self.policy_model(observations, self.rnn_sequence_length, self.rnn_burn_in_length)
        with torch.no_grad():
            network_old_output = self.old_policy_model(observations, self.rnn_sequence_length, 0)

        # value clipping
        value_losses = []
        value_loss_sum = 0
        for i_head in range(self.value_head_count):
            transformed_return = self.invertible_value_functions[i_head](returns[:,i_head], False)
            old_value = network_old_output["value"][i_head]
            value = network_output["value"][i_head]
            clipped_value = old_value + torch.clip(value - old_value, -value_clip_range[i_head], value_clip_range[i_head])
            no_clipped_value_loss = torch.square(transformed_return - value)
            clipped_value_loss = torch.square(transformed_return - clipped_value)
            value_loss = torch.minimum(no_clipped_value_loss, clipped_value_loss).mean()
            value_losses.append(value_loss)
            value_loss_sum += value_coefficient[i] * value_loss

        # policy ratio
        ratio = 1
        for i_space in range(len(self.action_space)):
            action_log_prob = network_output["policy_distribution"][i_space].log_prob(actions[:, i_space])
            old_action_log_prob = network_old_output["policy_distribution"][i_space].log_prob(actions[:, i_space])
            ratio += action_log_prob - old_action_log_prob

        # PPO loss
        p_opt_a = ratio * advantages
        p_opt_b = torch.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        surrogate_loss = torch.minimum(p_opt_a, p_opt_b).mean()

        entropy = network_output["entropy"]
        loss = -surrogate_loss + value_loss_sum - entropy_coefficient * entropy

        clipped_ratio = torch.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        condition = torch.eq(ratio, clipped_ratio)
        policy_no_clip_event = torch.where(condition, torch.tensor(1.0), torch.tensor(0.0))
        policy_clip_event_ratio = (1 - policy_no_clip_event).mean()

        loss.backward()
        self.optimizer.step()

        return loss.item(), surrogate_loss.item(), value_loss_sum.item(), entropy.item(), policy_clip_event_ratio.item()

    def sample_actions_and_get_values(self, observations, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}
        
        self.policy_model.eval()
        observations = self.observation_prodiver(observations, self.device)

        if self.use_rnn:
            network_output = self.policy_model(observations, 1, 0)
            actions = np.asarray([act.to('cpu').detach().numpy() for act in network_output["sample_action"]])
            return np.transpose(actions, [1, 0]), [v.to('cpu').detach().numpy() for v in network_output["value"]], network_output["next_memory"].to('cpu').detach().numpy()
        else:
            network_output = self.policy_model(observations)
            actions = np.asarray([act.to('cpu').detach().numpy() for act in network_output["sample_action"]])
            return np.transpose(actions, [1, 0]), [v.to('cpu').detach().numpy() for v in network_output["value"]]

    def update_old_policy(self):
        self.old_policy_model.load_state_dict(self.policy_model.state_dict())

    def save(self, path, time_step):
        '''save NN model (give a directory name for the model) '''
        torch.save(self.policy_model.state_dict(), f"{path}/model_{time_step}.ckpt")
        print(f"Model saved in file: {path}/model_{time_step}.ckpt")

    def load(self, path):
        '''load NN model (give a directory name for the model) '''
        self.policy_model.load_state_dict(torch.load(path))
        print("Model restored.")

    def save_to_agent_pool(self, agent_pool_path, time_step="latest"):
        self.save(agent_pool_path, time_step)
        print(f"Model saved to an agent pool: {agent_pool_path}")
 



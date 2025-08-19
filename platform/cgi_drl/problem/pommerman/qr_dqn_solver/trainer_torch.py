from cgi_drl.problem.workflow_template.value_based_drl.qr_dqn_template_torch import QrDqnSolver
import importlib
import numpy as np
from tensorboardX import SummaryWriter
from collections import deque, defaultdict

class PommermanQrDqnSolver(QrDqnSolver):
    def __init__(self, solver_config):
        load = solver_config["load_function"]
        
        preprocessor_config = load(*solver_config["observation_preprocessor"])
        solver_config["observation_preprocessor"] = preprocessor_config

        buffer_config = load(*solver_config["experience_replay"])
        solver_config["experience_replay"] = buffer_config

        ppo_config = load(*solver_config["qr_dqn"])
        solver_config["qr_dqn"] = ppo_config

        super().__init__(solver_config)

        self.exploration_bonus_coefficient_scheduler = solver_config["exploration_bonus_coefficient_scheduler"]
        
        self.is_first_initialization = True
        
    def initialize(self, **kwargs):
        if self.is_first_initialization:
            from cgi_drl.environment.pommerman.pommerman_observation_preprocessor import PommermanObservationPreprocessor
            self.observation_preprocessor = PommermanObservationPreprocessor(self.solver_config["observation_preprocessor"])
            
            self.solver_config["qr_dqn"]["action_space"] = [6]
            from cgi_drl.decision_model.qr_dqn.policy_trainer_torch import PolicyTrainer
            self.policy = PolicyTrainer(self.solver_config["qr_dqn"])
            
            self.solver_config["experience_replay"]["agent_count"] = self.get_agent_count()
            buffer_config = self.solver_config["experience_replay"]
            if "is_prioritized" in buffer_config and buffer_config["is_prioritized"]:
                if self.policy.use_rnn:
                    from cgi_drl.data_storage.experience_replay.r2d2_experience_replay import R2d2ExperienceReplay
                    replay_buffer = R2d2ExperienceReplay(buffer_config)
                else:
                    from cgi_drl.data_storage.experience_replay.prioritized_experience_replay import PrioritizedExperienceReplay
                    replay_buffer = PrioritizedExperienceReplay(buffer_config)
                replay_buffer.is_prioritized = True
            else:
                if self.policy.use_rnn:
                    from cgi_drl.data_storage.experience_replay.recurrent_experience_replay import RecurrentExperienceReplay
                    replay_buffer = RecurrentExperienceReplay(buffer_config)
                else:
                    from cgi_drl.data_storage.experience_replay.simple_experience_replay import SimpleExperienceReplay
                    replay_buffer = SimpleExperienceReplay(buffer_config)
                replay_buffer.is_prioritized = False
            self.replay_buffer = replay_buffer
              
            super().initialize(**kwargs)
            self.current_stage = kwargs.get('stage', 0)
            self.is_first_initialization = False
        self.evaluation_scores = []
            
    def decide_agent_actions(self, is_valid_agent, is_train=True):
        if self.policy.use_rnn:
            behavior_q_values, memories = self.policy.get_behavior_q_values(self.observations[is_train])
        else:
            behavior_q_values = self.policy.get_behavior_q_values(self.observations[is_train])
        agent_count = self.get_agent_count(is_train)
        if is_train:
            action_epsilons = self.exploration_action_epsilon_scheduler({
                "current_timestep":self.total_time_step,
                "total_timestep":self.training_steps,
                "agent_count":agent_count,
            })
        else:
            action_epsilons = np.full(agent_count, self.evaluation_action_epsilon)
            
        random_values = np.random.uniform(0, 1, agent_count)
        random_actions = np.random.randint(6, size=agent_count)
        actions = []
        
        for i in range(agent_count):
            if random_values[i] <= action_epsilons[i]:
                actions.append(random_actions[i])
            else:
                action = []
                for i_space in range(len(self.policy.action_space)):
                    q = np.zeros(self.policy.action_space[i_space]) 
                    for i_head in range(self.policy.value_head_count):
                        q += behavior_q_values[i_head][i_space][i]
                    action.append(np.argmax(q))
                actions.append(action[0])
            
        if self.policy.use_rnn:
            decision = {
                "actions": actions,
                "behavior_q_values": behavior_q_values,
                "memories": memories,
            }
        else:
            decision = {
                "actions": actions,
                "behavior_q_values": behavior_q_values,
            }
        return decision
        
    def on_time_step(self, decision, is_valid_agent, is_train=True):
        actions = decision["actions"]
        behavior_q_values = decision["behavior_q_values"]
        if self.policy.use_rnn:
            memories = decision["memories"]
        
        next_observations, rewards, dones, infos = self.get_environment(is_train).step(actions) 
        next_observations = self.observation_preprocessor.process(next_observations)
        _next_observations = {}
        for key in next_observations:
            if key not in _next_observations:
                _next_observations[key] = [None for _ in range(self.get_agent_count(is_train))]
            for i in range(self.get_agent_count(is_train)):
                _next_observations[key][i] = self.observation_preprocessor.observation_aggregator(key, self.observations[is_train][key][i], next_observations[key][i])
        
        for i in range(self.get_agent_count(is_train)):
            q_sum = 0
            for i_head in range(self.policy.value_head_count):
                q_sum += behavior_q_values[i_head][0][i][actions[i]]
            infos[i]["Q Value Sum"] = q_sum
            infos[i]["Is Valid Agent"] = is_valid_agent[i]
        self.agent_statistics_aggregator(self.agent_statistics[is_train], rewards, infos)
        exploration_bonus_coefficient = self.exploration_bonus_coefficient_scheduler({
            "current_timestep" : self.total_time_step
        })
        for i in range(self.get_agent_count(is_train)):
            rewards[i] += exploration_bonus_coefficient * infos[i]["exploration_reward"]
        rewards = self.reward_transformer(rewards, infos)

        if is_train:
            for i in range(self.get_agent_count(is_train)):
                if is_valid_agent[i]:
                    obs = {}
                    for key in self.observations[is_train]:
                        obs[key] = np.asarray(self.observations[is_train][key][i])
                    self.previous_observations[i].append(obs)
                    self.previous_actions[i].append([actions[i]])
                    self.previous_rewards[i].append([rewards[i]])
                    
        for i in range(self.get_agent_count(is_train)):
            max_game_step = self.max_game_step if is_train else self.evaluation_max_game_step
            if is_valid_agent[i] and self.agent_statistics[is_train][i]["Episode Length"] >= max_game_step:
                dones[i] = True
                
        if self.policy.use_rnn:
            _next_observations["observation_memory"] = memories
            _next_observations["observation_previous_reward"] = np.asarray(rewards, dtype=np.float32)[..., None]
            _next_observations["observation_previous_action"] = np.asarray(actions, dtype=np.float32)[..., None]


        if is_train:
            for i in range(self.get_agent_count(is_train)):
                if is_valid_agent[i]:
                    next_obs = {}
                    for key in self.observations[is_train]:
                        next_obs[key] = np.asarray(_next_observations[key][i])
                    if self.policy.use_rnn:
                        self.replay_buffer.append(i, (self.previous_observations[i][0], self.previous_actions[i][0], np.asarray(self.previous_rewards[i], dtype=np.float32), dones[i], next_obs))
                    else:
                        self.replay_buffer.append(self.previous_observations[i][0], self.previous_actions[i][0], np.asarray(self.previous_rewards[i], dtype=np.float32), dones[i], next_obs)

                    if dones[i]:
                        self.previous_observations[i].clear()
                        self.previous_actions[i].clear()
                        self.previous_rewards[i] = deque([np.zeros(self.policy.value_head_count) for _ in range(self.n_step_size)], maxlen=self.n_step_size)
                        
        if is_train:
            if self.is_preparing_data:
                if self.replay_buffer.size() >= self.minimal_replay_memory_size:
                    self.is_preparing_data = False
                self.total_time_step -= sum(is_valid_agent)
            else:
                if (self.total_time_step % self.update_step_frequent) == 0:
                    self.update()

        self.observations[is_train] = _next_observations
        return dones

    def save_model(self):
        self.policy.save(self.model_path, self.total_time_step)
        self.latest_model_name = f"model_{self.total_time_step}.ckpt"

    def save_to_agent_pool(self, agent_pool_path):
        self.policy.save_to_agent_pool(agent_pool_path)

    def load_from_agent_pool(self, agent_pool_path):
        self.policy.load(agent_pool_path)

    def load_model(self, is_from_file=False):
        if is_from_file:
            model_path = self.solver_config.get("load_policy_model_path", self.model_path)
            self.policy.load(path=model_path)
        else:
            self.policy.load(self.model_path + f"/{self.latest_model_name}")
from cgi_drl.problem.workflow_template.ppo_template_torch import PpoSolver

class PommermanPpoSolver(PpoSolver):
    def __init__(self, solver_config):
        load = solver_config["load_function"]
        
        preprocessor_config = load(*solver_config["observation_preprocessor"])
        solver_config["observation_preprocessor"] = preprocessor_config

        buffer_config = load(*solver_config["gae"])
        solver_config["gae"] = buffer_config

        ppo_config = load(*solver_config["ppo"])
        solver_config["ppo"] = ppo_config

        super().__init__(solver_config)

        self.exploration_bonus_coefficient_scheduler = solver_config["exploration_bonus_coefficient_scheduler"]
        self.is_initialized = False

    def initialize(self, **kwargs):
        if not self.is_initialized:
            from cgi_drl.environment.pommerman.pommerman_observation_preprocessor import PommermanObservationPreprocessor
            self.observation_preprocessor = PommermanObservationPreprocessor(self.solver_config["observation_preprocessor"])
            
            self.solver_config["gae"]["agent_count"] = self.get_agent_count()
            from cgi_drl.data_storage.gae_sample_memory.gae_sample_memory import GaeSampleMemory
            self.gae_replay_buffer = GaeSampleMemory(self.solver_config["gae"])

            self.solver_config["ppo"]["action_space"] = [6]
            from cgi_drl.decision_model.ppo.policy_trainer_torch import PolicyTrainer
            self.policy = PolicyTrainer(self.solver_config["ppo"])
            
            super().initialize(**kwargs)
            self.current_stage = kwargs.get('stage', 0)
            self.is_initialized = True
            
    def terminate(self):
        for i in range(self.get_agent_count(True)):
            self.gae_replay_buffer.early_terminate(i)
        super().terminate()

    def on_time_step(self, decision, is_valid_agent, is_train=True):
        actions = decision["actions"]
        values = decision["values"][0]
        if self.policy.use_rnn:
            memory_outputs = decision["memory"]

        next_observations, rewards, dones, infos = self.get_environment(is_train).step(actions) 
        next_observations = self.observation_preprocessor.process(next_observations)
        _next_observations = {}
        for key in next_observations:
            if key not in _next_observations:
                _next_observations[key] = [None for _ in range(self.get_agent_count(is_train))]
            for i in range(self.get_agent_count(is_train)):
                _next_observations[key][i] = self.observation_preprocessor.observation_aggregator(key, self.observations[is_train][key][i], next_observations[key][i])
        if self.policy.use_rnn:
            _next_observations["observation_memory"] = memory_outputs
            _next_observations["observation_previous_action"] = actions
            _next_observations["observation_previous_reward"] = self.reward_transformer(rewards, infos)

        for i in range(self.get_agent_count(is_train)):
            infos[i]["Value"] = values[i][0]
            infos[i]["Is Valid Agent"] = is_valid_agent[i]
        self.agent_statistics_aggregator(self.agent_statistics[is_train], rewards, infos)
        exploration_bonus_coefficient = self.exploration_bonus_coefficient_scheduler({
            "current_timestep" : self.total_time_step
        })
        for i in range(self.get_agent_count(is_train)):
            rewards[i] += exploration_bonus_coefficient * infos[i]["exploration_reward"]
        rewards = self.reward_transformer(rewards, infos)
        for i in range(self.get_agent_count(is_train)):
            max_game_step = self.max_game_step if is_train else self.evaluation_max_game_step
            if is_valid_agent[i] and self.agent_statistics[is_train][i]["Episode Length"] >= max_game_step:
                dones[i] = True

        if is_train:
            for i in range(self.get_agent_count(is_train)):
                if is_valid_agent[i]:
                    obs = {}
                    for key in self.observations[is_train]:
                        obs[key] = self.observations[is_train][key][i]
                    self.gae_replay_buffer.append(i, {
                        "observation": obs,
                        "action": actions[i],
                        "reward": rewards[i],
                        "value": values[i][0],
                        "done": dones[i],
                    })

            if len(self.gae_replay_buffer) >= self.update_sample_count:
                self.update()
                self.gae_replay_buffer.clear_buffer()

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
from cgi_drl.problem.workflow_template.ppo_template_torch import PpoSolver
import importlib
import numpy as np

def launch(problem_config):
    load = problem_config["load_function"]
    # setup environment

    env_config = load(*problem_config["environment"])
    problem_config["environment"] = env_config
    # from cgi_drl.environment.unity_gym.unity_environment_wrapper import UnityEnvironmentWrapper
    from cgi_drl.environment.distributed_framework.environment_requester import EnvironmentRequester
    env = EnvironmentRequester(env_config) 
    # env = UnityEnvironmentWrapper(env_config)

    # setup observation preprocessor
    preprocessor_config = load(*problem_config["observation_preprocessor"])
    problem_config["observation_preprocessor"] = preprocessor_config
    from cgi_drl.environment.unity_gym.food_collector_observation_preprocessor import FoodCollectorObservationPreprocessor
    preprocessor = FoodCollectorObservationPreprocessor(preprocessor_config)

    # setup gae
    buffer_config = load(*problem_config["gae"])
    buffer_config["agent_count"] = env.agent_count
    problem_config["gae"] = buffer_config
    from cgi_drl.data_storage.gae_sample_memory.gae_sample_memory import GaeSampleMemory
    gae_replay_buffer = GaeSampleMemory(buffer_config)

    # setup policy
    ppo_config = load(*problem_config["ppo"])
    ppo_config["action_space"] = env.get_action_space()
    problem_config["ppo"] = ppo_config
    from cgi_drl.decision_model.ppo.policy_trainer_torch import PolicyTrainer
    policy = PolicyTrainer(ppo_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = UnityPpoSolver(problem_config)
    solver.environment = env
    solver.evaluation_environment = env
    solver.gae_replay_buffer = gae_replay_buffer
    solver.observation_preprocessor = preprocessor
    solver.policy = policy

    solver.train()

class UnityPpoSolver(PpoSolver):
    def __init__(self, solver_config):
        super().__init__(solver_config)

    def on_time_step(self, decision, is_valid_agent, is_train=True):
        actions = decision["actions"]
        values = decision["values"]
        if self.policy.use_rnn:
            memorys = decision["memorys"]

        next_observations, rewards, dones, infos = self.get_environment(is_train).step(actions) 
        next_observations = self.observation_preprocessor.process(next_observations)
        _next_observations = {}
        for key in next_observations:
            if key not in _next_observations:
                _next_observations[key] = [None for _ in range(self.get_agent_count(is_train))]
            for i in range(self.get_agent_count(is_train)):
                _next_observations[key][i] = self.observation_preprocessor.observation_aggregator(key, self.observations[is_train][key][i], next_observations[key][i])
        if self.policy.use_rnn:
            _next_observations["observation_memory"] = memorys

        for i in range(self.get_agent_count(is_train)):
            infos[i]["Value"] = np.asarray(values)[:,i].mean()
            infos[i]["Is Valid Agent"] = is_valid_agent[i]
            infos[i]["Action"] = actions[i]
            if len(self.evaluation_scores) == 0:
                infos[i]["Previous Evaluation Mean Score"] = -999
            else:
                infos[i]["Previous Evaluation Mean Score"] = np.mean(self.evaluation_scores)
        self.agent_statistics_aggregator(self.agent_statistics[is_train], rewards, infos)
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
                        "value": np.asarray(values)[:,i].mean(),
                        "done": dones[i],
                    })

            if len(self.gae_replay_buffer) >= self.update_sample_count:
                self.update()
                self.gae_replay_buffer.clear_buffer()

            if self.epoch_steps > 0 and (self.total_time_step % self.epoch_steps) == 0:
                self.evaluation()
                # Gym Unity with vision only support 1 local port by default
                for i in range(self.get_agent_count(is_train)):
                    dones[i] = True

        self.observations[is_train] = _next_observations
        return dones

    def evaluation(self):
        super().evaluation()
        self.observation_preprocessor.create_video(self.evaluation_best_observations, self.video_path + '/at_time_step{}_best_score_{}.mp4'.format(self.total_time_step, np.max(self.evaluation_scores)))

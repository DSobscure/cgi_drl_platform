from cgi_drl.problem.workflow_template.ppo_eval_template_torch import PpoSolver
import importlib
import numpy as np

def launch(problem_config):
    load = problem_config["load_function"]
    # setup environment

    env_config = load(*problem_config["environment"])
    problem_config["environment"] = env_config
    from cgi_drl.environment.distributed_framework.environment_requester import EnvironmentRequester
    env = EnvironmentRequester(env_config) 

    # setup observation preprocessor
    preprocessor_config = load(*problem_config["observation_preprocessor"])
    problem_config["observation_preprocessor"] = preprocessor_config
    from cgi_drl.environment.unity_gym.food_collector_observation_preprocessor import FoodCollectorObservationPreprocessor
    preprocessor = FoodCollectorObservationPreprocessor(preprocessor_config)

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
    solver.observation_preprocessor = preprocessor
    solver.policy = policy

    solver.evaluate()

class UnityPpoSolver(PpoSolver):
    def __init__(self, solver_config):
        super().__init__(solver_config)

    def on_time_step(self, decision, is_valid_agent):
        actions = decision["actions"]
        values = decision["values"]
        if self.policy.use_rnn:
            memories = decision["memories"]

        next_observations, rewards, dones, infos = self.get_environment().step(actions) 
        next_observations = self.observation_preprocessor.process(next_observations)
        _next_observations = {}
        for key in next_observations:
            if key not in _next_observations:
                _next_observations[key] = [None for _ in range(self.get_agent_count())]
            for i in range(self.get_agent_count()):
                _next_observations[key][i] = self.observation_preprocessor.observation_aggregator(key, self.observations[key][i], next_observations[key][i])

        for i in range(self.get_agent_count()):
            infos[i]["Value"] = np.asarray(values)[:,i].mean()
            infos[i]["Is Valid Agent"] = is_valid_agent[i]
            infos[i]["Action"] = actions[i]
            if len(self.evaluation_scores) == 0:
                infos[i]["Previous Evaluation Mean Score"] = -999
            else:
                infos[i]["Previous Evaluation Mean Score"] = np.mean(self.evaluation_scores)
        self.agent_statistics_aggregator(self.agent_statistics, rewards, infos)
        rewards = self.reward_transformer(rewards, infos)
        for i in range(self.get_agent_count()):
            max_game_step = self.evaluation_max_game_step
            if is_valid_agent[i] and self.agent_statistics[i]["Episode Length"] >= max_game_step:
                dones[i] = True
                
        if self.policy.use_rnn:
            _next_observations["observation_memory"] = memories
            _next_observations["observation_previous_reward"] = rewards
            _next_observations["observation_previous_action"] = actions

        self.observations = _next_observations
        return dones

    def summarize_evaluation(self):
        super().summarize_evaluation()
        self.observation_preprocessor.create_video(self.evaluation_best_observations, self.video_path + '/best_score_{}.mp4'.format(np.max(self.evaluation_scores)))

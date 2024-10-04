from cgi_drl.problem.reinforcement_learning_evaluator import ReinforcementLearningEvaluator
import numpy as np
from collections import defaultdict
from tensorboardX import SummaryWriter

class PpoSolver(ReinforcementLearningEvaluator):
    def __init__(self, solver_config):
        super().__init__(solver_config)
        self.solver_config = solver_config

        # logs
        self.summary_writer = SummaryWriter(self.log_path)
        self.log_file = self._open_log(self.log_path, "log_file.txt")

        # setup policy
        self.policy_config = solver_config["ppo"]

        self.agent_statistics_aggregator = solver_config["agent_statistics_aggregator"]
        self.evaluation_max_game_step = solver_config["evaluation_max_game_step"]
        self.reward_transformer = solver_config["reward_transformer"]

        self.evaluation_scores = []
        
    def get_agent_count(self):
        return self.environment.agent_count

    def get_environment(self):
        return self.environment

    def initialize(self, **kwargs):
        from os.path import basename
        model_path = self.solver_config["load_policy_model_path"]
        self.policy.load(path=model_path)
        print("load policy model from {}".format(model_path))

        print("=" * 18 + " Setup model " + "=" * 19)
        print("mode: evaluation with PPO")
        print("=" * 50)

        # episode info
        self.episode_count = 0
        self.observations = {}
        self.agent_statistics = [{} for _ in range(self.get_agent_count())]
        self.eval_trajectory_observations = [defaultdict(list) for _ in range(self.get_agent_count())]

    def terminate(self):
        self.environment.close()

    def episode_initiate(self, dones, is_valid_agent):
        for i in range(self.get_agent_count()):
            if is_valid_agent[i] and dones[i]:
                dones[i] = False
                observation = self.observation_preprocessor.process(self.get_environment().reset(i))
                for key in observation:
                    observation[key] = self.observation_preprocessor.observation_aggregator(key, None, observation[key][0], True)
                for key in observation:
                    if key not in self.observations:
                        self.observations[key] = [None for _ in range(self.get_agent_count())]
                    self.observations[key][i] = observation[key]
                if self.policy.use_rnn:
                    if "observation_memory" not in self.observations:
                        self.observations["observation_memory"] = [None for _ in range(self.get_agent_count())]
                    self.observations["observation_memory"][i] = np.zeros((self.policy.memory_size), dtype=np.float32)
                    
                    if "observation_previous_reward" not in self.observations:
                        self.observations["observation_previous_reward"] = [None for _ in range(self.get_agent_count())]
                    self.observations["observation_previous_reward"][i] = np.zeros(self.policy.value_head_count, dtype=np.float32)
                    
                    if "observation_previous_action" not in self.observations:
                        self.observations["observation_previous_action"] = [None for _ in range(self.get_agent_count())]
                self.agent_statistics[i] = {}
                self.eval_trajectory_observations[i] = defaultdict(list)
        return dones

    def episode_terminate(self, dones, is_valid_agent):
        self.episode_count += [dones[i] and is_valid_agent[i] for i in range(len(is_valid_agent))].count(True)
        shared_log_str = "Evaluation: {}/{}".format(self.episode_count, self.evaluation_episode_count)

        for i in range(self.get_agent_count()):
            if is_valid_agent[i] and dones[i]:
                log_str = shared_log_str + "-agent{} statistic| ".format(i)
                for key in self.agent_statistics[i]:
                    log_str = log_str + "{}: {}, ".format(key, self.agent_statistics[i][key])
                    prefix = "Evaluation/Agent Statistics/"
                    self.summary_writer.add_scalar(prefix + key, self.agent_statistics[i][key])
                self.log_file.write(log_str)
                print(log_str)
                if len(self.evaluation_scores) == 0 or self.agent_statistics[i]["Cumulated Extrinsic Reward"] > max(self.evaluation_scores):
                    self.evaluation_best_observations = self.eval_trajectory_observations[i]
                self.evaluation_scores.append(self.agent_statistics[i]["Cumulated Extrinsic Reward"])

    def decide_agent_actions(self, is_valid_agent):
        if self.policy.use_rnn:
            actions, values, memorys = self.policy.sample_actions_and_get_values(self.observations)
            decision = {
                "actions": actions,
                "values": values,
                "memorys": memorys,
            }
        else:
            actions, values = self.policy.sample_actions_and_get_values(self.observations)
            decision = {
                "actions": actions,
                "values": values
            }
        for i in range(self.get_agent_count()):
            for key in self.observations:
                self.eval_trajectory_observations[i][key].append(self.observations[key][i])
        return decision

    def on_time_step(self, decision, is_valid_agent):
        actions = decision["actions"]
        values = decision["values"]
        if self.policy.use_rnn:
            memorys = decision["memorys"]

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
        self.agent_statistics_aggregator(self.agent_statistics, rewards, infos)
        rewards = self.reward_transformer(rewards, infos)
        for i in range(self.get_agent_count()):
            max_game_step = self.evaluation_max_game_step
            if is_valid_agent[i] and self.agent_statistics[i]["Episode Length"] >= max_game_step:
                dones[i] = True
                
        if self.policy.use_rnn:
            _next_observations["observation_memory"] = memorys
            _next_observations["observation_previous_reward"] = np.asarray(rewards, dtype=np.float32)
            _next_observations["observation_previous_action"] = np.asarray(actions, dtype=np.float32)

        self.observations = _next_observations
        return dones

    def summarize_evaluation(self):
        log_str = "Evaluation MeanScore: {}, MaxScore: {}, ScoreStd: {}\n".format(np.mean(self.evaluation_scores), np.max(self.evaluation_scores), np.std(self.evaluation_scores))
        self.summary_writer.add_scalar('Evaluation/MeanScore', np.mean(self.evaluation_scores), 0)
        self.summary_writer.add_scalar('Evaluation/MaxScore', np.max(self.evaluation_scores), 0)
        self.summary_writer.add_scalar('Evaluation/ScoreStd', np.std(self.evaluation_scores), 0)
        print(log_str, end='')
        self.log_file.write(log_str)
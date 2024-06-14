from cgi_drl.problem.reinforcement_learning_trainer import ReinforcementLearningTrainer
import numpy as np
from collections import defaultdict
from tensorboardX import SummaryWriter

class PpoSolver(ReinforcementLearningTrainer):
    def __init__(self, solver_config):
        super().__init__(solver_config)
        self.solver_config = solver_config

        # logs
        self.summary_writer = SummaryWriter(self.log_path)
        self.log_file = self._open_log(self.log_path, "log.txt")
        self.evaluation_log_file = self._open_log(self.log_path, "evaluation_log_file.txt")

        # setup policy
        self.policy_config = solver_config["ppo"]

        # solver config
        self.batch_size = solver_config["batch_size"]
        self.epoch_steps = solver_config["epoch_steps"]
        self.training_steps = solver_config["training_steps"]
        self.update_sample_count = solver_config["update_sample_count"]
        self.update_epoch_count = solver_config["update_epoch_count"]

        self.learning_rate_scheduler = solver_config["learning_rate_scheduler"]
        self.clip_epsilon_scheduler = solver_config["clip_epsilon_scheduler"]
        self.entropy_coefficient_scheduler = solver_config["entropy_coefficient_scheduler"]
        self.value_coefficient_scheduler = solver_config["value_coefficient_scheduler"]
        self.value_clip_range_scheduler = solver_config["value_clip_range_scheduler"]

        self.max_game_step = solver_config["max_game_step"]
        self.agent_statistics_aggregator = solver_config["agent_statistics_aggregator"]
        self.discount_factor_gamma = solver_config["discount_factor_gamma"]
        self.discount_factor_lambda = solver_config["discount_factor_lambda"]

        self.evaluation_max_game_step = solver_config["evaluation_max_game_step"]
        self.evaluation_episode_count = solver_config["evaluation_episode_count"]
        self.reward_transformer = solver_config["reward_transformer"]

        self.evaluation_scores = []
        
    def get_agent_count(self, is_train=True):
        if is_train:
            return self.environment.agent_count
        else:
            return self.evaluation_environment.agent_count

    def get_environment(self, is_train=True):
        if is_train:
            return self.environment
        else:
            return self.evaluation_environment

    def initialize(self, **kwargs):
        if self.epoch_steps > 0 and self.epoch_steps % self.get_agent_count() != 0:
            raise AssertionError(f"epoch_steps should be divisible by agent count epoch_steps: {self.epoch_steps}, agent_count: {self.get_agent_count()}")
        if self.update_sample_count % self.get_agent_count() != 0:
            raise AssertionError(f"update_sample_count should be divisible by agent count")

        # load policy
        if self.solver_config.get("is_load_policy", False):
            from os.path import basename
            model_path = self.solver_config.get("load_policy_model_path", self.model_path)
            self.policy.load(path=model_path)
            print("load policy model from {}".format(model_path))

        print("=" * 18 + " Setup model " + "=" * 19)
        print("mode: training with PPO")
        print("=" * 50)

        self.total_time_step = self.solver_config.get("initial_time_step", 0)
        if self.total_time_step % self.get_agent_count() != 0:
            raise AssertionError(f"total_time_step should be divisible by agent count")
        print("Current timestep:", self.total_time_step)

        # episode info
        self.episode_count = 0
        self.observations = {
            True: {}, # for train
            False: {} # for eval
        }
        self.agent_statistics = {
            True: [{} for _ in range(self.get_agent_count())], # for train
            False: [{} for _ in range(self.get_agent_count(is_train=False))] # for eval
        }

        self.eval_trajectory_observations = [defaultdict(list) for _ in range(self.get_agent_count(is_train=False))]

    def terminate(self):
        self.environment.close()
        self.evaluation_environment.close()

    def episode_initiate(self, dones, is_valid_agent, is_train=True):
        for i in range(self.get_agent_count(is_train)):
            if is_valid_agent[i] and dones[i]:
                dones[i] = False
                observation = self.observation_preprocessor.process(self.get_environment(is_train).reset(i))
                for key in observation:
                    observation[key] = self.observation_preprocessor.observation_aggregator(key, None, observation[key][0], True)
                for key in observation:
                    if key not in self.observations[is_train]:
                        self.observations[is_train][key] = [None for _ in range(self.get_agent_count(is_train))]
                    self.observations[is_train][key][i] = observation[key]
                if self.policy.use_rnn:
                    if "observation_memory" not in self.observations[is_train]:
                        self.observations[is_train]["observation_memory"] = [None for _ in range(self.get_agent_count(is_train))]
                    self.observations[is_train]["observation_memory"][i] = np.zeros((self.policy.memory_size), dtype=np.float32)
                self.agent_statistics[is_train][i] = {}
                if not is_train:
                    self.eval_trajectory_observations[i] = defaultdict(list)
        return dones

    def episode_terminate(self, dones, is_valid_agent, is_train=True):
        if is_train:
            self.episode_count += [dones[i] and is_valid_agent[i] for i in range(len(is_valid_agent))].count(True)

        if is_train:
            shared_log_str = "{}: {}/{}".format(self.episode_count, self.total_time_step, self.training_steps)
        else:
            shared_log_str = "Evaluation: {}/{}".format(self.total_time_step, self.training_steps)

        for i in range(self.get_agent_count(is_train)):
            if is_valid_agent[i] and dones[i]:
                log_str = shared_log_str + "-agent{} statistic| ".format(i)
                for key in self.agent_statistics[is_train][i]:
                    log_str = log_str + "{}: {}, ".format(key, self.agent_statistics[is_train][i][key])
                    prefix = "Agent Statistics/" if is_train else "Evaluation/Agent Statistics/"
                    self.summary_writer.add_scalar(prefix + key, self.agent_statistics[is_train][i][key], self.total_time_step)
                if is_train:
                    print(log_str)
                    self.log_file.write(log_str+'\n')
                else:
                    self.evaluation_log_file.write(log_str)
                    if len(self.evaluation_scores) == 0 or self.agent_statistics[is_train][i]["Cumulated Extrinsic Reward"] > max(self.evaluation_scores):
                        self.evaluation_best_observations = self.eval_trajectory_observations[i]
                    self.evaluation_scores.append(self.agent_statistics[is_train][i]["Cumulated Extrinsic Reward"])

    def update(self):
        self.policy.update_old_policy()
        self.total_loss = 0
        self.total_surrogate_loss, self.total_value_loss, self.total_entropy = 0, 0, 0
        self.total_policy_clip_event_ratio = 0
        loss_counter = 0.0001

        batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
        sample_count = len(batches["action"])
        for _ in range(self.update_epoch_count):
            if self.policy.use_rnn:
                sequence_count = sample_count // self.policy.rnn_sequence_length
                random_sequence = np.random.permutation(sequence_count)
                batch_index = np.array(
                    [[k + i * self.policy.rnn_sequence_length for k in range(self.policy.rnn_sequence_length)] for i in random_sequence]
                ).ravel()
            else:
                batch_index = np.random.permutation(sample_count)
            observation_batch = {}
            for key in batches["observation"]:
                observation_batch[key] = batches["observation"][key][batch_index]
            action_batch = batches["action"][batch_index]
            return_batch = batches["return"][batch_index]
            adv_batch = batches["adv"][batch_index]

            for start in range(0, sample_count, self.batch_size):
                ob_train_batch = {}
                for key in observation_batch:
                    ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
                ac_train_batch = action_batch[start:start + self.batch_size]
                return_train_batch = return_batch[start:start + self.batch_size]
                adv_train_batch = adv_batch[start:start + self.batch_size]
                loss, surrogate_loss, value_loss, entropy, policy_clip_event_ratio = self.policy.update(
                    {
                        "observations": ob_train_batch,
                        "actions": ac_train_batch,
                        "returns": np.reshape(return_train_batch, [-1, 1]),
                        "advantages": adv_train_batch
                    },
                    {
                        "learning_rate": self.learning_rate_scheduler({
                            "current_timestep":self.total_time_step,
                            "total_timestep":self.training_steps}
                            ),
                        "clip_epsilon": self.clip_epsilon_scheduler({
                            "current_timestep":self.total_time_step,
                            "total_timestep":self.training_steps}
                            ),
                        "entropy_coefficient": self.entropy_coefficient_scheduler({
                            "current_timestep":self.total_time_step,
                            "total_timestep":self.training_steps}
                            ),
                        "value_coefficient": [self.value_coefficient_scheduler({
                            "current_timestep":self.total_time_step,
                            "total_timestep":self.training_steps}
                            )],
                        "value_clip_range": [self.value_clip_range_scheduler({
                            "current_timestep":self.total_time_step,
                            "total_timestep":self.training_steps}
                            )],
                    }
                )
                self.total_loss += loss
                self.total_surrogate_loss += surrogate_loss
                self.total_value_loss += value_loss
                self.total_entropy += entropy
                self.total_policy_clip_event_ratio += policy_clip_event_ratio
                loss_counter += 1
        self.total_loss /= loss_counter
        self.total_surrogate_loss /= loss_counter
        self.total_value_loss /= loss_counter
        self.total_entropy /= loss_counter
        self.total_policy_clip_event_ratio /= loss_counter
        log_str = "{}: {}/{}, loss:{}, surrogate_loss:{}, value_loss:{}, entropy:{}, policy_clip_event_ratio:{}".format(
            self.episode_count,
            self.total_time_step,
            self.training_steps,
            self.total_loss,
            self.total_surrogate_loss,
            self.total_value_loss,
            self.total_entropy,
            self.total_policy_clip_event_ratio,
        )
        self.summary_writer.add_scalar('PPO/Loss', self.total_loss, self.total_time_step)
        self.summary_writer.add_scalar('PPO/Surrogate Loss', self.total_surrogate_loss, self.total_time_step)
        self.summary_writer.add_scalar('PPO/Value Loss', self.total_value_loss, self.total_time_step)
        self.summary_writer.add_scalar('PPO/Entropy', self.total_entropy, self.total_time_step)
        self.summary_writer.add_scalar('PPO/Policy Clip Event Ratio', self.total_policy_clip_event_ratio, self.total_time_step)
        print(log_str)
        self.log_file.write(log_str+'\n')

    def decide_agent_actions(self, is_valid_agent, is_train=True):
        if self.policy.use_rnn:
            actions, values, memorys = self.policy.sample_actions_and_get_values(self.observations[is_train])
            decision = {
                "actions": actions,
                "values": values,
                "memorys": memorys,
            }
        else:
            actions, values = self.policy.sample_actions_and_get_values(self.observations[is_train])
            decision = {
                "actions": actions,
                "values": values
            }
        return decision

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

        self.observations[is_train] = _next_observations
        return dones

    def evaluation(self):
        # do saving
        self.policy.save(self.model_path, self.total_time_step)

        self.evaluation_scores = []
        episode_count = 0
        dones = [True for _ in range(self.get_agent_count(False))]
        is_valid_agent = [True for _ in range(self.get_agent_count(False))]
        while episode_count < self.evaluation_episode_count:
            dones = self.episode_initiate(dones, is_valid_agent, is_train=False)
            while not any(dones):
                for i in range(self.get_agent_count(False)):
                    for key in self.observations[False]:
                        self.eval_trajectory_observations[i][key].append(self.observations[False][key][i])
                decision = self.decide_agent_actions(is_valid_agent, is_train=False)
                dones = self.on_time_step(decision, is_valid_agent, is_train=False)
            episode_count += dones.count(True)
            self.episode_terminate(dones, is_valid_agent, is_train=False)

        log_str = "Evaluation at {}: {}/{}, MeanScore: {}, MaxScore: {}, ScoreStd: {}\n".format(
            self.episode_count, self.total_time_step, self.training_steps, 
            np.mean(self.evaluation_scores), np.max(self.evaluation_scores), np.std(self.evaluation_scores))
        
        self.summary_writer.add_scalar('Evaluation/MeanScore', np.mean(self.evaluation_scores), self.total_time_step)
        self.summary_writer.add_scalar('Evaluation/MaxScore', np.max(self.evaluation_scores), self.total_time_step)
        self.summary_writer.add_scalar('Evaluation/ScoreStd', np.std(self.evaluation_scores), self.total_time_step)
        print(log_str, end='')
        self.evaluation_log_file.write(log_str)
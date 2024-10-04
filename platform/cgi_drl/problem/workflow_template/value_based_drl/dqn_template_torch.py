from cgi_drl.problem.reinforcement_learning_trainer import ReinforcementLearningTrainer
import numpy as np
from tensorboardX import SummaryWriter
from collections import deque, defaultdict

class DqnSolver(ReinforcementLearningTrainer):
    def __init__(self, solver_config):
        super().__init__(solver_config)
        self.solver_config = solver_config

        # logs
        self.summary_writer = SummaryWriter(self.log_path)
        self.log_file = self._open_log(self.log_path, "log.txt")
        self.evaluation_log_file = self._open_log(self.log_path, "evaluation_log_file.txt")

        # solver config
        self.batch_size = solver_config["batch_size"]
        self.epoch_steps = solver_config["epoch_steps"]
        self.training_steps = solver_config["training_steps"]
        self.update_step_frequent = solver_config["update_step_frequent"]
        self.update_target_step_frequent = solver_config["update_target_step_frequent"]
        self.epoch_steps = solver_config["epoch_steps"]
        self.minimal_replay_memory_size = solver_config["minimal_replay_memory_size"]

        self.learning_rate_scheduler = solver_config["learning_rate_scheduler"]
        self.exploration_action_epsilon_scheduler = solver_config["exploration_action_epsilon_scheduler"]

        self.max_game_step = solver_config["max_game_step"]
        self.agent_statistics_aggregator = solver_config["agent_statistics_aggregator"]
        self.discount_factor_gammas = solver_config["discount_factor_gammas"]
        self.evaluation_action_epsilon = solver_config["evaluation_action_epsilon"]

        self.evaluation_max_game_step = solver_config["evaluation_max_game_step"]
        self.evaluation_episode_count = solver_config["evaluation_episode_count"]
        self.reward_transformer = solver_config["reward_transformer"]
        self.use_double_q = solver_config["use_double_q"]
        self.n_step_size = solver_config["n_step_size"]
        
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
        if self.epoch_steps % self.get_agent_count() != 0:
            raise AssertionError(f"epoch_steps should be divisible by agent count epoch_steps: {self.epoch_steps}, agent_count: {self.get_agent_count()}")
        if self.update_step_frequent % self.get_agent_count() != 0:
            raise AssertionError("update_step_frequent should be divisible by agent count")
        if self.update_target_step_frequent % self.get_agent_count() != 0:
            raise AssertionError("update_target_step_frequent should be divisible by agent count")

        if self.replay_buffer.is_prioritized: 
            self.importance_sampling_beta_scheduler = self.solver_config["importance_sampling_beta_scheduler"]

        # load policy
        if self.solver_config.get("is_load_policy", False):
            from os.path import basename
            model_path = self.solver_config.get("load_policy_model_path", self.model_path)
            self.policy.load(path=model_path)
            print("load policy model from {}".format(model_path))
        self.policy.update_target_policy()

        print("=" * 18 + " Setup model " + "=" * 19)
        print("mode: training with DQN")
        print("=" * 50)

        self.total_time_step = self.solver_config.get("initial_time_step", 0)
        if self.total_time_step % self.get_agent_count() != 0:
            raise AssertionError(f"total_time_step should be divisible by agent count")
        print("Current timestep:", self.total_time_step)

        # episode info
        self.episode_count = 0
        
        self.previous_observations = [deque(maxlen=self.n_step_size) for _ in range(self.get_agent_count())]
        self.previous_actions = [deque(maxlen=self.n_step_size) for _ in range(self.get_agent_count())]
        self.previous_rewards = [deque([np.zeros(self.policy.value_head_count) for _ in range(self.n_step_size)], maxlen=self.n_step_size) for _ in range(self.get_agent_count())]
        
        self.observations = {
            True: {}, # for train
            False: {} # for eval
        }
        self.agent_statistics = {
            True: [{} for _ in range(self.get_agent_count())], # for train
            False: [{} for _ in range(self.get_agent_count(is_train=False))] # for eval
        }

        self.eval_trajectory_observations = [defaultdict(list) for _ in range(self.get_agent_count(is_train=False))]

        self.is_preparing_data = True
        self.total_loss = 0
        self.update_counter = 0

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
                    self.observations[is_train]["observation_memory"][i] = np.zeros(self.policy.memory_size, dtype=np.float32)
                    
                    if "observation_previous_reward" not in self.observations[is_train]:
                        self.observations[is_train]["observation_previous_reward"] = [None for _ in range(self.get_agent_count(is_train))]
                    self.observations[is_train]["observation_previous_reward"][i] = np.zeros(self.policy.value_head_count, dtype=np.float32)
                    
                    if "observation_previous_action" not in self.observations[is_train]:
                        self.observations[is_train]["observation_previous_action"] = [None for _ in range(self.get_agent_count(is_train))]
                    self.observations[is_train]["observation_previous_action"][i] = np.zeros(len(self.policy.action_space), dtype=np.float32)
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

        if is_train and self.update_counter != 0:
            loss = self.total_loss / self.update_counter
            log_str = "{}: {}/{}, loss:{}".format(
                self.episode_count,
                self.total_time_step,
                self.training_steps,
                loss,
            )
            self.summary_writer.add_scalar("Agent Statistics/DQN/Loss", loss, self.total_time_step)
            print(log_str)
            self.log_file.write(log_str+'\n')

            self.update_counter = 0
            self.total_loss = 0  

    def update(self):
        if (self.total_time_step % self.update_target_step_frequent) == 0:
            self.policy.update_target_policy()

        if self.replay_buffer.is_prioritized: 
            importance_sampling_beta = self.importance_sampling_beta_scheduler({"current_timestep":self.total_time_step, "total_timestep":self.training_steps})
            state_batch, action_batch, reward_batch, done_batch, next_state_batch, random_indexes, importance_sampling_weights = self.replay_buffer.sample_mini_batch(self.batch_size, importance_sampling_beta)
        else:
            state_batch, action_batch, reward_batch, done_batch, next_state_batch = self.replay_buffer.sample_mini_batch(self.batch_size)

        if self.policy.use_rnn:
            target_q, _ = self.policy.get_target_q_values(next_state_batch)
        else:
            target_q = self.policy.get_target_q_values(next_state_batch)
        target_q_values = target_q
        q_values = np.zeros([self.batch_size, self.policy.value_head_count, len(self.policy.action_space)])

        if self.use_double_q:
            if self.policy.use_rnn:
                target_q_values, _ = self.policy.get_behavior_q_values(next_state_batch)
            else:
                target_q_values = self.policy.get_behavior_q_values(next_state_batch)

        non_terminal = (1 - np.asarray(done_batch, dtype=np.float32))
        for j in range(self.policy.value_head_count):
            for k in range(len(self.policy.action_space)):
                q_values[:, j, k] += non_terminal * target_q[j][k][:][np.arange(self.batch_size), np.argmax(target_q_values[j][k], axis=-1)]
                for l in range(self.n_step_size):
                    q_values[:, j, k] = reward_batch[:,-1-l,j] + self.discount_factor_gammas[j] * q_values[:, j, k]
        q_values = np.mean(q_values, axis=-1)
    
        if self.replay_buffer.is_prioritized:
            q_loss, q_losses = self.policy.update(
                {
                    "observations": state_batch,
                    "actions": action_batch,
                    "q_values": q_values,
                },
                {
                    "learning_rate": self.learning_rate_scheduler({"current_timestep":self.total_time_step, "total_timestep":self.training_steps}),
                    "loss_weights": importance_sampling_weights
                }
            )
        else:
            q_loss, q_losses = self.policy.update(
                {
                    "observations": state_batch,
                    "actions": action_batch,
                    "q_values": q_values,
                },
                {
                    "learning_rate": self.learning_rate_scheduler({"current_timestep":self.total_time_step, "total_timestep":self.training_steps}),
                }
            )
        self.total_loss += q_loss
        self.update_counter += 1
        
        if self.replay_buffer.is_prioritized: 
            self.replay_buffer.update_batch(random_indexes, q_losses)

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
        random_actions = self.get_environment(is_train).sample()
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
                actions.append(action)
            
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
            for i_space in range(len(self.policy.action_space)):
                for i_head in range(self.policy.value_head_count):
                    q_sum += behavior_q_values[i_head][i_space][i][actions[i][i_space]]
            infos[i]["Q Value Sum"] = q_sum
            infos[i]["Is Valid Agent"] = is_valid_agent[i]
        self.agent_statistics_aggregator(self.agent_statistics[is_train], rewards, infos)
        rewards = self.reward_transformer(rewards, infos)

        if is_train:
            for i in range(self.get_agent_count(is_train)):
                if is_valid_agent[i]:
                    obs = {}
                    for key in self.observations[is_train]:
                        obs[key] = np.asarray(self.observations[is_train][key][i])
                    self.previous_observations[i].append(obs)
                    self.previous_actions[i].append(actions[i])
                    self.previous_rewards[i].append(rewards[i])

        for i in range(self.get_agent_count(is_train)):
            max_game_step = self.max_game_step if is_train else self.evaluation_max_game_step
            if is_valid_agent[i] and self.agent_statistics[is_train][i]["Episode Length"] >= max_game_step:
                dones[i] = True
                
        if self.policy.use_rnn:
            _next_observations["observation_memory"] = memories
            _next_observations["observation_previous_reward"] = np.asarray(rewards, dtype=np.float32)
            _next_observations["observation_previous_action"] = np.asarray(actions, dtype=np.float32)

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
                if (self.total_time_step % self.epoch_steps) == 0:
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
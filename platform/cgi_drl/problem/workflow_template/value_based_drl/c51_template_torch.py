from cgi_drl.problem.workflow_template.value_based_drl.dqn_template_torch import DqnSolver
import numpy as np
from tensorboardX import SummaryWriter
from collections import deque, defaultdict

def cross_entropy(label, predict, minimal_epsilon=1e-6):
    h = 0
    label = label + minimal_epsilon
    predict = predict + minimal_epsilon
    for i_atom in range(len(label)):
        if label[i_atom] > 0 and predict[i_atom] > 0:
            h += -1 * label[i_atom] * np.log(predict[i_atom])
    return h

class C51Solver(DqnSolver):
    def initialize(self):
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
        print("mode: training with C51")
        print("=" * 50)

        self.total_time_step = self.solver_config.get("initial_time_step", 0)
        if self.total_time_step % self.get_agent_count() != 0:
            raise AssertionError(f"total_time_step should be divisible by agent count")
        print("Current timestep:", self.total_time_step)

        # episode info
        self.episode_count = 0
        
        self.previous_observations = [deque(maxlen=self.n_step_size) for _ in range(self.get_agent_count())]
        self.previous_actions = [deque(maxlen=self.n_step_size) for _ in range(self.get_agent_count())]
        self.previous_rewards = [deque(maxlen=self.n_step_size) for _ in range(self.get_agent_count())]
        
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

    def update(self):
        if (self.total_time_step % self.update_target_step_frequent) == 0:
            self.policy.update_target_policy()

        if self.replay_buffer.is_prioritized: 
            importance_sampling_beta = self.importance_sampling_beta_scheduler({"current_timestep":self.total_time_step, "total_timestep":self.training_steps})
            state_batch, action_batch, reward_batch, done_batch, next_state_batch, random_indexes, importance_sampling_weights = self.replay_buffer.sample_mini_batch(self.batch_size, importance_sampling_beta)
        else:
            state_batch, action_batch, reward_batch, done_batch, next_state_batch = self.replay_buffer.sample_mini_batch(self.batch_size)

        target_q_value, target_q_distribution = self.policy.get_target_q_values_and_distributions(next_state_batch)
        q_distributions = np.zeros([self.batch_size, self.policy.value_head_count, len(self.policy.action_space), self.policy.value_atom_count])
        q_distributions[:,:,:,self.policy.value_zero_index] = 1

        if self.use_double_q:
            next_behavior_q_values = self.policy.get_behavior_q_values(next_state_batch)

        for i in range(self.batch_size):
            for j in range(self.policy.value_head_count):
                for k in range(len(self.policy.action_space)):
                    if not done_batch[i]:
                        if self.use_double_q:
                            q_distributions[i, j, k] = target_q_distribution[j][k][i][np.argmax(next_behavior_q_values[j][k][i])]
                        else:
                            q_distributions[i, j, k] = target_q_distribution[j][k][i][np.argmax(target_q_value[j][k][i])]
                    for l in range(len(reward_batch[i])):
                        q_distributions[i, j, k] = self.policy.distributional_bellman_operator(q_distributions[i, j, k], self.discount_factor_gamma, reward_batch[i][-1-l])
        q_distributions = np.mean(q_distributions, axis=-2)
    
        if self.replay_buffer.is_prioritized: 
            predict_behavior_q_distributions = self.policy.get_behavior_q_distributions(state_batch)
            priority_values = np.zeros(self.batch_size)
            for i in range(self.batch_size):
                for j in range(self.policy.value_head_count):
                    for k in range(len(self.policy.action_space)):
                        priority_values[i] += cross_entropy(q_distributions[i, j], predict_behavior_q_distributions[j][k][i][action_batch[i][k]])
            self.replay_buffer.update_batch(random_indexes, priority_values)

        if self.replay_buffer.is_prioritized:
            distribution_loss, distribution_losses = self.policy.update(
                {
                    "observations": state_batch,
                    "actions": action_batch,
                    "q_distributions": q_distributions,
                },
                {
                    "learning_rate": self.learning_rate_scheduler({"current_timestep":self.total_time_step, "total_timestep":self.training_steps}),
                    "loss_weights": importance_sampling_weights
                }
            )
        else:
            distribution_loss, distribution_losses = self.policy.update(
                {
                    "observations": state_batch,
                    "actions": action_batch,
                    "q_distributions": q_distributions,
                },
                {
                    "learning_rate": self.learning_rate_scheduler({"current_timestep":self.total_time_step, "total_timestep":self.training_steps}),
                    "loss_weights": importance_sampling_weights
                }
            )
        self.total_loss += distribution_loss
        self.update_counter += 1
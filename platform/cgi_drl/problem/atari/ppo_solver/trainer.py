from cgi_drl.problem.workflow_template.ppo_template import PpoSolver
import importlib
import numpy as np
import tensorflow as tf


def launch(problem_config):
    load = problem_config["load_function"]
    # setup environment
    env_config = load(*problem_config["environment"])
    evaluation_env_config = load(*problem_config["evaluation_environment"])
    problem_config["environment"] = env_config
    problem_config["evaluation_environment"] = evaluation_env_config
    from cgi_drl.environment.distributed_framework.environment_requester import EnvironmentRequester
    train_env = EnvironmentRequester(env_config)
    eval_env = EnvironmentRequester(evaluation_env_config)

    # setup observation preprocessor
    preprocessor_config = load(*problem_config["observation_preprocessor"])
    problem_config["observation_preprocessor"] = preprocessor_config
    from cgi_drl.environment.atari.atari_observation_preprocessor import AtariObservationPreprocessor
    preprocessor = AtariObservationPreprocessor(preprocessor_config)

    # setup gae
    buffer_config = load(*problem_config["gae"])
    buffer_config["agent_count"] = train_env.agent_count
    problem_config["gae"] = buffer_config
    from cgi_drl.data_storage.gae_sample_memory.gae_sample_memory import GaeSampleMemory
    gae_replay_buffer = GaeSampleMemory(buffer_config)

    # setup policy
    ppo_config = load(*problem_config["ppo"])
    ppo_config["action_space"] = train_env.get_action_space()
    problem_config["ppo"] = ppo_config
    from cgi_drl.decision_model.ppo.policy_trainer import PolicyTrainer
    policy = PolicyTrainer(ppo_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = AtariPpoSolver(problem_config)
    solver.environment = train_env
    solver.evaluation_environment = eval_env
    solver.gae_replay_buffer = gae_replay_buffer
    solver.observation_preprocessor = preprocessor
    solver.policy = policy

    solver.train()

class AtariPpoSolver(PpoSolver):
    def __init__(self, solver_config):
        super().__init__(solver_config)

    def episode_terminate(self, dones, is_valid_agent, is_train=True):
        if is_train:
            shared_log_str = "{}: {}/{}".format(self.episode_count, self.total_time_step, self.training_steps)
        else:
            shared_log_str = "Evaluation: {}/{}".format(self.total_time_step, self.training_steps)
        for i in range(self.get_agent_count(is_train)):
            if is_valid_agent[i] and dones[i]:
                log_str = shared_log_str + "-agent{} statistic| ".format(i)
                summary_list = []
                for key in self.agent_statistics[is_train][i]:
                    log_str = log_str + "{}: {}, ".format(key, self.agent_statistics[is_train][i][key])
                    prefix = "Agent Statistics/" if is_train else "Evaluation/Agent Statistics/"
                    summary_list.append(tf.compat.v1.Summary.Value(
                        tag= prefix + key, 
                        simple_value=self.agent_statistics[is_train][i][key]
                    ))
                agent_summary = tf.compat.v1.Summary(value=summary_list)
                self.summary_writer.add_summary(agent_summary, self.total_time_step)
                if is_train:
                    print(log_str)
                    self.log_file.write(log_str+'\n')
                else:
                    self.evaluation_log_file.write(log_str)
                    if len(self.evaluation_scores) == 0 or self.agent_statistics[is_train][i]["Cumulated Extrinsic Reward"] > max(self.evaluation_scores):
                        self.evaluation_best_observations = self.eval_trajectory_observations[i]
                    self.evaluation_scores.append(self.agent_statistics[is_train][i]["Cumulated Extrinsic Reward"])

    def evaluation(self):
        super().evaluation()
        self.observation_preprocessor.create_video(self.evaluation_best_observations, self.video_path + '/at_time_step{}_best_score_{}.mp4'.format(self.total_time_step, np.max(self.evaluation_scores)))

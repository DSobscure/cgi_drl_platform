from cgi_drl.problem.workflow_template.value_based_drl.c51_template_torch import C51Solver
import importlib
import numpy as np


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

    # setup experience replay
    buffer_config = load(*problem_config["experience_replay"])
    buffer_config["agent_count"] = train_env.agent_count
    problem_config["experience_replay"] = buffer_config
    if "is_prioritized" in buffer_config and buffer_config["is_prioritized"]:
        from cgi_drl.data_storage.experience_replay.prioritized_experience_replay import PrioritizedExperienceReplay
        replay_buffer = PrioritizedExperienceReplay(buffer_config)
        replay_buffer.is_prioritized = True
    else:
        from cgi_drl.data_storage.experience_replay.simple_experience_replay import SimpleExperienceReplay
        replay_buffer = SimpleExperienceReplay(buffer_config)
        replay_buffer.is_prioritized = False

    # setup policy
    c51_config = load(*problem_config["c51"])
    c51_config["action_space"] = train_env.get_action_space()
    problem_config["c51"] = c51_config
    from cgi_drl.decision_model.c51.policy_trainer_torch import PolicyTrainer
    policy = PolicyTrainer(c51_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = AtariC51Solver(problem_config)
    solver.environment = train_env
    solver.evaluation_environment = eval_env
    solver.replay_buffer  = replay_buffer
    solver.observation_preprocessor = preprocessor
    solver.policy = policy

    solver.train()

class AtariC51Solver(C51Solver):
    def __init__(self, solver_config):
        super().__init__(solver_config)

    def evaluation(self):
        super().evaluation()
        self.observation_preprocessor.create_video(self.evaluation_best_observations, self.video_path + '/at_time_step{}_best_score_{}.mp4'.format(self.total_time_step, np.max(self.evaluation_scores)))

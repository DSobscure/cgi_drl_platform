from cgi_drl.problem.workflow_template.ppo_template_torch import PpoSolver
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
    from cgi_drl.decision_model.ppo.policy_trainer_torch import PolicyTrainer
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

    def evaluation(self):
        super().evaluation()
        self.observation_preprocessor.create_video(self.evaluation_best_observations, self.video_path + '/at_time_step{}_best_score_{}.mp4'.format(self.total_time_step, np.max(self.evaluation_scores)))

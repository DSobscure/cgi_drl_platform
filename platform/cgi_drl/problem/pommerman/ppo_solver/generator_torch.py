from cgi_drl.problem.workflow_template.ppo_template_torch import PpoSolver
from tensorboardX import SummaryWriter
from collections import defaultdict

class PommermanPpoSolver(PpoSolver):
    def __init__(self, solver_config):
        load = solver_config["load_function"]
        
        preprocessor_config = load(*solver_config["observation_preprocessor"])
        solver_config["observation_preprocessor"] = preprocessor_config

        buffer_config = load(*solver_config["demonstration_memory"])
        solver_config["demonstration_memory"] = buffer_config

        ppo_config = load(*solver_config["ppo"])
        solver_config["ppo"] = ppo_config

        # version path
        self.run_id = solver_config.get("run_id", "")
        self.version_path = solver_config["version"] + solver_config.get("run_id", "")
        self.log_path = self._create_path(self.version_path, "log")
        self.model_path = self._create_path(self.version_path, "model")
        self.video_path = self._create_path(self.log_path, "video")
        
        # training parameter
        self.training_steps = solver_config["training_steps"]
        
        self.solver_config = solver_config

        # logs
        self.summary_writer = SummaryWriter(self.log_path)
        self.log_file = self._open_log(self.log_path, "log.txt")
        self.evaluation_log_file = self._open_log(self.log_path, "evaluation_log_file.txt")

        # setup policy
        self.policy_config = solver_config["ppo"]

        self.max_game_step = solver_config["max_game_step"]
        self.evaluation_max_game_step = solver_config["evaluation_max_game_step"]
        self.agent_statistics_aggregator = solver_config["agent_statistics_aggregator"]
        
        self.reward_transformer = solver_config["reward_transformer"]

        self.evaluation_scores = []
        
        self.total_time_step = self.solver_config.get("initial_time_step", 0)

        self.exploration_bonus_coefficient_scheduler = solver_config["exploration_bonus_coefficient_scheduler"]
        self.is_initialized = False
        self.demonstration_file_name = solver_config["demonstration_file_name"]

    def initialize(self, **kwargs):
        if not self.is_initialized:
            from cgi_drl.environment.pommerman.pommerman_observation_preprocessor import PommermanObservationPreprocessor
            self.observation_preprocessor = PommermanObservationPreprocessor(self.solver_config["observation_preprocessor"])
            
            from cgi_drl.data_storage.demonstration_memory.simple_demonstration_memory import SimpleDemonstrationMemory
            self.demonstration_memory = SimpleDemonstrationMemory(self.solver_config["demonstration_memory"])

            self.solver_config["ppo"]["action_space"] = [6]
            from cgi_drl.decision_model.ppo.policy_trainer_torch import PolicyTrainer
            self.policy = PolicyTrainer(self.solver_config["ppo"])
            
            # load policy
            if self.solver_config.get("is_load_policy", False):
                from os.path import basename
                model_path = self.solver_config.get("load_policy_model_path", self.model_path)
                self.policy.load(path=model_path)
                print("load policy model from {}".format(model_path))

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
            self.is_initialized = True
            
    def terminate(self):
        self.demonstration_memory.save(self.log_path + f"/{self.demonstration_file_name}")
        super().terminate()

    def episode_terminate(self, dones, is_valid_agent, is_train=True):
        pass

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
                
        for i in range(self.get_agent_count(is_train)):
            self.demonstration_memory.save_experience(
                observation_visual = self.observations[is_train]["observation_visual"][i],
                observation_vector = self.observations[is_train]["observation_vector"][i],
                action = actions[i],
                reward = rewards[i],
                done = dones[i],
                next_observation_visual = _next_observations["observation_visual"][i],
                next_observation_vector = _next_observations["observation_vector"][i]
            )

        self.observations[is_train] = _next_observations
        return dones

    def save_model(self):
        pass

    def save_to_agent_pool(self, agent_pool_path):
        pass

    def load_from_agent_pool(self, agent_pool_path):
        self.policy.load(agent_pool_path + "/model_latest.ckpt")

    def load_model(self, is_from_file=False):
        if is_from_file:
            model_path = self.solver_config.get("load_policy_model_path", self.model_path)
            self.policy.load(path=model_path)
        else:
            self.policy.load(self.model_path)
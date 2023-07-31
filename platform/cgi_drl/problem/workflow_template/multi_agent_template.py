import importlib
from cgi_drl.multi_agent_system.multi_agent_race_schedule import MultiAgentRaceSchedule
import tensorflow as tf
from cgi_drl.multi_agent_system.agent_pool.agent_pool_manager import AgentPoolManager

class MultiAgentTrainer:
    @staticmethod
    def _create_path(parent, dname):
        import os
        dpath = os.path.join(parent, dname)
        if not os.path.exists(dpath):
            os.makedirs(dpath)
        return dpath

    @staticmethod
    def _open_log(path, name):
        import os.path as osp
        fname = osp.join(path, name)
        print(strftime("%Y-%m-%d %H:%M:%S"), "create", fname, "log file")
        return open(fname, mode="a", buffering=1)

    def __init__(self, problem_config):
        # version path
        self.run_id = problem_config.get("run_id", "")
        self.version = problem_config["version"]
        self.version_path = problem_config["version"] + problem_config.get("run_id", "")
        self.log_path = self._create_path(self.version_path, "log")
        self.video_path = self._create_path(self.log_path, "video")
        self.training_varaibles_path = self._create_path(self.version_path, "training_varaibles")
        self.multi_agent_race_schedule = MultiAgentRaceSchedule(problem_config["agent_mapping"])

        solver_config = problem_config["multi_agent_solver"]
        self.load_config_function = problem_config["load_function"]
        self.solver_config = solver_config
        self.solver_config["single_solvers"] = problem_config["single_solvers"]

        self.curriculum_plan = problem_config["curriculum_plan"]
        self.training_phase_statistics_aggregator = problem_config["training_phase_statistics_aggregator"]
        self.evaluation_phase_statistics_aggregator = problem_config["evaluation_phase_statistics_aggregator"]
        if "stage_phase_statistics_aggregator" in problem_config:
            self.stage_phase_statistics_aggregator = problem_config["stage_phase_statistics_aggregator"]
        else:
            self.stage_phase_statistics_aggregator = lambda x: {}

        # agent pool manager
        agent_pool_manager_config = self.load_config_function(*problem_config["agent_pool_manager"])
        self.agent_pool_manager = AgentPoolManager(self.version_path, agent_pool_manager_config)

        self.summary_writer = tf.compat.v1.summary.FileWriter(self.log_path)

    def run_workflow(self):
        self.initialize()
        is_end = False
        while not is_end:
            training_phase_results = self.run_training_phase()
            evaluation_phase_results = self.run_evaluation_phase()
            is_end = self.summarize(training_phase_results, evaluation_phase_results)
        self.terminate()

    def initialize(self):
        # initialize solvers
        self.single_solvers = {}
        self.sampled_anchor = {}

        first_agent_name = self.environment.agent_names[0]
        for name, config in self.solver_config["single_solvers"].items():
            SolverClass = getattr(importlib.import_module(config["solver"][0]), config["solver"][1])

            with tf.Graph().as_default() as g:
                solver_config = self.load_config_function(*config["config"])
                solver_config["load_function"] = self.load_config_function
                solver_config["run_id"] = self.run_id
                solver_config["version"] = self.version
                solver = SolverClass(solver_config)
                self.single_solvers[name] = solver
                solver.environment = self.environment.sub_environments[first_agent_name]
                solver.evaluation_environment = self.evaluation_environment.sub_environments[first_agent_name]
                solver.initialize(tf_graph=g)     

            if "static_solver" in config:
                StaticSolverClass = getattr(importlib.import_module(config["static_solver"][0]), config["static_solver"][1])
                # Create static solver which is served as the opponent with the old model
                # The statistics will not be logged
                with tf.Graph().as_default() as g:
                    solver_config = self.load_config_function(*config["config"])
                    solver_config["load_function"] = self.load_config_function
                    solver_config["run_id"] = self.run_id
                    solver_config["version"] = self.version
                    solver = StaticSolverClass(solver_config)
                    self.single_solvers[name + AgentPoolManager.from_agent_pool_mark] = solver
                    solver.environment = self.environment.sub_environments[first_agent_name]
                    solver.evaluation_environment = self.evaluation_environment.sub_environments[first_agent_name]
                    solver.initialize(tf_graph=g)    

            if "shared_experience" in config:
                shared_solver = self.single_solvers[config["shared_experience"]]
                self.single_solvers[name].share_experience(shared_solver)
                
            # Other solver config
            self.sampled_anchor[name] = config.get("anchor", None)
        # create agent mapping
        self.subphase_counts = self.multi_agent_race_schedule.get_subphase_counts()

        self.current_stage = 0
        if "start_stage" in self.solver_config:
            self.current_stage = self.solver_config["start_stage"]
        self.current_iteration_count = 0
        if "start_iteration" in self.solver_config:
            self.current_iteration_count = self.solver_config["start_iteration"]
        self.training_iteration_count = self.solver_config["training_iteration_count"]

        self.agent_sampling_scheme = self.solver_config["agent_sampling_scheme"]
        self.change_agent_step_period = self.solver_config["change_agent_step_period"]
        self.subphase_step_period = self.solver_config["subphase_step_period"]

        # add agents to the agent pool
        for name, solver in self.single_solvers.items():
            if not AgentPoolManager.is_from_agent_pool(name):
                if hasattr(solver, "save_model"):
                    solver.save_model()
                self.agent_pool_manager.add_agent_to_pool(
                    full_name=name,
                    series_name=name,
                    save_model_callback=solver.save_to_agent_pool,
                    is_static=True
                )

    def matchmaking(self, agent_name_solver_name_mapping, scheme):
        """Sample the agent if needed.
        It will also load the model to the corresponding solver.
        """
        participant_solver_agent_metas = {}
        for agent_name, solver_name in agent_name_solver_name_mapping.items():
            if AgentPoolManager.is_from_agent_pool(solver_name):
                raw_name = AgentPoolManager.remove_from_agent_pool_mark(solver_name)
                if self.agent_pool_manager.get_pool_size(raw_name) > 0:
                    if scheme == "uniform":
                        agent_meta = self.agent_pool_manager.sample_agent_from_pool_uniformly(raw_name)
                    elif scheme == "prioritized":
                        agent_meta = self.agent_pool_manager.sample_agent_with_prioritized_fictitious_self_play(self.sampled_anchor[raw_name], raw_name)
                else:
                    agent_meta = self.agent_pool_manager.get_agent_meta(raw_name)
            else:
                agent_meta = self.agent_pool_manager.get_agent_meta(solver_name)
            participant_solver_agent_metas[solver_name] = agent_meta

        return participant_solver_agent_metas

    def run_training_phase(self):
        training_phase_results = {}

        for subphase in range(self.subphase_counts["train"][self.current_stage]):
            race_schedule = self.multi_agent_race_schedule.get_race_participant_mapping(
                mode="train", 
                stage=self.current_stage, 
                subphase=subphase
            )

            step_count = 0
            next_matchmaking_step = 0
            while step_count < self.subphase_step_period["train"]:
                if step_count >= next_matchmaking_step:
                    next_matchmaking_step = step_count + self.change_agent_step_period["train"]
                    participant_solver_agent_metas = self.matchmaking(race_schedule, self.agent_sampling_scheme["train"])

                    participant_solvers = {
                        agent_name: self.single_solvers[solver_name]
                        for agent_name, solver_name in race_schedule.items()
                    }

                    for agent_name, solver in participant_solvers.items():
                        solver.environment = self.environment.sub_environments[agent_name]
                        solver.evaluation_environment = self.evaluation_environment.sub_environments[agent_name]  
                        solver_name = race_schedule[agent_name]
                        if AgentPoolManager.is_from_agent_pool(solver_name):
                            solver.terminate()
                            with tf.Graph().as_default() as g:
                                solver.initialize(tf_graph=g)   
                            solver.load_from_agent_pool(participant_solver_agent_metas[solver_name].model_directory)
                        else:
                            if hasattr(solver, "save_model"):
                                solver.save_model()
                            solver.terminate()
                            with tf.Graph().as_default() as g:
                                solver.initialize(tf_graph=g)   
                            if hasattr(solver, "load_model"):
                                solver.load_model()

                    game_dones = [True for _ in range(self.environment.environment_count)]
                    solver_dones = {
                        agent_name: [True for _ in range(self.environment.sub_environments[agent_name].get_agent_count())]
                        for agent_name in participant_solvers
                    }

                for i in range(self.environment.environment_count):
                    if game_dones[i]:
                        self.environment.reset_game(i)
                        game_dones[i] = False

                while not any(game_dones):
                    decisions = {}
                    for agent_name, single_solver in participant_solvers.items():
                        is_valid_agent = self.environment.agent_turn[agent_name]
                        if any(solver_dones[agent_name]):
                            solver_dones[agent_name] = single_solver.episode_initiate(solver_dones[agent_name], is_valid_agent)
                        decisions[agent_name] = single_solver.decide_agent_actions(is_valid_agent)
                    game_dones, game_infos = self.environment.step({
                        agent_name: decisions[agent_name]["actions"]
                        for agent_name in participant_solvers
                    })
                    self.training_phase_statistics_aggregator(training_phase_results, race_schedule, game_dones, game_infos)
                    for agent_name, single_solver in participant_solvers.items():
                        is_valid_agent = self.environment.agent_turn[agent_name]
                        solver_dones[agent_name] = single_solver.on_time_step(decisions[agent_name], is_valid_agent)
                        single_solver.total_time_step += sum(is_valid_agent)
                        step_count += sum(is_valid_agent)
                        if any(solver_dones[agent_name]):
                            single_solver.episode_terminate(solver_dones[agent_name], is_valid_agent)

        return training_phase_results

    def run_evaluation_phase(self):
        evaluation_phase_results = {}

        for subphase in range(self.subphase_counts["eval"][self.current_stage]):
            race_schedule = self.multi_agent_race_schedule.get_race_participant_mapping(
                mode="eval", 
                stage=self.current_stage, 
                subphase=subphase
            )

            step_count = 0
            next_matchmaking_step = 0
            while step_count < self.subphase_step_period["eval"]:
                if step_count >= next_matchmaking_step:
                    next_matchmaking_step = step_count + self.change_agent_step_period["eval"]
                    participant_solver_agent_metas = self.matchmaking(race_schedule, self.agent_sampling_scheme["eval"])

                    participant_solvers = {
                        agent_name: self.single_solvers[solver_name]
                        for agent_name, solver_name in race_schedule.items()
                    }

                    for agent_name, solver in participant_solvers.items():
                        solver.environment = self.environment.sub_environments[agent_name]
                        solver.evaluation_environment = self.evaluation_environment.sub_environments[agent_name]  
                        solver_name = race_schedule[agent_name]
                        if AgentPoolManager.is_from_agent_pool(solver_name):
                            solver.terminate()
                            with tf.Graph().as_default() as g:
                                solver.initialize(tf_graph=g)   
                            solver.load_from_agent_pool(participant_solver_agent_metas[solver_name].model_directory)
                        else:
                            if hasattr(solver, "save_model"):
                                solver.save_model()
                            solver.terminate()
                            with tf.Graph().as_default() as g:
                                solver.initialize(tf_graph=g)   
                            if hasattr(solver, "load_model"):
                                solver.load_model()

                    game_dones = [True for _ in range(self.evaluation_environment.environment_count)]
                    solver_dones = {
                        agent_name: [True for _ in range(self.evaluation_environment.sub_environments[agent_name].get_agent_count())]
                        for agent_name in participant_solvers
                    }

                for i in range(self.evaluation_environment.environment_count):
                    if game_dones[i]:
                        self.evaluation_environment.reset_game(i)
                        game_dones[i] = False

                while not any(game_dones):
                    decisions = {}
                    for agent_name, single_solver in participant_solvers.items():
                        is_valid_agent = self.evaluation_environment.agent_turn[agent_name]
                        if any(solver_dones[agent_name]):
                            solver_dones[agent_name] = single_solver.episode_initiate(solver_dones[agent_name], is_valid_agent, is_train=False)
                        decisions[agent_name] = single_solver.decide_agent_actions(is_valid_agent, is_train=False)
                    game_dones, game_infos = self.evaluation_environment.step({
                        agent_name: decisions[agent_name]["actions"]
                        for agent_name in participant_solvers
                    })
                    self.evaluation_phase_statistics_aggregator(evaluation_phase_results, race_schedule, game_dones, game_infos)
                    for agent_name, single_solver in participant_solvers.items():
                        is_valid_agent = self.evaluation_environment.agent_turn[agent_name]
                        solver_dones[agent_name] = single_solver.on_time_step(decisions[agent_name], is_valid_agent, is_train=False)
                        step_count += sum(is_valid_agent)
                        if any(solver_dones[agent_name]):
                            single_solver.episode_terminate(solver_dones[agent_name], is_valid_agent, is_train=False)
                
                for i in range(self.evaluation_environment.environment_count):
                    if game_dones[i]:
                        score_dict = {}
                        for info in game_infos[i]:
                            solver_name = participant_solver_agent_metas[race_schedule[info["agent_name"]]].full_name
                            score_dict[solver_name] = info["result"]
                        print(self.agent_pool_manager.update_elo_rating(score_dict))   

        return evaluation_phase_results

    def summarize(self, training_phase_results, evaluation_phase_results):
        summary_list = []
        summary_list.append(tf.compat.v1.Summary.Value(
            tag= "Multi-Agent/Curriculum Stage", 
            simple_value=self.current_stage
        ))
        for key in training_phase_results:
            summary_list.append(tf.compat.v1.Summary.Value(
                tag= "Multi-Agent/Training/" + key, 
                simple_value=training_phase_results[key]
            ))
        for key in evaluation_phase_results:
            summary_list.append(tf.compat.v1.Summary.Value(
                tag= "Multi-Agent/Evaluation/" + key, 
                simple_value=evaluation_phase_results[key]
            ))
        phase_results = self.stage_phase_statistics_aggregator({
            "current_stage": self.current_stage,
            "current_iteration_count": self.current_iteration_count,
            "training_phase_results": training_phase_results,
            "evaluation_phase_results": evaluation_phase_results,
        })
        for key in phase_results:
            summary_list.append(tf.compat.v1.Summary.Value(
                tag= "Multi-Agent/" + key, 
                simple_value=phase_results[key]
            ))

        agent_summary = tf.compat.v1.Summary(value=summary_list)
        self.summary_writer.add_summary(agent_summary, self.current_iteration_count)

        self.current_stage = self.curriculum_plan({
            "current_stage": self.current_stage,
            "current_iteration_count": self.current_iteration_count,
            "training_phase_results": training_phase_results,
            "evaluation_phase_results": evaluation_phase_results,
        })
        self.save_agents()
        self.current_iteration_count += 1
        return self.current_iteration_count >= self.training_iteration_count

    def terminate(self):
        self.environment.close()
        self.evaluation_environment.close()   

    def save_agents(self):
        for solver_name, solver in self.single_solvers.items():
            if hasattr(solver, "save_model"):
                solver.save_model()
                if not AgentPoolManager.is_from_agent_pool(solver_name):
                    new_solver_name = solver_name + f"_{self.current_iteration_count}"
                    self.agent_pool_manager.add_agent_to_pool(
                        full_name=new_solver_name,
                        series_name=solver_name,
                        save_model_callback=solver.save_to_agent_pool,
                        rating=self.agent_pool_manager.get_agent_meta(solver_name).rating,
                    )
                    summary_list = []
                    summary_list.append(tf.compat.v1.Summary.Value(
                        tag= "Multi-Agent/Rating/" + solver_name, 
                        simple_value=self.agent_pool_manager.get_agent_meta(new_solver_name).rating
                    ))
                    agent_summary = tf.compat.v1.Summary(value=summary_list)
                    self.summary_writer.add_summary(agent_summary, self.current_iteration_count)


class MultiAgentEvaluator:
    @staticmethod
    def _create_path(parent, dname):
        import os
        dpath = os.path.join(parent, dname)
        if not os.path.exists(dpath):
            os.makedirs(dpath)
        return dpath

    @staticmethod
    def _open_log(path, name):
        import os.path as osp
        fname = osp.join(path, name)
        print(strftime("%Y-%m-%d %H:%M:%S"), "create", fname, "log file")
        return open(fname, mode="a", buffering=1)

    def __init__(self, problem_config):
        # version path
        self.run_id = problem_config.get("run_id", "")
        self.version = problem_config["version"]
        self.version_path = problem_config["version"] + problem_config.get("run_id", "")
        self.log_path = self._create_path(self.version_path, "log")
        self.video_path = self._create_path(self.log_path, "video")
        self.training_varaibles_path = self._create_path(self.version_path, "training_varaibles")

        self.multi_agent_race_schedule = MultiAgentRaceSchedule(problem_config["agent_mapping"])

        solver_config = problem_config["multi_agent_solver"]
        self.load_config_function = problem_config["load_function"]
        self.solver_config = solver_config
        self.solver_config["single_solvers"] = problem_config["single_solvers"]

        self.curriculum_plan = problem_config["curriculum_plan"]
        self.evaluation_phase_statistics_aggregator = problem_config["evaluation_phase_statistics_aggregator"]

        # agent pool manager
        agent_pool_manager_config = self.load_config_function(*problem_config["agent_pool_manager"])
        self.agent_pool_manager = AgentPoolManager(self.version_path, agent_pool_manager_config)

        self.summary_writer = tf.compat.v1.summary.FileWriter(self.log_path)

    def run_workflow(self):
        self.initialize()
        evaluation_phase_results = self.run_evaluation_phase()
        self.summarize(evaluation_phase_results)
        self.terminate()

    def initialize(self):
        # initialize solvers
        self.single_solvers = {}
        self.sampled_anchor = {}

        first_agent_name = self.environment.agent_names[0]
        for name, config in self.solver_config["single_solvers"].items():
            SolverClass = getattr(importlib.import_module(config["solver"][0]), config["solver"][1])

            with tf.Graph().as_default() as g:
                solver_config = self.load_config_function(*config["config"])
                solver_config["load_function"] = self.load_config_function
                solver_config["run_id"] = self.run_id
                solver_config["version"] = self.version
                solver = SolverClass(solver_config)
                self.single_solvers[name] = solver
                solver.environment = self.environment.sub_environments[first_agent_name]
                solver.evaluation_environment = self.evaluation_environment.sub_environments[first_agent_name]
                solver.initialize(tf_graph=g)     

            # Other solver config
            self.sampled_anchor[name] = config.get("anchor", None)
        # create agent mapping
        self.subphase_counts = self.multi_agent_race_schedule.get_subphase_counts()

        self.current_stage = 0

        self.agent_sampling_scheme = self.solver_config["agent_sampling_scheme"]
        self.change_agent_step_period = self.solver_config["change_agent_step_period"]
        self.subphase_step_period = self.solver_config["subphase_step_period"]

        # add agents to the agent pool
        for name, solver in self.single_solvers.items():
            if not AgentPoolManager.is_from_agent_pool(name):
                self.agent_pool_manager.add_agent_to_pool(
                    full_name=name,
                    series_name=name,
                    save_model_callback=lambda x: x,
                    is_static=True
                )

    def matchmaking(self, agent_name_solver_name_mapping, scheme):
        """Sample the agent if needed.
        It will also load the model to the corresponding solver.
        """
        participant_solver_agent_metas = {}
        for agent_name, solver_name in agent_name_solver_name_mapping.items():
            if AgentPoolManager.is_from_agent_pool(solver_name):
                raw_name = AgentPoolManager.remove_from_agent_pool_mark(solver_name)
                if self.agent_pool_manager.get_pool_size(raw_name) > 0:
                    if scheme == "uniform":
                        agent_meta = self.agent_pool_manager.sample_agent_from_pool_uniformly(raw_name)
                    elif scheme == "prioritized":
                        agent_meta = self.agent_pool_manager.sample_agent_with_prioritized_fictitious_self_play(self.sampled_anchor[raw_name], raw_name)
                else:
                    agent_meta = self.agent_pool_manager.get_agent_meta(raw_name)
            else:
                agent_meta = self.agent_pool_manager.get_agent_meta(solver_name)
            participant_solver_agent_metas[solver_name] = agent_meta

        return participant_solver_agent_metas

    def run_evaluation_phase(self):
        evaluation_phase_results = {}

        for subphase in range(self.subphase_counts["eval"][self.current_stage]):
            print("Evaluation: Start subphase {}".format(subphase))
            race_schedule = self.multi_agent_race_schedule.get_race_participant_mapping(
                mode="eval", 
                stage=self.current_stage, 
                subphase=subphase
            )

            step_count = 0
            next_matchmaking_step = 0
            while step_count < self.subphase_step_period["eval"]:
                if step_count >= next_matchmaking_step:
                    next_matchmaking_step = step_count + self.change_agent_step_period["eval"]
                    participant_solver_agent_metas = self.matchmaking(race_schedule, self.agent_sampling_scheme["eval"])

                    participant_solvers = {
                        agent_name: self.single_solvers[solver_name]
                        for agent_name, solver_name in race_schedule.items()
                    }

                    for agent_name, solver in participant_solvers.items():
                        solver.environment = self.environment.sub_environments[agent_name]
                        solver.evaluation_environment = self.evaluation_environment.sub_environments[agent_name]  
                        solver_name = race_schedule[agent_name]
                        if AgentPoolManager.is_from_agent_pool(solver_name):
                            solver.terminate()
                            with tf.Graph().as_default() as g:
                                solver.initialize(tf_graph=g)   
                            solver.load_from_agent_pool(participant_solver_agent_metas[solver_name].model_directory)
                        else:
                            solver.terminate()
                            with tf.Graph().as_default() as g:
                                solver.initialize(tf_graph=g)   
                            if hasattr(solver, "load_model"):
                                solver.load_model(True)

                    game_dones = [True for _ in range(self.evaluation_environment.environment_count)]
                    solver_dones = {
                        agent_name: [True for _ in range(self.evaluation_environment.sub_environments[agent_name].get_agent_count())]
                        for agent_name in participant_solvers
                    }

                for i in range(self.evaluation_environment.environment_count):
                    if game_dones[i]:
                        self.evaluation_environment.reset_game(i)
                        game_dones[i] = False

                while not any(game_dones):
                    decisions = {}
                    for agent_name, single_solver in participant_solvers.items():
                        is_valid_agent = self.evaluation_environment.agent_turn[agent_name]
                        if any(solver_dones[agent_name]):
                            solver_dones[agent_name] = single_solver.episode_initiate(solver_dones[agent_name], is_valid_agent, is_train=False)
                        decisions[agent_name] = single_solver.decide_agent_actions(is_valid_agent, is_train=False)
                    game_dones, game_infos = self.evaluation_environment.step({
                        agent_name: decisions[agent_name]["actions"]
                        for agent_name in participant_solvers
                    })
                    self.evaluation_phase_statistics_aggregator(evaluation_phase_results, race_schedule, game_dones, game_infos)
                    for agent_name, single_solver in participant_solvers.items():
                        is_valid_agent = self.evaluation_environment.agent_turn[agent_name]
                        solver_dones[agent_name] = single_solver.on_time_step(decisions[agent_name], is_valid_agent, is_train=False)
                        step_count += sum(is_valid_agent)
                        if any(solver_dones[agent_name]):
                            single_solver.episode_terminate(solver_dones[agent_name], is_valid_agent, is_train=False)
                
                for i in range(self.evaluation_environment.environment_count):
                    if game_dones[i]:
                        score_dict = {}
                        for info in game_infos[i]:
                            solver_name = participant_solver_agent_metas[race_schedule[info["agent_name"]]].full_name
                            score_dict[solver_name] = info["result"]
                        print(self.agent_pool_manager.update_elo_rating(score_dict))   

        return evaluation_phase_results

    def summarize(self, evaluation_phase_results):
        summary_list = []
        for key in evaluation_phase_results:
            summary_list.append(tf.compat.v1.Summary.Value(
                tag= "Multi-Agent/Evaluation/" + key, 
                simple_value=evaluation_phase_results[key]
            ))
        agent_summary = tf.compat.v1.Summary(value=summary_list)
        self.summary_writer.add_summary(agent_summary, 0)

        self.current_stage = self.curriculum_plan({
            "evaluation_phase_results": evaluation_phase_results,
        })

    def terminate(self):
        self.environment.close()   
        self.evaluation_environment.close()   

class MultiAgentEvaluator2:
    @staticmethod
    def _create_path(parent, dname):
        import os
        dpath = os.path.join(parent, dname)
        if not os.path.exists(dpath):
            os.makedirs(dpath)
        return dpath

    @staticmethod
    def _open_log(path, name):
        import os.path as osp
        fname = osp.join(path, name)
        print(strftime("%Y-%m-%d %H:%M:%S"), "create", fname, "log file")
        return open(fname, mode="a", buffering=1)

    def __init__(self, problem_config):
        # version path
        self.run_id = problem_config.get("run_id", "")
        self.version = problem_config["version"]
        self.version_path = problem_config["version"] + problem_config.get("run_id", "")
        self.log_path = self._create_path(self.version_path, "log")
        self.video_path = self._create_path(self.log_path, "video")
        self.training_varaibles_path = self._create_path(self.version_path, "training_varaibles")

        self.multi_agent_race_schedule = MultiAgentRaceSchedule(problem_config["agent_mapping"])

        solver_config = problem_config["multi_agent_solver"]
        self.load_config_function = problem_config["load_function"]
        self.solver_config = solver_config
        self.solver_config["single_solvers"] = problem_config["single_solvers"]

        self.curriculum_plan = problem_config["curriculum_plan"]
        self.evaluation_phase_statistics_aggregator = problem_config["evaluation_phase_statistics_aggregator"]

        # agent pool manager
        agent_pool_manager_config = self.load_config_function(*problem_config["agent_pool_manager"])
        self.agent_pool_manager = AgentPoolManager(self.version_path, agent_pool_manager_config)

        self.summary_writer = tf.compat.v1.summary.FileWriter(self.log_path)

    def run_workflow(self):
        self.initialize()
        evaluation_phase_results = self.run_evaluation_phase()
        self.summarize(evaluation_phase_results)
        self.terminate()

    def initialize(self):
        # initialize solvers
        self.single_solvers = {}
        self.sampled_anchor = {}

        first_agent_name = self.environment.agent_names[0]
        for name, config in self.solver_config["single_solvers"].items():
            SolverClass = getattr(importlib.import_module(config["solver"][0]), config["solver"][1])

            with tf.Graph().as_default() as g:
                solver_config = self.load_config_function(*config["config"])
                solver_config["load_function"] = self.load_config_function
                solver_config["run_id"] = self.run_id
                solver_config["version"] = self.version
                solver = SolverClass(solver_config)
                self.single_solvers[name] = solver
                solver.environment = self.environment.sub_environments[first_agent_name]
                solver.evaluation_environment = self.evaluation_environment.sub_environments[first_agent_name]
                solver.initialize(tf_graph=g)     

            # Other solver config
            self.sampled_anchor[name] = config.get("anchor", None)
        # create agent mapping
        self.subphase_counts = self.multi_agent_race_schedule.get_subphase_counts()

        self.current_stage = 0

        self.agent_sampling_scheme = self.solver_config["agent_sampling_scheme"]
        self.change_agent_episode_count = self.solver_config["change_agent_episode_count"]
        self.subphase_episode_count = self.solver_config["subphase_episode_count"]

        # add agents to the agent pool
        for name, solver in self.single_solvers.items():
            if not AgentPoolManager.is_from_agent_pool(name):
                self.agent_pool_manager.add_agent_to_pool(
                    full_name=name,
                    series_name=name,
                    save_model_callback=lambda x: x,
                    is_static=True
                )

    def matchmaking(self, agent_name_solver_name_mapping, scheme):
        """Sample the agent if needed.
        It will also load the model to the corresponding solver.
        """
        participant_solver_agent_metas = {}
        for agent_name, solver_name in agent_name_solver_name_mapping.items():
            if AgentPoolManager.is_from_agent_pool(solver_name):
                raw_name = AgentPoolManager.remove_from_agent_pool_mark(solver_name)
                if self.agent_pool_manager.get_pool_size(raw_name) > 0:
                    if scheme == "uniform":
                        agent_meta = self.agent_pool_manager.sample_agent_from_pool_uniformly(raw_name)
                    elif scheme == "prioritized":
                        agent_meta = self.agent_pool_manager.sample_agent_with_prioritized_fictitious_self_play(self.sampled_anchor[raw_name], raw_name)
                else:
                    agent_meta = self.agent_pool_manager.get_agent_meta(raw_name)
            else:
                agent_meta = self.agent_pool_manager.get_agent_meta(solver_name)
            participant_solver_agent_metas[solver_name] = agent_meta

        return participant_solver_agent_metas

    def run_evaluation_phase(self):
        evaluation_phase_results = {}

        for subphase in range(self.subphase_counts["eval"][self.current_stage]):
            print("Evaluation: Start subphase {}".format(subphase))
            race_schedule = self.multi_agent_race_schedule.get_race_participant_mapping(
                mode="eval", 
                stage=self.current_stage, 
                subphase=subphase
            )

            episode_count = 0
            next_matchmaking_episode = 0
            while episode_count < self.subphase_episode_count["eval"]:
                if episode_count >= next_matchmaking_episode:
                    next_matchmaking_episode = episode_count + self.change_agent_episode_count["eval"]
                    participant_solver_agent_metas = self.matchmaking(race_schedule, self.agent_sampling_scheme["eval"])

                    participant_solvers = {
                        agent_name: self.single_solvers[solver_name]
                        for agent_name, solver_name in race_schedule.items()
                    }

                    for agent_name, solver in participant_solvers.items():
                        solver.environment = self.environment.sub_environments[agent_name]
                        solver.evaluation_environment = self.evaluation_environment.sub_environments[agent_name]  
                        solver_name = race_schedule[agent_name]
                        if AgentPoolManager.is_from_agent_pool(solver_name):
                            solver.terminate()
                            with tf.Graph().as_default() as g:
                                solver.initialize(tf_graph=g)   
                            solver.load_from_agent_pool(participant_solver_agent_metas[solver_name].model_directory)
                        else:
                            solver.terminate()
                            with tf.Graph().as_default() as g:
                                solver.initialize(tf_graph=g)   
                            if hasattr(solver, "load_model"):
                                solver.load_model(True)

                    game_dones = [True for _ in range(self.evaluation_environment.environment_count)]
                    solver_dones = {
                        agent_name: [True for _ in range(self.evaluation_environment.sub_environments[agent_name].get_agent_count())]
                        for agent_name in participant_solvers
                    }

                for i in range(self.evaluation_environment.environment_count):
                    if game_dones[i]:
                        self.evaluation_environment.reset_game(i)
                        game_dones[i] = False

                while not any(game_dones):
                    decisions = {}
                    for agent_name, single_solver in participant_solvers.items():
                        is_valid_agent = self.evaluation_environment.agent_turn[agent_name]
                        if any(solver_dones[agent_name]):
                            solver_dones[agent_name] = single_solver.episode_initiate(solver_dones[agent_name], is_valid_agent, is_train=False)
                        decisions[agent_name] = single_solver.decide_agent_actions(is_valid_agent, is_train=False)
                    game_dones, game_infos = self.evaluation_environment.step({
                        agent_name: decisions[agent_name]["actions"]
                        for agent_name in participant_solvers
                    })
                    self.evaluation_phase_statistics_aggregator(evaluation_phase_results, race_schedule, game_dones, game_infos)
                    for agent_name, single_solver in participant_solvers.items():
                        is_valid_agent = self.evaluation_environment.agent_turn[agent_name]
                        solver_dones[agent_name] = single_solver.on_time_step(decisions[agent_name], is_valid_agent, is_train=False)
                        if any(solver_dones[agent_name]):
                            single_solver.episode_terminate(solver_dones[agent_name], is_valid_agent, is_train=False)
                
                for i in range(self.evaluation_environment.environment_count):
                    if game_dones[i]:
                        episode_count += 1
                        score_dict = {}
                        for info in game_infos[i]:
                            solver_name = participant_solver_agent_metas[race_schedule[info["agent_name"]]].full_name
                            score_dict[solver_name] = info["result"]
                        print(self.agent_pool_manager.update_elo_rating(score_dict))   

        return evaluation_phase_results

    def summarize(self, evaluation_phase_results):
        summary_list = []
        for key in evaluation_phase_results:
            summary_list.append(tf.compat.v1.Summary.Value(
                tag= "Multi-Agent/Evaluation/" + key, 
                simple_value=evaluation_phase_results[key]
            ))
        agent_summary = tf.compat.v1.Summary(value=summary_list)
        self.summary_writer.add_summary(agent_summary, 0)

        self.current_stage = self.curriculum_plan({
            "evaluation_phase_results": evaluation_phase_results,
        })

    def terminate(self):
        self.environment.close()   
        self.evaluation_environment.close()   
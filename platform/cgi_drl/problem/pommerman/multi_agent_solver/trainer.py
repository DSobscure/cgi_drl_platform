from cgi_drl.problem.workflow_template.multi_agent_template import MultiAgentTrainer
import importlib
import numpy as np
import tensorflow as tf
from cgi_drl.multi_agent_system.agent_pool.agent_pool_manager import AgentPoolManager
import time

def launch(problem_config):
    load = problem_config["load_function"]

    # setup environment
    env_config = load(*problem_config["environment"])
    evaluation_env_config = load(*problem_config["evaluation_environment"])
    problem_config["environment"] = env_config
    problem_config["evaluation_environment"] = evaluation_env_config

    solver = PommermanMultiAgentSolver(problem_config)

    from cgi_drl.environment.multi_agent_environment_broker import MultiAgentEnvironmentBroker
    train_env = MultiAgentEnvironmentBroker(env_config)
    eval_env = MultiAgentEnvironmentBroker(evaluation_env_config)
    solver.environment = train_env
    solver.evaluation_environment = eval_env
    
    solver.run_workflow()

class PommermanMultiAgentSolver(MultiAgentTrainer):
    def __init__(self, solver_config):
        super().__init__(solver_config)

    def run_training_phase(self):
        training_phase_results = {}

        for subphase in range(self.subphase_counts["train"][self.current_stage]):
            race_schedule = self.multi_agent_race_schedule.get_race_participant_mapping(
                mode="train", 
                stage=self.current_stage, 
                subphase=subphase
            )
            race_config = self.multi_agent_race_schedule.get_race_config(
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
                        solver_name = race_schedule[agent_name]
                        if AgentPoolManager.is_from_agent_pool(solver_name):
                            solver.terminate()
                            with tf.Graph().as_default() as g:
                                solver.initialize(tf_graph=g, stage=self.current_stage)
                            solver.load_from_agent_pool(participant_solver_agent_metas[solver_name].model_directory)
                        else:
                            if hasattr(solver, "save_model"):
                                solver.save_model()
                            solver.terminate()
                            with tf.Graph().as_default() as g:
                                solver.initialize(tf_graph=g, stage=self.current_stage)
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
                for i in range(self.environment.environment_count):
                    if game_dones[i]:
                        for agent_name, single_solver in participant_solvers.items():
                            is_valid_agent = self.environment.agent_turn[agent_name]
                            if any(solver_dones[agent_name]):
                                solver_dones[agent_name] = single_solver.episode_initiate(solver_dones[agent_name], is_valid_agent)
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
                    need_restarts = self.environment.need_restart()
                    for i in range(self.environment.environment_count):
                        if need_restarts[i]:
                            self.environment.restart(i)
                            game_dones[i] = True
                            for agent_name in solver_dones:
                                solver_dones[agent_name][i] = True
                # for i in range(self.environment.environment_count):
                #     if game_dones[i]:
                #         score_dict = {}
                #         for info in game_infos[i]:
                #             if "result" in info:
                #                 solver_name = participant_solver_agent_metas[race_schedule[info["agent_name"]]].full_name
                #                 score_dict[solver_name] = info["result"]
                #         if len(score_dict) > 0:
                #             print(self.agent_pool_manager.update_elo_rating(score_dict))   

        return training_phase_results

    def run_evaluation_phase(self):
        evaluation_phase_results = {}

        for subphase in range(self.subphase_counts["eval"][self.current_stage]):
            has_opening_AI = False
            race_schedule = self.multi_agent_race_schedule.get_race_participant_mapping(
                mode="eval", 
                stage=self.current_stage, 
                subphase=subphase
            )
            race_config = self.multi_agent_race_schedule.get_race_config(
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
                for i in range(self.evaluation_environment.environment_count):
                    if game_dones[i]:
                        for agent_name, single_solver in participant_solvers.items():
                            is_valid_agent = self.evaluation_environment.agent_turn[agent_name]
                            if any(solver_dones[agent_name]):
                                solver_dones[agent_name] = single_solver.episode_initiate(solver_dones[agent_name], is_valid_agent, is_train=False)
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
                    need_restarts = self.evaluation_environment.need_restart()
                    for i in range(self.evaluation_environment.environment_count):
                        if need_restarts[i]:
                            self.evaluation_environment.restart(i)
                            game_dones[i] = True
                            for agent_name in solver_dones:
                                solver_dones[agent_name][i] = True
                
                for i in range(self.evaluation_environment.environment_count):
                    if game_dones[i]:
                        score_dict = {}
                        for info in game_infos[i]:
                            if "result" in info:
                                solver_name = participant_solver_agent_metas[race_schedule[info["agent_name"]]].full_name
                                score_dict[solver_name] = info["result"]
                        if len(score_dict) > 0 and not has_opening_AI:
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

    def run_workflow(self):
        self.initialize()
        is_end = False
        while not is_end:
            training_phase_results = self.run_training_phase()
            evaluation_phase_results = self.run_evaluation_phase()
            is_end = self.summarize(training_phase_results, evaluation_phase_results)
        self.terminate()
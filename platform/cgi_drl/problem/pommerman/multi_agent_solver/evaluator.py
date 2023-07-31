from cgi_drl.problem.workflow_template.multi_agent_template import MultiAgentEvaluator2
import importlib
import numpy as np
import tensorflow as tf
from cgi_drl.multi_agent_system.agent_pool.agent_pool_manager import AgentPoolManager
import time

def launch(problem_config):
    load = problem_config["load_function"]

    # setup environment
    evaluation_env_config = load(*problem_config["evaluation_environment"])
    problem_config["evaluation_environment"] = evaluation_env_config

    solver = PommermanMultiAgentEvaluator(problem_config)

    from cgi_drl.environment.multi_agent_environment_broker import MultiAgentEnvironmentBroker
    eval_env = MultiAgentEnvironmentBroker(evaluation_env_config)
    solver.environment = eval_env
    solver.evaluation_environment = eval_env
    
    solver.run_workflow()

class PommermanMultiAgentEvaluator(MultiAgentEvaluator2):
    def __init__(self, solver_config):
        super().__init__(solver_config)

    def run_evaluation_phase(self):
        evaluation_phase_results = {}

        for subphase in range(self.subphase_counts["eval"][self.current_stage]):
            print("Evaluation: Start subphase {}".format(subphase))
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

            episode_count = 0
            next_matchmaking_episode = 0
            done_count = 0
            
            while done_count < self.evaluation_environment.environment_count:
                print("Evaluation Progress - subphase: ", subphase, "Episdoe:", episode_count, "/", self.subphase_episode_count["eval"], end='\r')
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
                for i in range(self.evaluation_environment.environment_count):
                    if game_dones[i]:
                        if episode_count < self.subphase_episode_count["eval"]:
                            for agent_name, single_solver in participant_solvers.items():
                                is_valid_agent = self.evaluation_environment.agent_turn[agent_name]
                                if any(solver_dones[agent_name]):
                                    solver_dones[agent_name] = single_solver.episode_initiate(solver_dones[agent_name], is_valid_agent, is_train=False)
                        else:
                            done_count += 1
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
                    need_restarts = self.evaluation_environment.need_restart()
                    for i in range(self.evaluation_environment.environment_count):
                        if need_restarts[i]:
                            self.evaluation_environment.restart(i)
                            game_dones[i] = True
                            for agent_name in solver_dones:
                                solver_dones[agent_name][i] = True
                    
        return evaluation_phase_results
        
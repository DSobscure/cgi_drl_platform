class DefaultTemplate(dict):
    def __init__(self, config):
        def curriculum_plan(training_infos):
            return 0
        self["curriculum_plan"] = curriculum_plan

        def training_phase_statistics_aggregator(results, participant_mappings, game_dones, game_infos):
            if any(game_dones):
                for i in range(len(game_dones)):
                    if game_dones[i]:
                        info = game_infos[i][0]
                        if "result" in info:
                            solver_result = info["result"]
                            solver_name = participant_mappings[info["agent_name"]]
                            opponent_name = participant_mappings[game_infos[i][1]["agent_name"]]
                            solver_info = info
                            
                            results["{} vs {} Episode Count".format(solver_name, opponent_name)] = results.get("{} vs {} Episode Count".format(solver_name, opponent_name), 0) + 1
                            
                            if solver_result > 0.8: # win
                                results["{} vs {} Win Episode Count".format(solver_name, opponent_name)] = results.get("{} vs {} Win Episode Count".format(solver_name, opponent_name), 0) + 1
                            elif solver_result < 0.2: # lose
                                results["{} vs {} Lose Episode Count".format(solver_name, opponent_name)] = results.get("{} vs {} Lose Episode Count".format(solver_name, opponent_name), 0) + 1
                            else:
                                results["{} vs {} Tie Episode Count".format(solver_name, opponent_name)] = results.get("{} vs {} Tie Episode Count".format(solver_name, opponent_name), 0) + 1
                            results["Average {} vs {} Win Value".format(solver_name, opponent_name)] = ((results["{} vs {} Episode Count".format(solver_name, opponent_name)] - 1) * results.get("Average {} vs {} Win Value".format(solver_name, opponent_name), 0) + solver_result) / results["{} vs {} Episode Count".format(solver_name, opponent_name)]

        self["training_phase_statistics_aggregator"] = training_phase_statistics_aggregator

        def evaluation_phase_statistics_aggregator(results, participant_mappings, game_dones, game_infos):
            if any(game_dones):
                for i in range(len(game_dones)):
                    if game_dones[i]:
                        info = game_infos[i][0]
                        if "result" in info:
                            solver_result = info["result"]
                            solver_name = participant_mappings[info["agent_name"]]
                            opponent_name = participant_mappings[game_infos[i][1]["agent_name"]]
                            solver_info = info

                            results["{} vs {} Episode Count".format(solver_name, opponent_name)] = results.get("{} vs {} Episode Count".format(solver_name, opponent_name), 0) + 1
                            
                            if solver_result > 0.8: # win
                                results["{} vs {} Win Episode Count".format(solver_name, opponent_name)] = results.get("{} vs {} Win Episode Count".format(solver_name, opponent_name), 0) + 1
                            elif solver_result < 0.2: # lose
                                results["{} vs {} Lose Episode Count".format(solver_name, opponent_name)] = results.get("{} vs {} Lose Episode Count".format(solver_name, opponent_name), 0) + 1
                            else:
                                results["{} vs {} Tie Episode Count".format(solver_name, opponent_name)] = results.get("{} vs {} Tie Episode Count".format(solver_name, opponent_name), 0) + 1
                            results["Average {} vs {} Win Value".format(solver_name, opponent_name)] = ((results["{} vs {} Episode Count".format(solver_name, opponent_name)] - 1) * results.get("Average {} vs {} Win Value".format(solver_name, opponent_name), 0) + solver_result) / results["{} vs {} Episode Count".format(solver_name, opponent_name)]

        self["evaluation_phase_statistics_aggregator"] = evaluation_phase_statistics_aggregator

        def stage_phase_statistics_aggregator(training_infos):
            result = {}
            return result
        self["stage_phase_statistics_aggregator"] = stage_phase_statistics_aggregator
        super().__init__(config)
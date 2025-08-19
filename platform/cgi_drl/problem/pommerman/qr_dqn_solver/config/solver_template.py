import numpy as np

class PrioritizedTrainingTemplate(dict):
    def __init__(self, config):
        def learning_rate_scheduler(schedule_infos):
            return 0.00025 / 4
        self["learning_rate_scheduler"] = config.get("learning_rate_scheduler", learning_rate_scheduler)

        def exploration_action_epsilon_scheduler(schedule_infos):
            current_timestep = schedule_infos["current_timestep"]
            agent_count = schedule_infos["agent_count"]
            
            action_epsilon_base = 0.4
            action_epsilon_alpha = 7
            
            epsilons = []
            for i_agent in range(agent_count):
                epsilon = pow(action_epsilon_base, (1+((i_agent) / (agent_count-1)) * action_epsilon_alpha))
                if current_timestep < 256000:
                    epsilon = 1.0 - (1 - epsilon) * (current_timestep / 256000)
                epsilons.append(epsilon)
                
            return epsilons
        self["exploration_action_epsilon_scheduler"] = config.get("exploration_action_epsilon_scheduler", exploration_action_epsilon_scheduler)
        
        def importance_sampling_beta_scheduler(schedule_infos):
            current_timestep = schedule_infos["current_timestep"]
            total_timestep = schedule_infos["total_timestep"]
            return 0.4 + 0.6 * (current_timestep / total_timestep)

        self["importance_sampling_beta_scheduler"] = config.get("importance_sampling_beta_scheduler", importance_sampling_beta_scheduler)

        def agent_statistics_aggregator(agent_statistics, rewards, infos):
            for index in range(len(agent_statistics)):
                if infos[index]["Is Valid Agent"]:
                    agent_statistics[index]["Episode Length"] = agent_statistics[index].get("Episode Length", 0) + 1
                    agent_statistics[index]["Cumulated Extrinsic Reward"] = agent_statistics[index].get("Cumulated Extrinsic Reward", 0) + rewards[index]
                    agent_statistics[index]["Cumulated Exploration Bonus"] = agent_statistics[index].get("Cumulated Exploration Bonus", 0) + infos[index]["exploration_reward"]
                    agent_statistics[index]["Average Q Value Sum"] = ((agent_statistics[index]["Episode Length"] - 1) * agent_statistics[index].get("Average Q Value Sum", 0) + infos[index]["Q Value Sum"]) / agent_statistics[index]["Episode Length"]
             
        def reward_transformer(rewards, infos):
            return np.asarray(rewards)
        self["reward_transformer"] = config.get("reward_transformer", reward_transformer)

        self["agent_statistics_aggregator"] = config.get("agent_statistics_aggregator", agent_statistics_aggregator)

        self["use_double_q"] = config.get("use_double_q", True)

        self["n_step_size"] = config.get("n_step_size", 1)

        def exploration_bonus_coefficient_scheduler(schedule_infos):
            return 1
        self["exploration_bonus_coefficient_scheduler"] = config.get("exploration_bonus_coefficient_scheduler", exploration_bonus_coefficient_scheduler)
        
        super().__init__(config)

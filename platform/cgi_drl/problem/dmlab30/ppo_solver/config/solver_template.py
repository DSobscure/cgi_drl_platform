import numpy as np

class DefaultTemplate(dict):
    def __init__(self, config):
        def learning_rate_scheduler(schedule_infos):
            return 0.00025
        self["learning_rate_scheduler"] = config.get("learning_rate_scheduler", learning_rate_scheduler)

        def clip_epsilon_scheduler(schedule_infos):
            return 0.1
        self["clip_epsilon_scheduler"] = config.get("clip_epsilon_scheduler", clip_epsilon_scheduler)

        def entropy_coefficient_scheduler(schedule_infos):
            return 0.01
        self["entropy_coefficient_scheduler"] = config.get("entropy_coefficient_scheduler", entropy_coefficient_scheduler)

        def value_coefficient_scheduler(schedule_infos):
            return 1
        self["value_coefficient_scheduler"] = config.get("value_coefficient_scheduler", value_coefficient_scheduler)

        def value_clip_range_scheduler(schedule_infos):
            return 1
        self["value_clip_range_scheduler"] = config.get("value_clip_range_scheduler", value_clip_range_scheduler)
        
        def agent_statistics_aggregator(agent_statistics, rewards, infos):
            for index in range(len(agent_statistics)):
                if infos[index]["Is Valid Agent"]:
                    agent_statistics[index]["Episode Length"] = agent_statistics[index].get("Episode Length", 0) + 1
                    agent_statistics[index]["Cumulated Extrinsic Reward"] = agent_statistics[index].get("Cumulated Extrinsic Reward", 0) + rewards[index]
                    agent_statistics[index]["Average Value"] = ((agent_statistics[index]["Episode Length"] - 1) * agent_statistics[index].get("Average Value", 0) + infos[index]["Value"]) / agent_statistics[index]["Episode Length"]

        def reward_transformer(rewards, infos):
            return np.clip(np.asarray(rewards), -1, 1)
        self["reward_transformer"] = config.get("reward_transformer", reward_transformer)

        self["discount_factor_gamma"] = config.get("discount_factor_gamma", 0.99)
        self["discount_factor_lambda"] = config.get("discount_factor_lambda", 0.95)

        self["agent_statistics_aggregator"] = config.get("agent_statistics_aggregator", agent_statistics_aggregator)
        super().__init__(config)

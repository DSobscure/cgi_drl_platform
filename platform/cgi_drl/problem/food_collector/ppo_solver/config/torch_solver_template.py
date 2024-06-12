import numpy as np
from collections import deque

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
        

        self.action_window_size = 20
        self.last_rotate_actions = []

        def agent_statistics_aggregator(agent_statistics, rewards, infos):
            if len(self.last_rotate_actions) == 0:
                for i_index in range(len(agent_statistics)):
                    self.last_rotate_actions.append(deque(maxlen=self.action_window_size))
            for i_index in range(len(agent_statistics)):
                if infos[i_index]["Is Valid Agent"]:
                    agent_statistics[i_index]["Episode Length"] = agent_statistics[i_index].get("Episode Length", 0) + 1
                    agent_statistics[i_index]["Cumulated Extrinsic Reward"] = agent_statistics[i_index].get("Cumulated Extrinsic Reward", 0) + rewards[i_index]
                    agent_statistics[i_index]["Average Value"] = ((agent_statistics[i_index]["Episode Length"] - 1) * agent_statistics[i_index].get("Average Value", 0) + infos[i_index]["Value"]) / agent_statistics[i_index]["Episode Length"]
                    
                    agent_statistics[i_index]["Shaking Cost"] = agent_statistics[i_index].get("Shaking Cost", 0)
                    agent_statistics[i_index]["Spinning Cost"] = agent_statistics[i_index].get("Spinning Cost", 0)

                    if agent_statistics[i_index]["Episode Length"] == 1:
                        self.last_rotate_actions[i_index] = deque(maxlen=self.action_window_size)
                    # Action: [move, rotate]
                    # move: 0-idle, 1-forward, 2-backword
                    # rotate: 0-idle, 1-left, 2-right
                    current_rotate_action = infos[i_index]["Action"][1]
                    self.last_rotate_actions[i_index].append(current_action)
                    # shaking cost
                    for i_action in range(len(self.last_rotate_actions[i_index]) - 2, -1, -1):
                        # use 1x2=2 or 2x1=2 to detect shaking
                        if self.last_rotate_actions[i_index][i_action] * current_rotate_action == 2:
                            agent_statistics[i_index]["Shaking Cost"] += 1
                            infos[i_index]["Shaking Cost"] = 1
                            break
                    # spinning cost
                    if self.last_rotate_actions[i_index].count(current_rotate_action) >= 15:
                        agent_statistics[i_index]["Spinning Cost"] += 1
                        infos[i_index]["Spinning Cost"] = 1
        self["agent_statistics_aggregator"] = config.get("agent_statistics_aggregator", agent_statistics_aggregator)

        def reward_transformer(rewards, infos):
            rewards = np.asarray(rewards)
            return rewards
        self["reward_transformer"] = config.get("reward_transformer", reward_transformer)

        self["discount_factor_gamma"] = config.get("discount_factor_gamma", 0.99)
        self["discount_factor_lambda"] = config.get("discount_factor_lambda", 0.95)

        super().__init__(config)

class ConstantCostTemplate(DefaultTemplate):
    def __init__(self, config):
        super().__init__(config)

        def reward_transformer(rewards, infos):
            rewards = np.asarray(rewards)
            behavior_cost_coefficient = 1
            for i_index in range(len(rewards)):
                if "Shaking Cost" in infos[i_index]:
                    rewards[i_index] -= behavior_cost_coefficient * infos[i_index]["Shaking Cost"]
                if "Spinning Cost" in infos[i_index]:
                    rewards[i_index] -= behavior_cost_coefficient * infos[i_index]["Spinning Cost"]
            return rewards
        self["reward_transformer"] = reward_transformer

class AbcRlTemplate(DefaultTemplate):
    def __init__(self, config):
        super().__init__(config)
        
        def reward_transformer(rewards, infos):
            rewards = np.asarray(rewards)
            for i_index in range(len(rewards)):
                behavior_cost_coefficient = infos["Behavior Cost Coefficient"]
                if "Shaking Cost" in infos[i_index]:
                    rewards[i_index] -= behavior_cost_coefficient * infos[i_index]["Shaking Cost"]
                if "Spinning Cost" in infos[i_index]:
                    rewards[i_index] -= behavior_cost_coefficient * infos[i_index]["Spinning Cost"]
            return rewards
        self["reward_transformer"] = config.get("reward_transformer", reward_transformer)

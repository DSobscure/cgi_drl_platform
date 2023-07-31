import numpy as np
from cgi_drl.environment.environment_wrapper import EnvironmentWrapper
import time
import pommerman

class PommermanEnvironmentWrapper(EnvironmentWrapper):
    def __init__(self, config):
        self.config = config
        self.agent_count = len(config["agent_names"])
        self.agent_list = [pommerman.agents.BaseAgent() for _ in range(self.agent_count)]
        self.render = config.get("render", False)
        self.unit_game_reward = 3
        self.environment_id = config["environment_id"]

        self.env = pommerman.make(self.environment_id, self.agent_list)
        self.env._max_steps = config["max_steps"]
        # ! For team mode
        self.env._is_partially_observable = config.get("partial_obs", False)

        self.action_space = self.env.action_space
        self.time_step = 0

        self.previous_observation = None

    def get_turn(self):
        return [agent.is_alive for agent in self.agent_list]

    @staticmethod
    def get_extra_reward(previous_observation, current_observation, action):
        """Get the bonus reward for explorationg.

        the `action` is between 2 observation.
        """
        reward = 0.0
        previous_board = previous_observation["board"]
        current_position = current_observation["position"]

        # Calculate the reward of picking items
        picking_item_flags = [
            pommerman.utility._position_is_item(previous_board, current_position, item)
            for item in [
                pommerman.constants.Item.ExtraBomb,
                pommerman.constants.Item.IncrRange,
                pommerman.constants.Item.Kick,
            ]
        ]
        reward += sum(0.3 for f in picking_item_flags if f)

        # Calculate the reward of placing bombs
        if (
            action == pommerman.constants.Action.Bomb.value
            and previous_observation["ammo"] > 0
            and previous_observation["bomb_life"][current_position] == 0.0
        ):
            # Placing bomb reward
            reward += 0.03

        return reward

    def get_reward_and_status(self, reward, agent_alive):
        new_reward = [0] * len(reward)
        win = [False] * len(reward)
        loss = [False] * len(reward)
        draw = False

        # Draw case:
        # 1. Game is over from time
        # 2. All agents died at the same time
        if len(agent_alive) == 0 or all(r == -1 for r in reward):
            new_reward = [-self.unit_game_reward] * len(reward)
            draw = True
        else:
            if self.environment_id.startswith("PommeTeam"):
                for i in range(4):
                    if i + 10 not in agent_alive:
                        # i died
                        loss[i] = True
                        new_reward[i] = -self.unit_game_reward
                        if i + 10 in self.previous_alive:
                            if i == 0 or i == 2:
                                new_reward[1] += self.unit_game_reward
                                new_reward[3] += self.unit_game_reward
                            else:
                                new_reward[0] += self.unit_game_reward
                                new_reward[2] += self.unit_game_reward
                if reward == [1, -1, 1, -1]:
                    # 0,2 win
                    win = [True, False, True, False]
                elif reward == [-1, 1, -1, 1]:
                    # 1,3 win
                    win = [False, True, False, True]
            elif self.environment_id.startswith("One"):
                for i in range(2):
                    if reward[i] == 1:
                        win[i] = True
                        new_reward[i] = self.unit_game_reward
                    elif i + 10 not in agent_alive:
                        # i die
                        loss[i] = True
                        new_reward[i] = -self.unit_game_reward

        self.previous_alive = agent_alive
        return new_reward, win, loss, draw

    def step(self, actions, action_settings=None):
        observations, rewards, dones, infos = [], [], [], []
        _obs, _reward, _terminal, _ = self.env.step(actions)
        _reward, win, loss, draw = self.get_reward_and_status(_reward, _obs[0]["alive"])
        self.time_step += 1

        for i, (obs, r) in enumerate(zip(_obs, _reward)):
            # agent 0, 1, 2, 3 => 10, 11, 12, 13
            is_alive = i + 10 in obs["alive"]
            # Determine the winner
            if draw:
                result = 0.5
            elif win[i]:
                result = 1.0
            else:
                result = 0  # die or not end
            obs["time_step"] = self.time_step
            observations.append(obs)
            # Use immediate reward when agent died
            # win: +1, die: -1, draw: -1
            rewards.append(r)
            dones.append(True if _terminal else (not is_alive))
            infos.append(
                {
                    "result": result,
                    "win": 1 if win[i] else 0,
                    "draw": 1 if draw else 0,
                    "placing_bomb": 1
                    if (
                        actions[i] == pommerman.constants.Action.Bomb.value
                        and self.previous_observation[i]["ammo"] > 0
                        and self.previous_observation[i]["bomb_life"][obs["position"]] == 0.0
                    )
                    else 0,
                    "exploration_reward": PommermanEnvironmentWrapper.get_extra_reward(
                        self.previous_observation[i], obs, actions[i]
                    ),
                }
            )

        if self.render:
            self.env.render()
            time.sleep(0.05)
        self.previous_observation = _obs

        return observations, rewards, dones, infos

    def reset(self, index=-1, reset_settings=None):
        self.previous_observation = obs = self.env.reset()
        self.previous_alive = obs[0]["alive"]
        self.time_step = 0
        observations = []
        for i in range(self.agent_count):
            obs[i]["time_step"] = self.time_step
            observations.append(obs[i])
        if self.render:
            self.env.render()
        return observations

    def render(self, index=-1):
        self.env.render()

    def get_action_space(self, index=0):
        return self.action_space.n

    def sample(self, index=-1):
        actions = []
        if index == -1:
            for i in range(self.agent_count):
                actions.append(np.random.randint(self.get_action_space(i)))
        else:
            actions = np.random.randint(self.get_action_space(index))
        return actions

    def close(self):
        self.env.env.close()

    def get_agent_count(self):
        return self.agent_count

    def need_restart(self):
        return False

# test
if __name__ == '__main__':
    env = PommermanEnvironmentWrapper({
        "environment_id" : "OneVsOne-v0",
        "agent_names" : ["Alice", "Bob"],
        "render" : False,
        "max_steps": 400
    }) 
    
    print(env.get_action_space())
    for i_episode in range(10):
        obs = env.reset()
        episode_done = False

        while not episode_done:
            actions = env.sample()
            obs, rewards, dones, infos = env.step(actions)
            print("actions, reward:", actions, rewards)
            episode_done = all(dones)
        print("Episode {} finished".format(i_episode))

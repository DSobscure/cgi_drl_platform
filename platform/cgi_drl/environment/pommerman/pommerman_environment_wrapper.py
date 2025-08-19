import numpy as np
from cgi_drl.environment.environment_wrapper import EnvironmentWrapper
import time
import pommerman
from PIL import Image
import random

class PommermanEnvironmentWrapper(EnvironmentWrapper):
    GAME_REWARD = 1
    PICKING_ITEM_REWARD = GAME_REWARD / 20
    PLACING_BOMB_REWARD = PICKING_ITEM_REWARD / 20
    def __init__(self, config):
        self.config = config
        self.agent_count = len(config["agent_names"])
        self.agent_list = [pommerman.agents.BaseAgent() for _ in range(self.agent_count)]
        self.is_render = config.get("render", False)
        self.environment_id = config["environment_id"]
        self.randomized_team_position = config.get("randomized_team_position", False)

        self.env = pommerman.make(self.environment_id, self.agent_list)
        self.env._max_steps = config["max_steps"]
        # ! For team mode
        self.env._is_partially_observable = config.get("partial_obs", False)

        self.action_space = self.env.action_space
        self.time_step = 0

        self.previous_observation = None
        if self.randomized_team_position:
            self.is_flip = random.choice([True, False])
        else:
            self.is_flip = False
          
        # For rendering images
        self.TILE_SIZE = 50
        RESOURCES_DIR = "/root/playground/pommerman/resources/"
        self.color_to_tile = {
            tuple(color): Image.open(RESOURCES_DIR + f"{tile['file_name']}").resize((self.TILE_SIZE, self.TILE_SIZE))
            for color, tile in zip(pommerman.constants.ITEM_COLORS, pommerman.constants.IMAGES_DICT.values())
        }
        agent_tiles = {
            idx: Image.open(RESOURCES_DIR + f"Agent{idx}.png").resize((self.TILE_SIZE, self.TILE_SIZE))
            for idx in range(len(pommerman.constants.AGENT_COLORS))
        }
        for idx, color in enumerate(pommerman.constants.AGENT_COLORS):
            self.color_to_tile[tuple(color)] = agent_tiles[idx]
        self.default_tile = Image.new("RGB", (self.TILE_SIZE, self.TILE_SIZE), color=(0, 0, 0))

    def get_turn(self):
        turns = [agent.is_alive for agent in self.agent_list]
        if self.is_flip:
            turns = turns[1::2] + turns[::2]
        return turns

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
        reward += sum(PommermanEnvironmentWrapper.PICKING_ITEM_REWARD for f in picking_item_flags if f)

        # Calculate the reward of placing bombs
        if (
            action == pommerman.constants.Action.Bomb.value
            and previous_observation["ammo"] > 0
            and previous_observation["bomb_life"][current_position] == 0.0
        ):
            # Placing bomb reward
            reward += PommermanEnvironmentWrapper.PLACING_BOMB_REWARD

        return reward

    def get_reward_and_status(self, reward, agent_alive):
        new_reward = [0] * len(reward)
        win = [False] * len(reward)
        draw = False

        # Draw case:
        # 1. Game is over from time
        # 2. All agents died at the same time
        if len(agent_alive) == 0 or all(r == -1 for r in reward):
            new_reward = [0] * len(reward) # [-PommermanEnvironmentWrapper.GAME_REWARD] * len(reward)
            draw = True
        else:
            if self.environment_id.startswith("PommeTeam"):
                for i in range(4):
                    if i + 10 not in agent_alive: # The four agent id is [10, 11, 12, 13]
                        # i died
                        new_reward[i] += 0 # -PommermanEnvironmentWrapper.GAME_REWARD
                        if i + 10 in self.previous_alive:
                            if i == 0 or i == 2:
                                new_reward[1] += PommermanEnvironmentWrapper.GAME_REWARD
                                new_reward[3] += PommermanEnvironmentWrapper.GAME_REWARD
                            else:
                                new_reward[0] += PommermanEnvironmentWrapper.GAME_REWARD
                                new_reward[2] += PommermanEnvironmentWrapper.GAME_REWARD
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
                        new_reward[i] = PommermanEnvironmentWrapper.GAME_REWARD
                    elif i + 10 not in agent_alive:
                        # i die
                        new_reward[i] = 0 # -PommermanEnvironmentWrapper.GAME_REWARD

        self.previous_alive = agent_alive
        return new_reward, win, draw

    def step(self, actions, action_settings=None):
        if self.is_flip:
            actions = actions[1::2] + actions[::2]
        observations, rewards, dones, infos = [], [], [], []
        _obs, _reward, _terminal, _ = self.env.step(actions)
        _reward, win, draw = self.get_reward_and_status(_reward, _obs[0]["alive"])
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
            # dones.append(True if _terminal else (not is_alive))
            dones.append(any(win) or draw)
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

        if self.is_render:
            self.env.render()
            time.sleep(0.05)
        self.previous_observation = _obs
        
        if self.is_flip:
            observations, rewards, dones, infos = observations[1::2] + observations[::2], rewards[1::2] + rewards[::2], dones[1::2] + dones[::2], infos[1::2] + infos[::2]
        return observations, rewards, dones, infos

    def reset(self, index=-1, reset_settings=None):
        if self.randomized_team_position:
            self.is_flip = random.choice([True, False])
        self.previous_observation = obs = self.env.reset()
        self.previous_alive = obs[0]["alive"]
        self.time_step = 0
        observations = []
        for i in range(self.agent_count):
            obs[i]["time_step"] = self.time_step
            observations.append(obs[i])
        if self.is_render:
            self.env.render()
        if self.is_flip:
            observations = observations[1::2] + observations[::2]
        return observations

    def render(self, index=-1):
        rgb_array = self.env.render(mode='rgb_array')
        height, width, _ = rgb_array.shape
        output_image = Image.new("RGB", (width * self.TILE_SIZE, height * self.TILE_SIZE))
        
        for row in range(height):
            for col in range(width):
                pixel_color = tuple(rgb_array[row, col])
                tile = self.color_to_tile.get(pixel_color, self.default_tile)
                
                x, y = col * self.TILE_SIZE, row * self.TILE_SIZE
                output_image.paste(tile, (x, y))
        output_image = np.array(output_image)
        return output_image

    def get_action_space(self, index=0):
        return self.action_space.n # 6 actions

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
    from cgi_drl.visualization_tool.gif_maker import make_video
    
    env = PommermanEnvironmentWrapper({
        "environment_id" : "OneVsOne-v0",
        "agent_names" : ["Alice", "Bob"],
        "render" : False,
        "max_steps": 400
    }) 
    
    print(env.get_action_space())
    for i_episode in range(1):
        obs = env.reset()
        episode_done = False
        images = []
        start_time = time.time()

        while not episode_done:
            images.append(env.render())
            actions = env.sample()
            actions = [5, 5]
            obs, rewards, dones, infos = env.step(actions)
            print("obs:", obs)
            exit()
            # print("actions, reward:", actions, rewards)
            episode_done = all(dones)
        print("Episode {} finished, using {} seconds".format(i_episode, time.time() - start_time))
        print(np.asarray(images).shape)
        make_video(images, f"pommerman_example{i_episode}.mp4", duration=len(images))

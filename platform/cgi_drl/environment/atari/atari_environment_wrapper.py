import numpy as np
from cgi_drl.environment.environment_wrapper import EnvironmentWrapper
import time

class AtariEnvironmentWrapper(EnvironmentWrapper):
    def __init__(self, config):
        self.config = config
        environment_id = self.config["environment_id"]
        self.episode_life = self.config.get("episode_life", True)

        from cgi_drl.environment.atari.open_ai_baseline_atari import make_atari, wrap_deepmind

        env = make_atari(environment_id)
        env = wrap_deepmind(
            env=env,
            episode_life=self.episode_life,
            frame_stack=False,
            clip_rewards=False,
            scale=False
        )
        self.env = env

    def step(self, actions, action_settings=None):
        observation, reward, done, info = self.env.step(actions[0])
        return [observation], [reward], [done], [info]

    def reset(self, index=-1, reset_settings=None):
        if reset_settings == None:
            reset_settings = {}
        if reset_settings.get("mode", "") == "hard":
            observation = self.env.env.env.env.reset()
        else:
            observation = self.env.reset()
        if index == -1:
            return [observation]
        else:
            return observation


    def render(self, index=-1):
        self.env.render()

    def get_action_space(self, index=0):
        return self.env.action_space.n

    def sample(self, index=-1):
        if index == -1:
            return [np.random.randint(self.get_action_space(0))]
        else:
            return np.random.randint(self.get_action_space(0))

    def close(self):
        self.env.close()

    def get_agent_count(self):
        return 1


# test
if __name__ == '__main__':
    env = AtariEnvironmentWrapper({
        "environment_id" : "BreakoutNoFrameskip-v4",
        "episode_life" : False,
    }) 
    env.reset()
    print(env.get_action_space())
    for i in range(1000):
        actions = env.sample()
        obs, rewards, dones, infos = env.step(actions)
        print(actions, dones, infos)
        if any(dones):
            env.reset(reset_settings={"mode":"hard"})

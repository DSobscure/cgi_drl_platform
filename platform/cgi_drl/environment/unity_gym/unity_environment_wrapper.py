import numpy as np
from cgi_drl.environment.environment_wrapper import EnvironmentWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

class UnityEnvironmentWrapper(EnvironmentWrapper):
    def __init__(self, config):
        self.config = config
        self.executable_path = self.config["executable_path"]
        self.uint8_visual = self.config.get("uint8_visual", True)
        self.flatten_branched = self.config.get("flatten_branched", False)
        self.allow_multiple_obs = self.config.get("allow_multiple_obs", True)
        self.action_space_seed = self.config.get("action_space_seed", None)
        self.worker_id = self.config.get("environment_index", 0)

        unity_env = UnityEnvironment(self.executable_path, worker_id=self.worker_id)
        env = UnityToGymWrapper(unity_env, 
            uint8_visual=self.uint8_visual, 
            flatten_branched=self.flatten_branched, 
            allow_multiple_obs=self.allow_multiple_obs, 
            action_space_seed=self.action_space_seed
        )
        self.env = env
        self.agent_count = 1

    def step(self, actions, action_settings=None):
        observation, reward, done, info = self.env.step(actions[0])
        return [observation], [reward], [done], [info]

    def reset(self, index=-1, reset_settings=None):
        observation = self.env.reset()

        if index == -1:
            return [observation]
        else:
            return observation

    def get_action_space(self, index=0):
        return self.env.action_space

    def sample(self, index=-1):
        if index == -1:
            return [self.env.action_space.sample()]
        else:
            return self.env.action_space.sample()

    def close(self):
        self.env.close()

    def get_agent_count(self):
        return 1


# test
if __name__ == '__main__':
    env = UnityEnvironmentWrapper({
        "executable_path" : r"..\..\..\..\infrastructure\GameExecutables\VisualFoodCollectorSingle",
        "uint8_visual": True
    }) 
    env.reset()
    print(env.get_action_space())
    for i in range(100):
        actions = env.sample()
        obs, rewards, dones, infos = env.step(actions)
        # [1, 1, 3, 84, 84]
        print(np.asarray(obs).shape, actions, dones)
        if any(dones):
            env.reset()
    env.close()

import numpy as np
from cgi_drl.environment.environment_wrapper import EnvironmentWrapper
import time

DEFAULT_ACTION_SET = [
    np.asarray([0, 0, 0, 1, 0, 0, 0], np.intc),    # Forward
    np.asarray([0, 0, 0, -1, 0, 0, 0], np.intc),   # Backward
    np.asarray([0, 0, -1, 0, 0, 0, 0], np.intc),   # Strafe Left
    np.asarray([0, 0, 1, 0, 0, 0, 0], np.intc),    # Strafe Right
    np.asarray([-20, 0, 0, 0, 0, 0, 0], np.intc),  # Look Left
    np.asarray([20, 0, 0, 0, 0, 0, 0], np.intc),   # Look Right
    np.asarray([-20, 0, 0, 1, 0, 0, 0], np.intc),  # Look Left + Forward
    np.asarray([20, 0, 0, 1, 0, 0, 0], np.intc),   # Look Right + Forward
    np.asarray([0, 0, 0, 0, 1, 0, 0], np.intc),    # Fire.
]

class DMLab30EnvironmentWrapper(EnvironmentWrapper):
    def __init__(self, config):
        self.config = config
        environment_id = self.config["environment_id"]
        self._action_skip = self.config["action_skip"]

        import deepmind_lab

        env = deepmind_lab.Lab("contributed/dmlab30/" + environment_id, ['RGB_INTERLEAVED'], {'fps': '30', 'width': '96', 'height': '72'})
        self.env = env

    def step(self, actions, action_settings=None):
        reward = self.env.step(DEFAULT_ACTION_SET[actions[0][0]], num_steps=self._action_skip)
        done = not self.env.is_running()
        if done:
            observation = np.zeros([72, 96, 3])
        else:
            observation = self.env.observations()['RGB_INTERLEAVED']
        return [observation], [reward], [done], [{}]

    def reset(self, index=-1, reset_settings=None):
        if reset_settings == None:
            reset_settings = {}
        self.env.reset()
        observation = self.env.observations()['RGB_INTERLEAVED']
        if index == -1:
            return [observation]
        else:
            return observation


    def render(self, index=-1):
        self.env.render()

    def get_action_space(self, index=0):
        return len(DEFAULT_ACTION_SET)

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
    env = DMLab30EnvironmentWrapper({
        "environment_id" : "rooms_watermaze",
        "action_skip" : 4
    }) 
    env.reset()
    print(env.get_action_space())
    score = 0
    for i in range(1000):
        actions = env.sample()
        obs, rewards, dones, infos = env.step(actions)
        score += rewards[0]
        print(np.shape(obs), rewards, actions, dones, infos)
        if any(dones):
            break
    print("Score: ", score)

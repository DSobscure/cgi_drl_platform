from cgi_drl.environment.environment_wrapper import EnvironmentWrapper
from collections import defaultdict
import importlib
import numpy as np

class MultiAgentEnvironmentBroker:
    '''
    MultiAgentEnvironmentWrapper define
    the interface between real environment and agent wanted environment.
    '''
    def __init__(self, config):
        EnvironmentWrapper = getattr(importlib.import_module(config["environment_wrapper_path"]), config["environment_wrapper_class_name"])
        self.environment_wrapper = EnvironmentWrapper(config["environment_config"])

        self.agent_names = config["agent_names"]
        self.agent_count = self.environment_wrapper.agent_count
        self.environment_count = self.agent_count // len(self.agent_names)
        
        agent_name_set = set(self.agent_names)
        self.solver_agent_counts = {}
        for name in agent_name_set:
            self.solver_agent_counts[name] = self.agent_names.count(name) * self.environment_count
        self.solver_count = len(agent_name_set)
        

        self.environment_turns = [True for _ in range(self.agent_count)]
        self.agent_turn = defaultdict(list)
        self._update_turn()
        self.sub_environments = {}
        for name in agent_name_set:
            self.sub_environments[name] = MultiAgentEnvironmentBroker.SubEnvironmentWrapper(self, self.solver_agent_counts[name])
        
        # given agent_name and environment_index, get the corresponding agent offset to an agent
        self.environment_index_agent_offset = defaultdict(list)
        offsets = defaultdict(int)
        for i in range(self.environment_count):
            for name in agent_name_set:
                self.environment_index_agent_offset[name].append([])
            for name in self.agent_names:
                self.environment_index_agent_offset[name][-1].append(offsets[name])
                offsets[name] += 1

    def step(self, action, action_settings = None):
        ''' Do an action to environment and return the 
        next observation, reward, is done(terminal), and info_dict from environment'''
        # Need to check turn before the env.step(),
        # since we need to know the exact step of terminal
        if action_settings == None:
            action_settings = {}
        self._update_turn()
        real_actions = self._filter_actions(action, action_settings)
        observations, rewards, dones, infos = self.environment_wrapper.step(real_actions, action_settings)

        game_dones = []
        game_infos = []

        total_offset = 0
        for i_e in range(self.environment_count):
            episode_dones = []
            episode_infos = []
            offsets = defaultdict(int)
            for i, name in enumerate(self.agent_names):
                offset = offsets[name]
                if self.environment_turns[total_offset]:
                    sub_env_offset = self.environment_index_agent_offset[name][i_e][offset]
                    self.sub_environments[name].observations[sub_env_offset] = observations[total_offset]
                    self.sub_environments[name].rewards[sub_env_offset] = rewards[total_offset]
                    self.sub_environments[name].dones[sub_env_offset] = dones[total_offset]
                    infos[total_offset]["agent_name"] = name
                    self.sub_environments[name].infos[sub_env_offset] = infos[total_offset]
                    episode_dones.append(dones[total_offset])
                    episode_infos.append(infos[total_offset])
                else:
                    sub_env_offset = self.environment_index_agent_offset[name][i_e][offset]
                    self.sub_environments[name].observations[sub_env_offset] = {}
                    self.sub_environments[name].rewards[sub_env_offset] = 0
                    self.sub_environments[name].dones[sub_env_offset] = False
                    infos[total_offset]["agent_name"] = name
                    self.sub_environments[name].infos[sub_env_offset] = {}
                    episode_infos.append({})
                offsets[name] += 1
                total_offset += 1
            if len(episode_dones) == 0:
                game_dones.append(False)
            else:
                game_dones.append(all(episode_dones))
            game_infos.append(episode_infos)

        return game_dones, game_infos

    def _update_turn(self):
        if hasattr(self.environment_wrapper, "get_turn"):
            self.environment_turns = self.environment_wrapper.get_turn()
        else:
            self.environment_turns = [True for _ in range(self.agent_count)]

        self.agent_turn.clear()
        total_offset = 0
        for i_e in range(self.environment_count):
            for i, name in enumerate(self.agent_names):
                self.agent_turn[name].append(self.environment_turns[total_offset])
                total_offset += 1

    def _filter_actions(self, actions, action_settings):
        real_actions = []
        agent_index_offset = defaultdict(int)
        total_offset = 0
        for i_e in range(self.environment_count):
            for i, name in enumerate(self.agent_names):
                action = actions[name][agent_index_offset[name]]
                real_actions.append(action)
                agent_index_offset[name] += 1
                total_offset += 1

        return real_actions

    def reset_game(self, environment_index=-1, reset_settings=None):
        ''' reset the environment'''
        if reset_settings == None:
            reset_settings = {}
        if environment_index != -1:
            reset_agent_index = environment_index * len(self.agent_names)
        else:
            reset_agent_index = -1
        observations = self.environment_wrapper.reset(reset_agent_index, reset_settings)
        self._update_turn()
        
        if environment_index == -1:
            total_offset = 0
            for i_e in range(self.environment_count):
                offsets = defaultdict(int)
                for i, name in enumerate(self.agent_names):
                    offset = offsets[name]
                    self.sub_environments[name].observations[self.environment_index_agent_offset[name][i_e][offset]] = observations[total_offset]
                    offsets[name] += 1
                    total_offset += 1
        else:
            offsets = defaultdict(int)
            for i, name in enumerate(self.agent_names):
                offset = offsets[name]
                self.sub_environments[name].observations[self.environment_index_agent_offset[name][environment_index][offset]] = observations[i]
                offsets[name] += 1

    def get_action_space(self, index=0):
        return self.environment_wrapper.get_action_space()

    def get_agent_count(self):
        return self.agent_count

    def close(self):
        self.environment_wrapper.close()

    def sample(self, index=-1):
        actions = defaultdict(list)
        if index == -1:
            offset = 0 
            for e in range(self.environment_count):
                for i, name in enumerate(self.agent_names):
                    actions[name].append(self.environment_wrapper.sample(i + offset))
                offset += len(self.agent_names)
        else:
            actions = self.environment_wrapper.sample(index)
        return actions

    def need_restart(self, environment_index=-1):
        return self.environment_wrapper.need_restart(environment_index)

    def restart(self, environment_index):
        return self.environment_wrapper.restart(environment_index)

    class SubEnvironmentWrapper(EnvironmentWrapper):
        def __init__(self, main_environment, agent_count):
            self.main_environment = main_environment
            self.agent_count = agent_count
            self.observations = [{} for _ in range(self.agent_count)]
            self.rewards = [None for _ in range(self.agent_count)]
            self.dones = [None for _ in range(self.agent_count)]
            self.infos = [None for _ in range(self.agent_count)]

        def step(self, actions, action_settings={}):
            return self.observations, self.rewards, self.dones, self.infos

        def reset(self, index=-1, reset_settings={}):
            if index == -1:
                return self.observations
            else:
                return [self.observations[index]]

        def get_action_space(self, index=0):
            return self.main_environment.get_action_space(index)

        def sample(self, index=-1):
            return self.main_environment.sample(index)

        def close(self):
            pass

        def get_agent_count(self):
            return self.agent_count

if __name__ == "__main__":   
    import time

    config = {
        "agent_names" : ["Alice", "Bob"],
        "environment_config" : {
            "environment_host_ip" : "127.0.0.1",
            "environment_host_port" : 30000,
            "environment_count" : 4,
            "environment_wrapper_path" : "cgi_drl.environment.pommerman.pommerman_environment_wrapper",
            "environment_wrapper_class_name" : "PommermanEnvironmentWrapper",
            "environment_id" : "OneVsOne-v0",
            "agent_names" : ["Alice", "Bob"],
            "render" : False,
            "max_steps": 400
        },
        "environment_wrapper_path" : "cgi_drl.environment.distributed_framework.environment_requester",
        "environment_wrapper_class_name" : "EnvironmentRequester"
    }
    from cgi_drl.environment.multi_agent_environment_broker import MultiAgentEnvironmentBroker
    env = MultiAgentEnvironmentBroker(config)

    env.reset_game()
    episode_count = 0
    while episode_count < 10:
        actions = env.sample()
        print(actions)
        game_dones, game_infos = env.step(actions)
        print(game_dones, game_infos)
        for i in range(len(game_dones)):
            if game_dones[i]:
                env.reset_game(i)
                episode_count += 1
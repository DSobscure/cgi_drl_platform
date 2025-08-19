from cgi_drl.problem.reinforcement_learning_trainer import ReinforcementLearningTrainer
import numpy as np
from pommerman import agents

# In pommerman/agents/simple_agent.py, line 381 you need to add a next_position check to prevent KeyError: None for 2v2 mode
class SimpleSolver(ReinforcementLearningTrainer):
    def __init__(self, solver_config):
        self.total_time_step = 0
        self.available_actions = solver_config["available_actions"]
        self.is_mixed = solver_config.get("is_mixed", False)
        self.agent_candidates = solver_config.get("agent_candidates", [])
        
    def get_agent_count(self, is_train=True):
        if is_train:
            return self.environment.agent_count
        else:
            return self.evaluation_environment.agent_count

    def get_environment(self, is_train=True):
        if is_train:
            return self.environment
        else:
            return self.evaluation_environment

    def initialize(self, **kwargs):
        self.agent_list = []
        for _ in range(self.get_agent_count()):
            self.agent_list.append(agents.SimpleAgent())
            
        self.observations = {
            True: [None for _ in range(self.get_agent_count())], # for train
            False: [None for _ in range(self.get_agent_count(False))] # for eval
        }
        if self.is_mixed:
            self.used_agent = {
                True: [0 for _ in range(self.get_agent_count())], # for train
                False: [0 for _ in range(self.get_agent_count(False))] # for eval
            }

    def terminate(self):
        self.environment.close()
        self.evaluation_environment.close()

    def episode_initiate(self, dones, is_valid_agent, is_train=True):
        for i in range(self.get_agent_count(is_train)):
            if is_valid_agent[i] and dones[i]:
                dones[i] = False
                self.observations[is_train][i] = self.get_environment(is_train).reset(i)[0]
                if self.is_mixed:
                    self.used_agent[is_train][i] = np.random.randint(len(self.agent_candidates))
        return dones

    def episode_terminate(self, dones, is_valid_agent, is_train=True):
        pass

    def decide_agent_actions(self, is_valid_agent, is_train=True):
        actions = []
        if self.is_mixed:
            for i_index in range(self.get_agent_count()):
                if not is_valid_agent[i_index]:
                    actions.append(0)
                else:
                    if self.agent_candidates[self.used_agent[is_train][i_index]] == "Camp":
                        actions.append(0)
                    elif self.agent_candidates[self.used_agent[is_train][i_index]] == "Roam":
                        actions.append(np.random.randint(5))
                    elif self.agent_candidates[self.used_agent[is_train][i_index]] == "SimpleNoBomb":
                        self.observations[is_train][i_index]['ammo'] = 0
                        actions.append(self.agent_list[i_index].act(self.observations[is_train][i_index], self.available_actions))
                    elif self.agent_candidates[self.used_agent[is_train][i_index]] == "Simple":
                        actions.append(self.agent_list[i_index].act(self.observations[is_train][i_index], self.available_actions))
                    else:
                        raise RuntimeError('Unknown agent candidate in mixed agent')
        else:
            for i_index in range(self.get_agent_count()):
                if not is_valid_agent[i_index]:
                    actions.append(0)
                else:
                    if 5 not in self.available_actions: # no bomb
                        self.observations[is_train][i_index]['ammo'] = 0
                    actions.append(self.agent_list[i_index].act(self.observations[is_train][i_index], self.available_actions))
        decision = {
            "actions": actions,
        }
        return decision

    def on_time_step(self, decision, is_valid_agent, is_train=True):
        actions = decision["actions"]
        next_observations, rewards, dones, infos = self.get_environment(is_train).step(actions) 
        for i in range(self.get_agent_count(is_train)):
            self.observations[is_train][i] = next_observations[i]
        return dones

    def save_model(self):
        pass

    def save_to_agent_pool(self, agent_pool_path):
        pass

    def load_from_agent_pool(self, agent_pool_path):
        pass

    def load_model(self, is_from_file=False):
        pass
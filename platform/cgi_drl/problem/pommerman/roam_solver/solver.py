from cgi_drl.problem.reinforcement_learning_trainer import ReinforcementLearningTrainer
import numpy as np

class RoamSolver(ReinforcementLearningTrainer):
    def __init__(self, solver_config):
        self.total_time_step = 0
        self.available_actions = solver_config["available_actions"]
        
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
        pass

    def terminate(self):
        self.environment.close()
        self.evaluation_environment.close()

    def episode_initiate(self, dones, is_valid_agent, is_train=True):
        for i in range(self.get_agent_count(is_train)):
            if is_valid_agent[i] and dones[i]:
                dones[i] = False
        return dones

    def episode_terminate(self, dones, is_valid_agent, is_train=True):
        pass


    def decide_agent_actions(self, is_valid_agent, is_train=True):
        actions = np.random.choice(self.available_actions, self.get_agent_count(is_train)).tolist()
        decision = {
            "actions": actions,
        }
        return decision

    def on_time_step(self, decision, is_valid_agent, is_train=True):
        actions = decision["actions"]
        next_observations, rewards, dones, infos = self.get_environment(is_train).step(actions) 
        return dones

    def save_model(self):
        pass

    def save_to_agent_pool(self, agent_pool_path):
        pass

    def load_from_agent_pool(self, agent_pool_path):
        pass

    def load_model(self, is_from_file=False):
        pass

import abc
from time import strftime

class ReinforcementLearningTrainer(abc.ABC):
    def __init__(self, solver_config):
        # version path
        self.run_id = solver_config.get("run_id", "")
        self.version_path = solver_config["version"] + solver_config.get("run_id", "")
        self.log_path = self._create_path(self.version_path, "log")
        self.model_path = self._create_path(self.version_path, "model")
        self.video_path = self._create_path(self.log_path, "video")
        self.training_varaibles_path = self._create_path(self.version_path, "training_varaibles")
        # training parameter
        self.total_time_step = 0
        self.training_steps = solver_config["training_steps"]

    @staticmethod
    def _create_path(parent, dname):
        import os
        dpath = os.path.join(parent, dname)
        if not os.path.exists(dpath):
            os.makedirs(dpath)
        return dpath

    @staticmethod
    def _open_log(path, name):
        import os.path as osp
        fname = osp.join(path, name)
        print(strftime("%Y-%m-%d %H:%M:%S"), "create", fname, "log file")
        return open(fname, mode="a", buffering=1)

    @abc.abstractmethod
    def get_agent_count(self, is_train=True):
        ''' get the simulation count per time step
        '''
        return NotImplemented

    @abc.abstractmethod
    def get_environment(self, is_train=True):
        ''' get the simulation environment
        '''
        return NotImplemented

    @abc.abstractmethod
    def initialize(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def terminate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def episode_initiate(self, dones, is_valid_agent, is_train=True):
        raise NotImplementedError

    @abc.abstractmethod
    def episode_terminate(self, dones, is_valid_agent, is_train=True):
        raise NotImplementedError

    @abc.abstractmethod
    def decide_agent_actions(self, is_valid_agent, is_train=True):
        raise NotImplementedError
    
    @abc.abstractmethod
    def on_time_step(self, decision, is_valid_agent, is_train=True):
        raise NotImplementedError

    def train(self):
        self.initialize()
        training_steps = self.training_steps
        dones = [True for _ in range(self.get_agent_count())]
        is_valid_agent = [True for _ in range(self.get_agent_count())]
        while self.total_time_step <= training_steps:
            dones = self.episode_initiate(dones, is_valid_agent)
            while not any(dones):
                decision = self.decide_agent_actions(is_valid_agent)
                dones = self.on_time_step(decision, is_valid_agent)
                self.total_time_step += self.get_agent_count()
            self.episode_terminate(dones, is_valid_agent)
        self.terminate()


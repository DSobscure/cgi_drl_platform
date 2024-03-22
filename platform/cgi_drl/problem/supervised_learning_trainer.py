import abc
import os

class SupervisedLearningTrainer(abc.ABC):
    '''define the supervised learning behavior'''
    def __init__(self, solver_config):
        # version path
        self.run_id = solver_config.get("run_id", "")
        self.version_path = solver_config["version"] + solver_config.get("run_id", "")
        self.log_path = self._create_path(self.version_path, "log")
        self.model_path = self._create_path(self.version_path, "model")
        self.video_path = self._create_path(self.log_path, "video")
        self.training_varaibles_path = self._create_path(self.version_path, "training_varaibles")
        self.solver_config = solver_config

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
    def initialize(self):
        ''' initialize all required setting before start running
        '''
        return NotImplemented

    @abc.abstractmethod
    def terminate(self):
        ''' release all resource after running
        '''
        return NotImplemented

    @abc.abstractmethod
    def on_epoch(self):
        ''' define the behavior in an epoch'''
        return NotImplemented
    
    def train(self):
        ''' define the behavior of training'''
        self.total_time_step = 0
        self.initialize()
        for self.epoch_number in range(self.solver_config["start_epoch_number"], self.solver_config["end_epoch_number"] + 1):
            self.on_epoch()
        self.terminate()


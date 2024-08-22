import numpy as np
from cgi_drl.environment.observation_preprocessor import ObservationPreprocessor
from cgi_drl.visualization_tool.gif_maker import make_video

class DMLab30ObservationPreprocessor(ObservationPreprocessor):
    def __init__(self, config):
        self.use_normalize = config["use_normalize"]
        print("use_normalize: {}".format(self.use_normalize))
        self.observation_aggregator = config["observation_aggregator"]

    def process(self, observations, process_settings = None):       
        processed_observations = {}
        if self.use_normalize:
            processed_observations["observation_2d"] = np.asarray(observations, dtype=np.float32) / 255.0
        else:
            processed_observations["observation_2d"] = np.asarray(observations, dtype=np.float32)   
        return processed_observations

    def create_video(self, observations, filename):
        observations = observations["observation_2d"]
        observations = np.asarray(observations, dtype=np.float32)
        observations = np.reshape(observations, [-1, 4, 72, 96, 3])
        observations = observations[:,0,:,:]
        if self.use_normalize:
            observations *= 255.0
        make_video(observations, filename, duration=len(observations) * 4 / 30)

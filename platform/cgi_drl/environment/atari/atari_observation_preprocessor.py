import numpy as np
from cgi_drl.environment.observation_preprocessor import ObservationPreprocessor
from cgi_drl.visualization_tool.gif_maker import make_video

class AtariObservationPreprocessor(ObservationPreprocessor):
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
        observations = np.reshape(observations, [-1, 4, 84, 84])
        observations = observations[:,1:,:,:]
        observations = np.transpose(observations, [0, 2, 3, 1])
        if self.use_normalize:
            observations *= 255.0
        make_video(observations, filename, duration=len(observations) * 4 / 60)

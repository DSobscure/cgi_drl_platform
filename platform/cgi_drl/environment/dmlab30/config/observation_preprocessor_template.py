import numpy as np

class DefaultTemplate(dict):
    def __init__(self, config):
        self["use_normalize"] = config.get("use_normalize", True)

        def observation_aggregator(observation_name, previous_observatios, new_observatios, is_first=False):
            if observation_name == "observation_2d":
                if is_first:
                    return np.repeat([new_observatios], 4, axis=0)
                else:
                    return np.concatenate([previous_observatios[1:], [new_observatios]], axis=0)
            else:
                return new_observatios
        self["observation_aggregator"] = config.get("observation_aggregator", observation_aggregator)

        super(DefaultTemplate, self).__init__(config)
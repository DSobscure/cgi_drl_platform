import numpy as np

class FoodCollectorTemplate(dict):
    def __init__(self, config):
        def observation_aggregator(observation_name, previous_observatios, new_observatios, is_first=False):
            return np.asarray(new_observatios)
        self["observation_aggregator"] = config.get("observation_aggregator", observation_aggregator)

        super(FoodCollectorTemplate, self).__init__(config)
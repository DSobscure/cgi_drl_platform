class DefaultTemplate(dict):
    def __init__(self, config):
        self["horizon"] = config.get("horizon", 128)
        self["episode_sequence_size"] = config.get("episode_sequence_size", 8)
        self["use_return_as_advantage"] = config.get("use_return_as_advantage", False)
        
        super(DefaultTemplate, self).__init__(config)

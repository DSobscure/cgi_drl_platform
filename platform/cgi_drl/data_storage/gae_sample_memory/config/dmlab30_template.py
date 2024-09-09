class DefaultTemplate(dict):
    def __init__(self, config):
        self["horizon"] = config.get("horizon", 128)
        self["sequence_length"] = config.get("sequence_length", 1)
        self["use_return_as_advantage"] = config.get("use_return_as_advantage", False)
        
        super(DefaultTemplate, self).__init__(config)

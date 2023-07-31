class DefaultTemplate(dict):
    def __init__(self, config):
        self["available_actions"] = config.get("available_actions", [0, 1, 2, 3, 4, 5])
        super(DefaultTemplate, self).__init__(config)
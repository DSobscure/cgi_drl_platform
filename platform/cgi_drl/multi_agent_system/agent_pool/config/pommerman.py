class DefaultTemplate(dict):
    def __init__(self, config):
        self["existing_agents"] = config.get("existing_agents", [])
        super().__init__(config)
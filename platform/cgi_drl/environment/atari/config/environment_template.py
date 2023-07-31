class DefaultTemplate(dict):
    def __init__(self, config):
        self["environment_host_ip"] = config.get("environment_host_ip", "127.0.0.1")
        self["environment_host_port"] = config.get("environment_host_port", 30000)
        self["environment_wrapper_path"] = config.get("environment_wrapper_path", "cgi_drl.environment.atari.atari_environment_wrapper")
        self["environment_wrapper_class_name"] = config.get("environment_wrapper_class_name", "AtariEnvironmentWrapper")

        self["environment_count"] = config.get("environment_count", 4)
        self["episode_life"] = config.get("episode_life", False)

        self["environment_name"] = config.get("environment_name", "Breakout")
        self["environment_postfix"] = config.get("environment_postfix", "NoFrameskip-v4")
        self["environment_id"] = config.get("environment_id", "{}{}".format(self["environment_name"], self["environment_postfix"]))

        super(DefaultTemplate, self).__init__(config)
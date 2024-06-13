class DefaultTemplate(dict):
    def __init__(self, config):
        self["environment_host_ip"] = config.get("environment_host_ip", "127.0.0.1")
        self["environment_host_port"] = config.get("environment_host_port", 30000)
        self["environment_wrapper_path"] = config.get("environment_wrapper_path", "cgi_drl.environment.unity_gym.unity_environment_wrapper")
        self["environment_wrapper_class_name"] = config.get("environment_wrapper_class_name", "UnityEnvironmentWrapper")

        self["environment_count"] = config.get("environment_count", 4)

        super(DefaultTemplate, self).__init__(config)
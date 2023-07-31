class DefaultTemplate(dict):
    def __init__(self, config):
        self["environment_config"] = config.get("environment_config", {
            "environment_host_ip" : "127.0.0.1",
            "environment_host_port" : 30000,
            "environment_wrapper_path" : "cgi_drl.environment.pommerman.pommerman_environment_wrapper",
            "environment_wrapper_class_name" : "PommermanEnvironmentWrapper",
            "environment_count" : 4,
            "environment_id" : "OneVsOne-v0",
            "agent_names" : ["Player1", "Player2"],
            "max_steps" : 400
        })
        for key in config:
            self["environment_config"][key] = config[key]
        self["environment_wrapper_path"] = config.get("environment_wrapper_path", "cgi_drl.environment.distributed_framework.environment_requester")
        self["environment_wrapper_class_name"] = config.get("environment_wrapper_class_name", "EnvironmentRequester")

        super(DefaultTemplate, self).__init__(config)
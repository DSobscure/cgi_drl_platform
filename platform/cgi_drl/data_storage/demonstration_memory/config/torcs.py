class UAI2021Template(dict):
    def __init__(self, config):
        self["visual_observation_frame_count"] = config.get("visual_observation_frame_count", 4)
        self["is_use_internal_state"] = config.get("is_use_internal_state", False)

        folder_path_prefix = config.get("folder_path_prefix", "/root/playstyle_uai2021/torcs/testing")
        speed_styles = [60, 65, 70, 75, 80]
        noise_styles = [0, 1, 2, 3, 4]
        demo_pairs = []

        for speed_style in speed_styles:
            for noise_style in noise_styles:
                demo_pairs.append((
                    "{}/Speed{}N{}/".format(folder_path_prefix, speed_style, noise_style),
                    "Speed{}N{}".format(speed_style, noise_style)
                ))
        self["demo_pairs"] = config.get("demo_pairs", demo_pairs)

        super().__init__(config)
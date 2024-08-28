class UAI2021Template(dict):
    def __init__(self, config):
        self["visual_observation_frame_count"] = config.get("visual_observation_frame_count", 4)

        npz_folder_path_prefix = config.get("npz_folder_path_prefix", "/root/playstyle_uai2021/rgsk/testing")
        style_gropus = ["road", "outer", "nonitro", "nitro", "inner", "grass", "drift", "slowdown"]
        demo_pairs = []

        for style_gropu in style_gropus:
            for i in range(3):
                demo_pairs.append((
                    "{}/{}/{}".format(npz_folder_path_prefix, style_gropu, i),
                    "{}_Player{}".format(style_gropu, i + 1)
                ))
        self["demo_pairs"] = config.get("demo_pairs", demo_pairs)

        super().__init__(config)

class UAI2021Template29(dict):
    def __init__(self, config):
        self["visual_observation_frame_count"] = config.get("visual_observation_frame_count", 4)

        npz_folder_path_prefix = config.get("npz_folder_path_prefix", "/root/playstyle_uai2021/rgsk/training")
        demo_pairs = []

        for i in range(29):
            demo_pairs.append((
                "{}/{}".format(npz_folder_path_prefix, i),
                "Player{}".format(i)
            ))
        self["demo_pairs"] = config.get("demo_pairs", demo_pairs)

        super().__init__(config)
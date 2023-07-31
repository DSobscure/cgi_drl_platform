class UAI2021Template(dict):
    def __init__(self, config):
        self["visual_observation_frame_count"] = config.get("visual_observation_frame_count", 4)

        npz_folder_path_prefix = config.get("npz_folder_path_prefix", "/root/playstyle_uai2021/atari/testing/Breakout")
        algorithms = ["DQN", "C51", "Rainbow", "IQN"]
        # algorithms = ["DQN"]
        demo_pairs = []

        for algorithm in algorithms:
            for i in range(5):
                # for j in range(1):
                #     demo_pairs.append((
                #         "{}/{}/model{}/{}".format(npz_folder_path_prefix, algorithm, i + 1, j + 1),
                #         "{}_Model{}_Sample{}".format(algorithm, i + 1, j + 1)
                #     ))
                demo_pairs.append((
                    "{}/{}/{}".format(npz_folder_path_prefix, algorithm, i + 1),
                    "{}_Model{}".format(algorithm, i + 1)
                ))
        self["demo_pairs"] = config.get("demo_pairs", demo_pairs)

        super().__init__(config)
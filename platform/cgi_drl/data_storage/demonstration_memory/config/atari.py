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

class UAI2021TrainingTemplate(dict):
    def __init__(self, config):
        self["visual_observation_frame_count"] = config.get("visual_observation_frame_count", 4)
        super().__init__(config)

class ICLR2024Template(dict):
    def __init__(self, config):
        self["visual_observation_frame_count"] = config.get("visual_observation_frame_count", 4)

        npz_folder_path_prefix = config.get("npz_folder_path_prefix", "/root/playstyle_iclr2024/atari/Breakout/IQN/model1")
        diversity_count = 4
        episode_count = 100
        demo_pairs = []

        for i_diversity in range(diversity_count):
            for i_episode in range(episode_count):
                demo_pairs.append((
                    "{}/diversity_{}/episode{}".format(npz_folder_path_prefix, i_diversity, i_episode + 1),
                    "diversity_{}_episode{}".format(i_diversity, i_episode + 1)
                ))
        self["demo_pairs"] = config.get("demo_pairs", demo_pairs)

        super().__init__(config)
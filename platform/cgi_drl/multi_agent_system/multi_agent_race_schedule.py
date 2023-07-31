class MultiAgentRaceSchedule:
    def __init__(self, config):
        # For each stage
        self._training_stage = []  # A list of training stage info
        self._eval_stage = []
        self._training_config = []
        self._eval_config = []
        for stage_list in config["train"]:
            match_infos = []
            match_configs = []
            for sunphase_list in stage_list:
                match_info = {}
                match_config = {}
                for match in sunphase_list:
                    if match == "config":
                        match_config = sunphase_list[match]
                    else:
                        match_info[match] = sunphase_list[match]
                match_infos.append(match_info)
                match_configs.append(match_config)
            self._training_stage.append(match_infos)
            self._training_config.append(match_configs)

        for stage_list in config["eval"]:
            match_infos = []
            match_configs = []
            for sunphase_list in stage_list:
                match_info = {}
                match_config = {}
                for match in sunphase_list:
                    if match == "config":
                        match_config = sunphase_list[match]
                    else:
                        match_info[match] = sunphase_list[match]
                match_infos.append(match_info)
                match_configs.append(match_config)
            self._eval_stage.append(match_infos)
            self._eval_config.append(match_configs)
        
        self.subphase_counts = {
            "train": [len(stage) for stage in config["train"]],
            "eval": [len(stage) for stage in config["eval"]],
        }

    def get_race_participant_mapping(self, **kwargs):
        mode = kwargs["mode"]
        stage = kwargs["stage"]
        subphase = kwargs["subphase"]

        if mode == "train":
            if stage >= len(self._training_stage) or subphase >= len(self._training_stage[stage]):
                raise RuntimeError(f"Invalid stage({stage}) or subphase({subphase})")
            info = self._training_stage[stage][subphase]
        elif mode == "eval":
            if stage >= len(self._eval_stage) or subphase >= len(self._eval_stage[stage]):
                raise RuntimeError(f"Invalid stage({stage}) or subphase({subphase})")
            info = self._eval_stage[stage][subphase]
        else:
            raise RuntimeError(f"Unknown mode {mode}")
        return info

    def get_race_config(self, **kwargs):
        mode = kwargs["mode"]
        stage = kwargs["stage"]
        subphase = kwargs["subphase"]

        if mode == "train":
            if stage >= len(self._training_stage) or subphase >= len(self._training_stage[stage]):
                raise RuntimeError(f"Invalid stage({stage}) or subphase({subphase})")
            info = self._training_config[stage][subphase]
        elif mode == "eval":
            if stage >= len(self._eval_stage) or subphase >= len(self._eval_stage[stage]):
                raise RuntimeError(f"Invalid stage({stage}) or subphase({subphase})")
            info = self._eval_config[stage][subphase]
        else:
            raise RuntimeError(f"Unknown mode {mode}")
        return info

    def get_subphase_counts(self):
        return self.subphase_counts

if __name__ == "__main__":
    config = {
        "train": [
            # Stage 0
            [
                # Subphase 0
                {"Player1": "alice_solver", "Player2": "roam_solver","config":{"BotID":123}},
                # Subphase 1
                {"Player1": "roam_solver", "Player2": "bob_solver"},
            ],
            # Stage 1
            [
                # Subphase 0
                {"Player1": "alice_solver", "Player2": "bob_solver_#"},
                # Subphase 1
                {"Player1": "alice_solver_#", "Player2": "bob_solver"},
            ],
        ],
        "eval": [
            [
                {"Player1": "alice_solver", "Player2": "roam_solver"},
                {"Player1": "roam_solver", "Player2": "bob_solver"},
            ],
            [
                {"Player1": "alice_solver", "Player2": "bob_solver_#"},
                {"Player1": "alice_solver_#", "Player2": "bob_solver"},
            ],
        ],
    }
    schedule = MultiAgentRaceSchedule(config)
    print("training")
    race_participant_mapping = schedule.get_race_participant_mapping(mode="train", stage=0, subphase=0)
    config = schedule.get_race_config(mode="train", stage=0, subphase=0)
    print("0-0", race_participant_mapping["Player1"] + " vs " + race_participant_mapping["Player2"], "config:", config)

    race_participant_mapping = schedule.get_race_participant_mapping(mode="train", stage=0, subphase=1)
    config = schedule.get_race_config(mode="train", stage=0, subphase=1)
    print("0-1", race_participant_mapping["Player1"] + " vs " + race_participant_mapping["Player2"], "config:", config)

    race_participant_mapping = schedule.get_race_participant_mapping(mode="train", stage=1, subphase=0)
    print("1-0", race_participant_mapping["Player1"] + " vs " + race_participant_mapping["Player2"])

    race_participant_mapping = schedule.get_race_participant_mapping(mode="train", stage=1, subphase=1)
    print("1-1", race_participant_mapping["Player1"] + " vs " + race_participant_mapping["Player2"])

    print("evaluation")
    race_participant_mapping = schedule.get_race_participant_mapping(mode="eval", stage=0, subphase=0)
    print("bh-0-0", race_participant_mapping["Player1"] + " vs " + race_participant_mapping["Player2"])

    race_participant_mapping = schedule.get_race_participant_mapping(mode="eval", stage=0, subphase=1)
    print("bh-0-1", race_participant_mapping["Player1"] + " vs " + race_participant_mapping["Player2"])

    race_participant_mapping = schedule.get_race_participant_mapping(mode="eval", stage=1, subphase=0)
    print("bh-1-0", race_participant_mapping["Player1"] + " vs " + race_participant_mapping["Player2"])
    race_participant_mapping = schedule.get_race_participant_mapping(mode="eval", stage=1, subphase=1)
    print("bh-1-1", race_participant_mapping["Player1"] + " vs " + race_participant_mapping["Player2"])

    # Should raise error
    race_participant_mapping = schedule.get_race_participant_mapping(mode="eval", stage=0, eval_type="rating", subphase=0)

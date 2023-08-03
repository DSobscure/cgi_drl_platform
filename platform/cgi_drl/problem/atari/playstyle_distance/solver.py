import importlib
import numpy as np
import tensorflow as tf
import random
import math

def launch(problem_config):
    load = problem_config["load_function"]

    # setup encoder
    hsd_config = load(*problem_config["hsd"])
    problem_config["hsd"] = hsd_config
    from cgi_drl.representation_model.hsd.encoder_trainer_uai2021 import EncoderTrainer
    encoder = EncoderTrainer(hsd_config)

    # setup demo
    demo_config = load(*problem_config["demo"])
    problem_config["demo"] = demo_config
    from cgi_drl.data_storage.demonstration_memory.atari_demonstration_memory import AtariDemonstrationMemory

    # setup tensorflow
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=tf_config)
    encoder.set_session(sess)
    
    sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
    encoder.load(problem_config["load_encoder_model_path"])

    sample_count = 512
    repeat_count = 100
    code_level = 0
    threshold_count = 1

    playstyle_dataset = {}
    for demo_pair in demo_config["demo_pairs"]:
        demo_config["npz_folder"] = demo_pair[0]
        memory = AtariDemonstrationMemory(demo_config)
        playstyle_dataset[demo_pair[1]] = []

        for batch in memory.sample_all_batch(128):
            observation_batch, action_batch = batch[0], batch[1]
            states = encoder.get_discrite_latent_codes({"visual" : observation_batch }, code_level)
            for i_state in range(len(action_batch)):
                playstyle_dataset[demo_pair[1]].append((states[i_state], np.asarray(action_batch[i_state])))
                # playstyle_dataset[demo_pair[1]].append(("none", np.asarray(action_batch[i_state])))

    algorithm_styles = ["DQN", "C51", "Rainbow", "IQN"]
    # algorithm_styles = ["DQN"]
    algorithm_styles = get_compound_style(algorithm_styles)
    model_accurate_list = []

    for i in range(repeat_count):
        model_accurate = 0
        print("Repated: {}/{}".format(i, repeat_count), end='\r')

        for test_style in algorithm_styles:
            playstyle_distances = {}
            for candidate_style in algorithm_styles:
                if test_style == candidate_style:
                    ls = sample_a_list_without_replacement(playstyle_dataset[test_style], sample_count * 2)
                    test_list = ls[:sample_count]
                    candidate_list = ls[sample_count:]
                else:
                    test_list = sample_a_list_without_replacement(playstyle_dataset[test_style], sample_count)
                    candidate_list = sample_a_list_without_replacement(playstyle_dataset[candidate_style], sample_count)              
                playstyle_distance = compute_similarity(test_list, candidate_list, threshold_count)
                playstyle_distances[candidate_style] = playstyle_distance
            sorted_by_similarity = sorted(playstyle_distances.items(), key=lambda d: d[1]) 
            if test_style == sorted_by_similarity[0][0]:
                model_accurate += 1
            # print(test_style, "in", [s[0]for s in sorted_by_similarity])
        model_accurate_list.append(model_accurate / len(algorithm_styles))

    print()
    print("* model: {:.2f}±{:.2f}%".format(np.mean(model_accurate_list) * 100, np.std(model_accurate_list) * 100))

def calculate_w2(act1, act2):
    mu1 = act1.mean(axis=0)
    mu2 = act2.mean(axis=0)
    return np.linalg.norm(mu1 - mu2, 2)

def sample_a_list_without_replacement(full_list, size):
    return random.sample(full_list, k=size)

def extract_style_info(style_list):
    style_info = {
        "state_set" : set(),
        "state_count" : {},
        "actions" : {},
    }

    for pair in style_list:
        state, action = pair
        style_info["state_set"].add(state)
        if state not in style_info["state_count"]:
            style_info["state_count"][state] = 0
        style_info["state_count"][state] += 1
        if state not in style_info["actions"]:
            style_info["actions"][state] = []
        style_info["actions"][state].append(action)

    return style_info

def compute_similarity(style_A_list, style_B_list, threshold_count):
    style_A = extract_style_info(style_A_list)
    style_B = extract_style_info(style_B_list)

    intersection_state = style_A["state_set"].intersection(style_B["state_set"])
    overlapping_count_in_A, overlapping_count_in_B = 0, 0
    valid_state_count = 0

    for s in intersection_state:
        if style_A["state_count"][s] < threshold_count or style_B["state_count"][s] < threshold_count:
            continue
        overlapping_count_in_A += style_A["state_count"][s]
        overlapping_count_in_B += style_B["state_count"][s]

    playstyle_distance = np.double(0.0)
    for s in intersection_state:
        if style_A["state_count"][s] < threshold_count or style_B["state_count"][s] < threshold_count:
            continue
        distance = calculate_w2(np.asarray(style_A["actions"][s] + style_A["actions"][s]), np.asarray(style_B["actions"][s] + style_B["actions"][s]))
        playstyle_distance += (distance * style_A["state_count"][s] / overlapping_count_in_A + distance * style_B["state_count"][s] / overlapping_count_in_B) / 2
        valid_state_count += 1
    
    if valid_state_count == 0:
        playstyle_distance = math.inf
    return playstyle_distance

def get_compound_style(style_names):
    indice = [1, 2, 3, 4, 5]
    compound_styles = []
    for style in style_names:
        for index in indice:
            compound_styles.append("{}_Model{}".format(style, index))
    return compound_styles
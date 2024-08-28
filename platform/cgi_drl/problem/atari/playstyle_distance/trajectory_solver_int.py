import importlib
import numpy as np
import tensorflow as tf
import random
import math
import csv

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

    code_level = [-2, -1, 0]
    threshold_count = 1

    playstyle_dataset = {}
    for demo_pair in demo_config["demo_pairs"]:
        demo_config["npz_folder"] = demo_pair[0]
        memory = AtariDemonstrationMemory(demo_config)
        playstyle_dataset[demo_pair[1]] = []

        for batch in memory.sample_all_batch(128):
            observation_batch, action_batch = batch[0], batch[1]
            states = []
            for i_level in code_level:
                s = encoder.get_discrite_latent_codes({"visual" : observation_batch }, i_level)
                states.append(s)
            for i_state in range(len(action_batch)):
                playstyle_dataset[demo_pair[1]].append(([state[i_state] for state in states], np.asarray(action_batch[i_state])))
    
    reference_styles = ["diversity_0","diversity_1","diversity_2","diversity_3"]
    reference_trajectories = get_reference_trajectories_style()
    reference_style_infos = {}
    for style in reference_styles:
        style_list = []
        for trajectory_style in reference_trajectories:
            if trajectory_style[:11] == style:
                style_list += playstyle_dataset[trajectory_style]
        reference_style_infos[style] = extract_style_info(style_list)

    target_trajectories = get_target_trajectories_style()
    target_style_infos = {}
    for trajectory_style in target_trajectories:
        target_style_infos[trajectory_style] = extract_style_info(playstyle_dataset[trajectory_style])

    model_accurate = 0
    for target_trajectory in target_trajectories:
        playstyle_similar_probabilities = {}
        target_style_info = target_style_infos[target_trajectory]
        all_distances = []
        for reference_style in reference_styles:
            reference_style_info = reference_style_infos[reference_style]
            playstyle_distances, jaccard_index, state_space = compute_similarity(target_style_info, reference_style_info, threshold_count)
            all_distances = all_distances + playstyle_distances
        if len(all_distances) > 0:
            distance_mean = np.mean(all_distances)
        for reference_style in reference_styles:
            reference_style_info = reference_style_infos[reference_style]
            playstyle_distances, jaccard_index, state_space = compute_similarity(target_style_info, reference_style_info, threshold_count)
            similar_probability = np.double(0.0)
            for d in playstyle_distances:
                d = d / (distance_mean + 1e-8)
                similar_probability += np.exp(-d) / state_space
            playstyle_similar_probabilities[reference_style] = similar_probability
        sorted_by_similarity = sorted(playstyle_distances.items(), key=lambda d: d[1], reverse=True) 
        if target_trajectory[:11] == sorted_by_similarity[0][0]:
            model_accurate += 1
    print("accuracy: {:.2f}%".format(model_accurate/len(target_trajectories) * 100))

def calculate_w2(act1, act2):
    mu1 = act1.mean(axis=0)
    mu2 = act2.mean(axis=0)
    return np.linalg.norm(mu1 - mu2, 2)

def calculate_bhattacharyya_coefficient(act1, act2):
    mu1 = act1.mean(axis=0)
    mu2 = act2.mean(axis=0)
    return np.sum(np.sqrt(mu1 * mu2), axis=0)

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
        for i_s in range(len(state)):
            s = state[i_s]
            style_info["state_set"].add(s)
            if s not in style_info["state_count"]:
                style_info["state_count"][s] = 0
            style_info["state_count"][s] += 1
            if s not in style_info["actions"]:
                style_info["actions"][s] = []
            style_info["actions"][s].append(action)

    return style_info

def compute_similarity(style_A, style_B, threshold_count):
    intersection_state = style_A["state_set"].intersection(style_B["state_set"])
    # state_space = len(intersection_state)
    state_space = len(style_A["state_set"].union(style_B["state_set"]))
    valid_state_count = 0

    # similar_probability = np.double(0.0)
    distances = []
    for s in intersection_state:
        if style_A["state_count"][s] < threshold_count or style_B["state_count"][s] < threshold_count:
            continue
        distance = calculate_w2(np.asarray(style_A["actions"][s] + style_A["actions"][s]), np.asarray(style_B["actions"][s] + style_B["actions"][s]))
        distances.append(distance)
        valid_state_count += 1
    jaccard_index = valid_state_count / state_space
    return distances, jaccard_index, valid_state_count

def get_reference_trajectories_style():
    styles = []
    for i_diversity in range(4):
        for i_episode in range(20):
            styles.append("diversity_{}_episode{}".format(i_diversity, i_episode + 1))
    return styles

def get_target_trajectories_style():
    styles = []
    for i_diversity in range(4):
        for i_episode in range(80):
            styles.append("diversity_{}_episode{}".format(i_diversity, i_episode + 21))
    return styles

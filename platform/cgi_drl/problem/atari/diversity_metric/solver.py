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
    trajectory_infos = {}
    for dataset_name in playstyle_dataset:
        trajectory_infos[dataset_name] = extract_style_info(playstyle_dataset[dataset_name])

    result_path = problem_config["result_path"] + "_t_005"

    diversity_count = 4
    episode_count = 100
    similarity_threshold = 0.05

    for i_diversity in range(diversity_count):
        with open(result_path + "_diversity{}.csv".format(i_diversity), "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            observed_trajectories = {}
            diversed_episode_count = 0
            for i_episode in range(episode_count):
                trajectory_name = "diversity_{}_episode{}".format(i_diversity, i_episode + 1)
                if len(observed_trajectories) == 0:
                    diversed_episode_count += 1
                else:
                    similarity = get_trajectory_similarity(observed_trajectories, trajectory_infos[trajectory_name])
                    if max(similarity.values()) < similarity_threshold:
                        diversed_episode_count += 1
                observed_trajectories[trajectory_name] = trajectory_infos[trajectory_name]
                print("Diversity {}: {}/{}".format(i_diversity, diversed_episode_count, i_episode + 1), end='\r')
                writer.writerow([diversed_episode_count])
            print("Diversity {}: {}/{}".format(i_diversity, diversed_episode_count, i_episode + 1))


def get_trajectory_similarity(observed_trajectories, new_trajectory):
    similarity = {}
    all_style_distances = {}
    all_state_spaces = {}
    all_distances = []
    for trajectory_name in observed_trajectories:
        playstyle_distances, state_space = compute_similarity(new_trajectory, observed_trajectories[trajectory_name])
        all_style_distances[trajectory_name] = playstyle_distances
        all_state_spaces[trajectory_name] = state_space
        all_distances = all_distances + playstyle_distances
    if len(all_distances) > 0:
        distance_mean = np.mean(all_distances)
    for trajectory_name in observed_trajectories:
        similar_probability = np.double(0.0)
        state_space = all_state_spaces[trajectory_name]
        for d in all_style_distances[trajectory_name]:
            d = d / (distance_mean + 1e-8)
            similar_probability += np.exp(-d) / state_space
        similarity[trajectory_name] = similar_probability
    return similarity

def calculate_w2(act1, act2):
    mu1 = act1.mean(axis=0)
    mu2 = act2.mean(axis=0)
    return np.linalg.norm(mu1 - mu2, 2)

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

def compute_similarity(style_A, style_B):
    intersection_state = style_A["state_set"].intersection(style_B["state_set"])
    state_space = len(style_A["state_set"].union(style_B["state_set"]))
    valid_state_count = 0

    distances = []
    for s in intersection_state:
        distance = calculate_w2(np.asarray(style_A["actions"][s] + style_A["actions"][s]), np.asarray(style_B["actions"][s] + style_B["actions"][s]))
        distances.append(distance)
        valid_state_count += 1
    return distances, state_space

def get_compound_style(style_names):
    indice = [1, 2, 3, 4, 5]
    compound_styles = []
    for style in style_names:
        for index in indice:
            compound_styles.append("{}_Model{}".format(style, index))
    return compound_styles
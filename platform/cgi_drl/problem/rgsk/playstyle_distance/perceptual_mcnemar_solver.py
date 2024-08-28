import importlib
import numpy as np
import tensorflow as tf
import random
import math
import csv
from scipy.stats.distributions import chi2

def action_to_one_hot(action):
    one_hot_action = np.zeros(27)
    one_hot_action[action[0] * 9 + action[1] * 3 + action[2]] = 1
    return one_hot_action

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
    from cgi_drl.data_storage.demonstration_memory.unity_demonstration_memory import UnityDemonstrationMemory

    # setup tensorflow
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=tf_config)
    encoder.set_session(sess)
    
    sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
    encoder.load(problem_config["load_encoder_model_path"])

    # sample_count = 1024
    max_sample_count_power = 0
    repeat_count = 100
    code_level = [0]
    threshold_count = 1

    playstyle_dataset = {}
    for demo_pair in demo_config["demo_pairs"]:
        demo_config["npz_folder"] = demo_pair[0]
        memory = UnityDemonstrationMemory(demo_config)
        playstyle_dataset[demo_pair[1]] = []

        for batch in memory.sample_all_batch(128):
            observation_batch, action_batch = batch[0], batch[1]
            states = []
            for i_level in code_level:
                s = encoder.get_discrite_latent_codes({"visual" : observation_batch / 255}, i_level)
                states.append(s)
            for i_state in range(len(action_batch)):
                playstyle_dataset[demo_pair[1]].append(([state[i_state] for state in states], action_to_one_hot(action_batch[i_state])))

    style_groups = ["road", "outer", "nonitro", "nitro", "inner", "grass", "drift", "slowdown"]
    all_styles = get_compound_style(style_groups)
    # all_styles = get_compound_style()

    result_path = problem_config["result_path"]

    for i_sample_power in range(max_sample_count_power + 1):
        sample_count = 1024 # 2 ** i_sample_power

        predict_count = 0
        playstyle_distance_correct_count = 0
        playstyle_int_similarity_correct_count = 0
        playstyle_int_bc_similarity_correct_count = 0

        pd_vs_pis = [0, 0]
        pd_vs_pibcs = [0, 0]

        for i in range(repeat_count):
            print("Repated: {}/{}".format(i, repeat_count), end='\r')
            for test_style in all_styles:
                playstyle_distances = {}
                playstyle_int_similarity = {}
                playstyle_int_bc_similarity = {}

                double_test_list = sample_a_list_without_replacement(playstyle_dataset[test_style], sample_count * 2)

                query_style_info = extract_style_info(double_test_list[sample_count:])

                all_style_distances = {}
                all_state_spaces = {}
                all_distances = []

                for candidate_style in all_styles:
                    if test_style == candidate_style:
                        candidate_style_info = extract_style_info(double_test_list[:sample_count])
                    else:
                        candidate_style_info = extract_style_info(sample_a_list_without_replacement(playstyle_dataset[candidate_style], sample_count))     
                    playstyle_distances[candidate_style] = compute_similarity(query_style_info, candidate_style_info)
                    playstyle_int_bc_similarity[candidate_style] = compute_similarity_int_bc(query_style_info, candidate_style_info)
                    _playstyle_distances, _state_space = compute_similarity_int(query_style_info, candidate_style_info)
                    all_style_distances[candidate_style] = _playstyle_distances
                    all_state_spaces[candidate_style] = _state_space
                    all_distances = all_distances + _playstyle_distances
                distance_mean = np.mean(all_distances)
                for candidate_style in all_styles:
                    similar_probability = np.double(0.0)
                    state_space = all_state_spaces[candidate_style]
                    for d in all_style_distances[candidate_style]:
                        d = d / (distance_mean + 1e-8)
                        similar_probability += np.exp(-d) / state_space
                    playstyle_int_similarity[candidate_style] = similar_probability

                predict_count += 1
                sorted_by_similarity = sorted(playstyle_distances.items(), key=lambda d: d[1]) 
                if test_style == sorted_by_similarity[0][0]:
                    playstyle_distance_correct_count += 1

                    sorted_by_similarity = sorted(playstyle_int_similarity.items(), key=lambda d: d[1], reverse=True) 
                    if test_style == sorted_by_similarity[0][0]:
                        playstyle_int_similarity_correct_count += 1
                    else:
                        pd_vs_pis[0] += 1

                    sorted_by_similarity = sorted(playstyle_int_bc_similarity.items(), key=lambda d: d[1], reverse=True) 
                    if test_style == sorted_by_similarity[0][0]:
                        playstyle_int_bc_similarity_correct_count += 1
                    else:
                        pd_vs_pibcs[0] += 1

                else:
                    sorted_by_similarity = sorted(playstyle_int_similarity.items(), key=lambda d: d[1], reverse=True) 
                    if test_style == sorted_by_similarity[0][0]:
                        playstyle_int_similarity_correct_count += 1
                        pd_vs_pis[1] += 1

                    sorted_by_similarity = sorted(playstyle_int_bc_similarity.items(), key=lambda d: d[1], reverse=True) 
                    if test_style == sorted_by_similarity[0][0]:
                        playstyle_int_bc_similarity_correct_count += 1
                        pd_vs_pibcs[1] += 1

        print()
        print("under {} sample size".format(sample_count))
        print(f"Playstyle Distance accuracy: {playstyle_distance_correct_count/predict_count}")
        print(f"Playstyle Int Similarity accuracy: {playstyle_int_similarity_correct_count/predict_count}, p-value: {chi2.sf((pd_vs_pis[0]-pd_vs_pis[1]) ** 2 / (pd_vs_pis[0]+pd_vs_pis[1] + 0.000001),1)}")
        print(f"Playstyle Int BC Similarity accuracy: {playstyle_int_bc_similarity_correct_count/predict_count}, p-value: {chi2.sf((pd_vs_pibcs[0]-pd_vs_pibcs[1]) ** 2 / (pd_vs_pibcs[0]+pd_vs_pibcs[1] + 0.000001),1)}")


def calculate_w2(act1, act2):
    act1 = np.asarray(act1)
    act2 = np.asarray(act2)
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

def compute_similarity(style_A, style_B):
    intersection_state = style_A["state_set"].intersection(style_B["state_set"])
    overlapping_count_in_A, overlapping_count_in_B = 0, 0
    valid_state_count = 0

    for s in intersection_state:
        overlapping_count_in_A += style_A["state_count"][s]
        overlapping_count_in_B += style_B["state_count"][s]

    playstyle_distance = np.double(0.0)
    for s in intersection_state:
        distance = calculate_w2(np.asarray(style_A["actions"][s] + style_A["actions"][s]), np.asarray(style_B["actions"][s] + style_B["actions"][s]))
        playstyle_distance += (distance * style_A["state_count"][s] / overlapping_count_in_A + distance * style_B["state_count"][s] / overlapping_count_in_B) / 2
        valid_state_count += 1
    
    if valid_state_count == 0:
        playstyle_distance = math.inf
    return playstyle_distance

def compute_similarity_int(style_A, style_B):
    intersection_state = style_A["state_set"].intersection(style_B["state_set"])
    # state_space = len(intersection_state)
    state_space = len(style_A["state_set"].union(style_B["state_set"]))
    valid_state_count = 0

    # similar_probability = np.double(0.0)
    distances = []
    for s in intersection_state:
        distance = calculate_w2(np.asarray(style_A["actions"][s] + style_A["actions"][s]), np.asarray(style_B["actions"][s] + style_B["actions"][s]))
        distances.append(distance)
        # similar_probability += np.exp(-distance * 10) / state_space
        # similar_probability += calculate_bhattacharyya_coefficient(np.asarray(style_A["actions"][s] + style_A["actions"][s]), np.asarray(style_B["actions"][s] + style_B["actions"][s])) / state_space
        valid_state_count += 1
    return distances, valid_state_count

def compute_similarity_int_bc(style_A, style_B):
    intersection_state = style_A["state_set"].intersection(style_B["state_set"])
    # state_space = len(intersection_state)
    state_space = len(style_A["state_set"].union(style_B["state_set"]))
    valid_state_count = 0

    similar_probability = np.double(0.0)
    for s in intersection_state:
        similar_probability += calculate_bhattacharyya_coefficient(np.asarray(style_A["actions"][s] + style_A["actions"][s]), np.asarray(style_B["actions"][s] + style_B["actions"][s]))
        valid_state_count += 1
    if valid_state_count != 0:
        similar_probability /= valid_state_count
    return similar_probability

def get_compound_style(style_names):
    indice = [1, 2, 3]
    compound_styles = []
    for style in style_names:
        for index in indice:
            compound_styles.append("{}_Player{}".format(style, index))
    return compound_styles
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

    # sample_count = 256
    max_sample_count_power = 9
    repeat_count = 100
    code_level = [0]
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

    algorithm_styles = ["DQN", "C51", "Rainbow", "IQN"]
    # algorithm_styles = ["DQN"]
    all_styles = get_compound_style(algorithm_styles)

    result_path = problem_config["result_path"]
    result_path += "code0"
    with open(result_path + "_playstyle_intersection_similarity.csv", "w", newline="") as distance_csvfile, open(result_path + "_playstyle_intersection_similarity_iou.csv", "w", newline="") as iou_csvfile:
        distance_writer = csv.writer(distance_csvfile)
        iou_writer = csv.writer(iou_csvfile)

        for i_sample_power in range(max_sample_count_power + 1):
            sample_count = 2 ** i_sample_power

            model_accurate_list = []
            jaccard_index_list = []

            for i in range(repeat_count):
                model_accurate = 0
                print("Repated: {}/{}".format(i, repeat_count), end='\r')

                for test_style in all_styles:
                    playstyle_similar_probabilities = {}
                    jaccard_indexes = []
                    double_test_list = sample_a_list_without_replacement(playstyle_dataset[test_style], sample_count * 2)
                    test_style_info = extract_style_info(double_test_list[sample_count:])
                    all_style_infos = {}
                    for candidate_style in all_styles:
                        if test_style == candidate_style:
                            all_style_infos[candidate_style] = extract_style_info(double_test_list[:sample_count])
                        else:
                            all_style_infos[candidate_style] = extract_style_info(sample_a_list_without_replacement(playstyle_dataset[candidate_style], sample_count))
                    all_style_distances = {}
                    all_state_spaces = {}
                    all_distances = []
                    for candidate_style in all_styles:
                        playstyle_disrances, jaccard_index, state_space = compute_similarity(test_style_info, all_style_infos[candidate_style], threshold_count)
                        all_style_distances[candidate_style] = playstyle_disrances
                        all_state_spaces[candidate_style] = state_space
                        all_distances = all_distances + playstyle_disrances
                        # playstyle_similar_probabilities[candidate_style] = playstyle_similar_probability
                        jaccard_indexes.append(jaccard_index)
                    if len(all_distances) > 0:
                        distance_mean = np.mean(all_distances)
                        distance_std = np.std(all_distances)
                    for candidate_style in all_styles:
                        similar_probability = np.double(0.0)
                        state_space = all_state_spaces[candidate_style]
                        for d in all_style_distances[candidate_style]:
                            # d = max(0, 1 + (d - distance_mean) / (distance_std + 1e-8) / 2)
                            d = d / (distance_mean + 1e-8)
                            similar_probability += np.exp(-d) / state_space
                        playstyle_similar_probabilities[candidate_style] = similar_probability
                    sorted_by_similarity = sorted(playstyle_similar_probabilities.items(), key=lambda d: d[1], reverse=True) 
                    if test_style == sorted_by_similarity[0][0]:
                        model_accurate += 1
                    # print(test_style, "in", [s[0]for s in sorted_by_similarity])
                model_accurate_list.append(model_accurate / len(all_styles))
                jaccard_index_list.append(np.mean(jaccard_indexes))

            print()
            print("under {} sample size, accuracy: {:.2f}%, jaccard index: {:.2f}%".format(sample_count, np.mean(model_accurate_list) * 100,  np.mean(jaccard_index_list) * 100))
            distance_writer.writerow(model_accurate_list)
            iou_writer.writerow(jaccard_index_list)

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
        # similar_probability += np.exp(-distance * 10) / state_space
        # similar_probability += calculate_bhattacharyya_coefficient(np.asarray(style_A["actions"][s] + style_A["actions"][s]), np.asarray(style_B["actions"][s] + style_B["actions"][s])) / state_space
        valid_state_count += 1
    jaccard_index = valid_state_count / state_space
    return distances, jaccard_index, valid_state_count

def get_compound_style(style_names):
    indice = [1, 2, 3, 4, 5]
    compound_styles = []
    for style in style_names:
        for index in indice:
            compound_styles.append("{}_Model{}".format(style, index))
    return compound_styles
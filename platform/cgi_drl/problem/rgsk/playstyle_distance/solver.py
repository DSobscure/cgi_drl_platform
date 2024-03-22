import importlib
import numpy as np
import tensorflow as tf
import random
import math
import csv

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
    max_sample_count_power = 10
    repeat_count = 100
    code_level = [-2, -1, 0]
    threshold_count = 2

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
    result_path += "v1_t2"
    with open(result_path + "_playstyle_distance.csv", "w", newline="") as distance_csvfile:
        distance_writer = csv.writer(distance_csvfile)

        for i_sample_power in range(max_sample_count_power + 1):
            sample_count = 2 ** i_sample_power

            model_accurate_list = []

            for i in range(repeat_count):
                model_accurate = 0
                print("Repated: {}/{}".format(i, repeat_count), end='\r')

                for test_style in all_styles:
                    playstyle_distances = {}
                    double_test_list = sample_a_list_without_replacement(playstyle_dataset[test_style], sample_count * 2)
                    test_style_info = extract_style_info(double_test_list[sample_count:], len(code_level))
                    all_style_infos = {}
                    for candidate_style in all_styles:
                        if test_style == candidate_style:
                            all_style_infos[candidate_style] = extract_style_info(double_test_list[:sample_count], len(code_level))
                        else:
                            all_style_infos[candidate_style] = extract_style_info(sample_a_list_without_replacement(playstyle_dataset[candidate_style], sample_count), len(code_level))     
                        playstyle_distance = compute_similarity(test_style_info, all_style_infos[candidate_style], threshold_count, len(code_level))
                        playstyle_distances[candidate_style] = playstyle_distance
                    sorted_by_similarity = sorted(playstyle_distances.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        model_accurate += 1
                model_accurate_list.append(model_accurate / len(all_styles))

            print()
            print("under {} sample size, accuracy: {:.2f}%".format(sample_count, np.mean(model_accurate_list) * 100))
            distance_writer.writerow(model_accurate_list)

    # sample_count = 1024

    # model_accurate_list = []

    # for i in range(repeat_count):
    #     model_accurate = 0
    #     print("Repated: {}/{}".format(i, repeat_count), end='\r')

    #     for test_style in all_styles:
    #         playstyle_distances = {}
    #         double_test_list = sample_a_list_without_replacement(playstyle_dataset[test_style], sample_count * 2)
    #         test_style_info = extract_style_info(double_test_list[sample_count:])
    #         all_style_infos = {}
    #         for candidate_style in all_styles:
    #             if test_style == candidate_style:
    #                 all_style_infos[candidate_style] = extract_style_info(double_test_list[:sample_count])
    #             else:
    #                 all_style_infos[candidate_style] = extract_style_info(sample_a_list_without_replacement(playstyle_dataset[candidate_style], sample_count))     
    #             playstyle_distance, jaccard_index = compute_similarity(test_style_info, all_style_infos[candidate_style], threshold_count)
    #             playstyle_distances[candidate_style] = playstyle_distance
    #         sorted_by_similarity = sorted(playstyle_distances.items(), key=lambda d: d[1]) 
    #         if test_style == sorted_by_similarity[0][0]:
    #             model_accurate += 1
    #     model_accurate_list.append(model_accurate / len(all_styles))

    # print("under {} sample size, accuracy: {:.2f}%, std: {:.2f}%".format(sample_count, np.mean(model_accurate_list) * 100, np.std(model_accurate_list) * 100))

def calculate_w2(act1, act2):
    act1 = np.asarray(act1)
    act2 = np.asarray(act2)
    mu1 = act1.mean(axis=0)
    mu2 = act2.mean(axis=0)
    return np.linalg.norm(mu1 - mu2, 2)

def sample_a_list_without_replacement(full_list, size):
    return random.sample(full_list, k=size)

def extract_style_info(style_list, hierarchy_count):
    style_info = {
        "state_set" : [set() for _ in range(hierarchy_count)],
        "state_count" : [dict() for _ in range(hierarchy_count)],
        "actions" : [dict() for _ in range(hierarchy_count)],
    }

    for pair in style_list:
        state, action = pair
        for i_h in range(hierarchy_count):
            s = state[i_h]
            style_info["state_set"][i_h].add(s)
            if s not in style_info["state_count"][i_h]:
                style_info["state_count"][i_h][s] = 0
            style_info["state_count"][i_h][s] += 1
            if s not in style_info["actions"][i_h]:
                style_info["actions"][i_h][s] = []
            style_info["actions"][i_h][s].append(action)

    return style_info

def compute_similarity(style_A, style_B, threshold_count, hierarchy_count):
    intersection_state_list = []
    for i_h in range(hierarchy_count):
        intersection_state = style_A["state_set"][i_h].intersection(style_B["state_set"][i_h])
        intersection_state_list.append(intersection_state)
    overlapping_count_in_A, overlapping_count_in_B = np.zeros(hierarchy_count), np.zeros(hierarchy_count)
    valid_state_count = np.zeros(hierarchy_count)

    for i_h in range(hierarchy_count):
        for s in intersection_state_list[i_h]:
            if style_A["state_count"][i_h][s] < threshold_count or style_B["state_count"][i_h][s] < threshold_count:
                continue
            overlapping_count_in_A[i_h] += style_A["state_count"][i_h][s]
            overlapping_count_in_B[i_h] += style_B["state_count"][i_h][s]
            valid_state_count[i_h] += 1
            
    playstyle_distance = np.double(0.0)
    valid_state_count_sum = np.sum(valid_state_count)
    if valid_state_count_sum == 0:
        playstyle_distance = math.inf
    else:
        for i_h in range(hierarchy_count):
            for s in intersection_state_list[i_h]:
                if style_A["state_count"][i_h][s] < threshold_count or style_B["state_count"][i_h][s] < threshold_count:
                    continue
                distance = calculate_w2(style_A["actions"][i_h][s], style_B["actions"][i_h][s])
                distance = (distance * style_A["state_count"][i_h][s] / overlapping_count_in_A[i_h] + distance * style_B["state_count"][i_h][s] / overlapping_count_in_B[i_h]) / 2
                playstyle_distance += distance * valid_state_count[i_h] / valid_state_count_sum

    return playstyle_distance

def get_compound_style(style_names):
    indice = [1, 2, 3]
    compound_styles = []
    for style in style_names:
        for index in indice:
            compound_styles.append("{}_Player{}".format(style, index))
    return compound_styles

# def get_compound_style():
#     compound_styles = []
#     for index in range(29):
#         compound_styles.append("Player{}".format(index))
#     return compound_styles
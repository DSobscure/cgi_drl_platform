import importlib
import numpy as np
import tensorflow as tf
import random
import math
import csv

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

from scipy.stats.distributions import chi2

def launch(problem_config):
    load = problem_config["load_function"]

    # setup encoder
    hsd_config = load(*problem_config["hsd"])
    problem_config["hsd"] = hsd_config
    from cgi_drl.representation_model.hsd.encoder_trainer_uai2021_regression import EncoderTrainer
    encoder = EncoderTrainer(hsd_config)

    # setup demo
    demo_config = load(*problem_config["demo"])
    problem_config["demo"] = demo_config
    from cgi_drl.data_storage.demonstration_memory.torcs_demonstration_memory import TorcsDemonstrationMemory

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
    code_level = [-2, 0, -1]

    playstyle_dataset = {}
    for demo_pair in demo_config["demo_pairs"]:
        demo_config["demonstration_directory_prefix"] = demo_pair[0]
        demo_config["demonstration_directory"] = "{}demo_data/label".format(demo_pair[0])

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as demo_sess:
            memory = TorcsDemonstrationMemory(demo_sess, 128, demo_config)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(demo_sess, coord=coord)

            playstyle_dataset[demo_pair[1]] = []

            iteration_count = int(np.ceil(memory.size() / memory.batch_size))
            for i in range(iteration_count):
                print("Progress: {} - {}/{}".format(demo_pair[1], i, iteration_count), end='\r')
                observation_batch, action_batch = memory.sample_mini_batch()
                states = []
                for i_level in code_level:
                    s = encoder.get_discrite_latent_codes({"visual" : observation_batch / 255 }, i_level)
                    states.append(s)
                for i_state in range(len(action_batch)):
                    playstyle_dataset[demo_pair[1]].append(([state[i_state] for state in states], np.asarray(action_batch[i_state])))

            coord.request_stop()
            coord.join(threads)

    speed_styles = [60, 65, 70, 75, 80]
    noise_styles = [0, 1, 2, 3, 4]
    all_styles = get_compound_style(speed_styles, noise_styles)

    result_path = problem_config["result_path"]

    for i_sample_power in range(max_sample_count_power + 1):
        sample_count = 1024 # 2 ** i_sample_power

        predict_count = 0
        single_correct_count = 0
        space_2_20_t2_correct_count = 0
        space_2_20_t1_correct_count = 0
        space_res_t2_correct_count = 0
        space_res_t1_correct_count = 0
        space_mix_t2_correct_count = 0
        space_mix_t1_correct_count = 0

        mixt1_vs_single = [0, 0]
        mixt1_vs_2_20_t2 = [0, 0]
        mixt1_vs_2_20_t1 = [0, 0]
        mixt1_vs_res_t2 = [0, 0]
        mixt1_vs_res_t1 = [0, 0]
        mixt1_vs_mixt2 = [0, 0]

        for i in range(repeat_count):
            print("Repated: {}/{}".format(i, repeat_count), end='\r')
            for test_style in all_styles:
                playstyle_distances_single = {}
                playstyle_distances_2_20_t2 = {}
                playstyle_distances_2_20_t1 = {}
                playstyle_distances_res_t2 = {}
                playstyle_distances_res_t1 = {}
                playstyle_distances_mix_t2 = {}
                playstyle_distances_mix_t1 = {}
                double_test_list = sample_a_list_without_replacement(playstyle_dataset[test_style], sample_count * 2)

                # single
                query_style_info_single = extract_style_info(double_test_list[sample_count:], [0])
                query_style_info_2_20 = extract_style_info(double_test_list[sample_count:], [1])
                query_style_info_res = extract_style_info(double_test_list[sample_count:], [2])
                query_style_info_mix = extract_style_info(double_test_list[sample_count:], [0, 1, 2])

                for candidate_style in all_styles:
                    if test_style == candidate_style:
                        candidate_style_info_single = extract_style_info(double_test_list[:sample_count], [0])
                        candidate_style_info_2_20 = extract_style_info(double_test_list[:sample_count], [1])
                        candidate_style_info_res = extract_style_info(double_test_list[:sample_count], [2])
                        candidate_style_info_mix = extract_style_info(double_test_list[:sample_count], [0, 1, 2])
                    else:
                        candidate_style_info_single = extract_style_info(sample_a_list_without_replacement(playstyle_dataset[candidate_style], sample_count), [0])     
                        candidate_style_info_2_20 = extract_style_info(sample_a_list_without_replacement(playstyle_dataset[candidate_style], sample_count), [1])
                        candidate_style_info_res = extract_style_info(sample_a_list_without_replacement(playstyle_dataset[candidate_style], sample_count), [2])
                        candidate_style_info_mix = extract_style_info(sample_a_list_without_replacement(playstyle_dataset[candidate_style], sample_count), [0, 1, 2])
                    playstyle_distances_single[candidate_style] = compute_similarity(query_style_info_single, candidate_style_info_single, 1)
                    playstyle_distances_2_20_t2[candidate_style] = compute_similarity(query_style_info_2_20, candidate_style_info_2_20, 2)
                    playstyle_distances_2_20_t1[candidate_style] = compute_similarity(query_style_info_2_20, candidate_style_info_2_20, 1)
                    playstyle_distances_res_t2[candidate_style] = compute_similarity(query_style_info_res, candidate_style_info_res, 2)
                    playstyle_distances_res_t1[candidate_style] = compute_similarity(query_style_info_res, candidate_style_info_res, 1)
                    playstyle_distances_mix_t2[candidate_style] = compute_similarity(query_style_info_mix, candidate_style_info_mix, 2)
                    playstyle_distances_mix_t1[candidate_style] = compute_similarity(query_style_info_mix, candidate_style_info_mix, 1)
                predict_count += 1
                sorted_by_similarity = sorted(playstyle_distances_mix_t1.items(), key=lambda d: d[1]) 
                if test_style == sorted_by_similarity[0][0]:
                    space_mix_t1_correct_count += 1

                    sorted_by_similarity = sorted(playstyle_distances_single.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        single_correct_count += 1
                    else:
                        mixt1_vs_single[0] += 1

                    sorted_by_similarity = sorted(playstyle_distances_2_20_t2.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        space_2_20_t2_correct_count += 1
                    else:
                        mixt1_vs_2_20_t2[0] += 1

                    sorted_by_similarity = sorted(playstyle_distances_2_20_t1.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        space_2_20_t1_correct_count += 1
                    else:
                        mixt1_vs_2_20_t1[0] += 1

                    sorted_by_similarity = sorted(playstyle_distances_res_t2.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        space_res_t2_correct_count += 1
                    else:
                        mixt1_vs_res_t2[0] += 1

                    sorted_by_similarity = sorted(playstyle_distances_res_t1.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        space_res_t1_correct_count += 1
                    else:
                        mixt1_vs_res_t1[0] += 1

                    sorted_by_similarity = sorted(playstyle_distances_mix_t2.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        space_mix_t2_correct_count += 1
                    else:
                        mixt1_vs_mixt2[0] += 1

                else:
                    sorted_by_similarity = sorted(playstyle_distances_single.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        single_correct_count += 1
                        mixt1_vs_single[1] += 1

                    sorted_by_similarity = sorted(playstyle_distances_2_20_t2.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        space_2_20_t2_correct_count += 1
                        mixt1_vs_2_20_t2[1] += 1

                    sorted_by_similarity = sorted(playstyle_distances_2_20_t1.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        space_2_20_t1_correct_count += 1
                        mixt1_vs_2_20_t1[1] += 1

                    sorted_by_similarity = sorted(playstyle_distances_res_t2.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        space_res_t2_correct_count += 1
                        mixt1_vs_res_t2[1] += 1

                    sorted_by_similarity = sorted(playstyle_distances_res_t1.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        space_res_t1_correct_count += 1
                        mixt1_vs_res_t1[1] += 1

                    sorted_by_similarity = sorted(playstyle_distances_mix_t2.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        space_mix_t2_correct_count += 1
                        mixt1_vs_mixt2[1] += 1

        print()
        print("under {} sample size".format(sample_count))
        print(f"Single accuracy: {single_correct_count/predict_count}, p-value: {chi2.sf((mixt1_vs_single[0]-mixt1_vs_single[1]) ** 2 / (mixt1_vs_single[0]+mixt1_vs_single[1] + 0.000001),1)}")
        print(f"2^20 t2 accuracy: {space_2_20_t2_correct_count/predict_count}, p-value: {chi2.sf((mixt1_vs_2_20_t2[0]-mixt1_vs_2_20_t2[1]) ** 2 / (mixt1_vs_2_20_t2[0]+mixt1_vs_2_20_t2[1] + 0.000001),1)}")
        print(f"2^20 t1 accuracy: {space_2_20_t1_correct_count/predict_count}, p-value: {chi2.sf((mixt1_vs_2_20_t1[0]-mixt1_vs_2_20_t1[1]) ** 2 / (mixt1_vs_2_20_t1[0]+mixt1_vs_2_20_t1[1] + 0.000001),1)}")
        print(f"res t2 accuracy: {space_res_t2_correct_count/predict_count}, p-value: {chi2.sf((mixt1_vs_res_t2[0]-mixt1_vs_res_t2[1]) ** 2 / (mixt1_vs_res_t2[0]+mixt1_vs_res_t2[1] + 0.000001),1)}")
        print(f"res t1 accuracy: {space_res_t1_correct_count/predict_count}, p-value: {chi2.sf((mixt1_vs_res_t1[0]-mixt1_vs_res_t1[1]) ** 2 / (mixt1_vs_res_t1[0]+mixt1_vs_res_t1[1] + 0.000001),1)}")
        print(f"Mix t2 accuracy: {space_mix_t2_correct_count/predict_count}, p-value: {chi2.sf((mixt1_vs_mixt2[0]-mixt1_vs_mixt2[1]) ** 2 / (mixt1_vs_mixt2[0]+mixt1_vs_mixt2[1] + 0.000001),1)}")
        print(f"Mix t1 accuracy: {space_mix_t1_correct_count/predict_count}")

def calculate_w2(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    w2 = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return w2

def sample_a_list_without_replacement(full_list, size):
    return random.sample(full_list, k=size)

def extract_style_info(style_list, available_state_index):
    style_info = {
        "state_set" : set(),
        "state_count" : {},
        "actions" : {},
    }

    for pair in style_list:
        state, action = pair
        for i_s in range(len(state)):
            if i_s in available_state_index:
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

def get_compound_style(speed_styles, noise_styles):
    compound_styles = []
    for speed_style in speed_styles:
        for noise_style in noise_styles:
            compound_styles.append("Speed{}N{}".format(speed_style, noise_style))
    return compound_styles
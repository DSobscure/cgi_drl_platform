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
    code_level = [0]

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

def calculate_bhattacharyya_coefficient (act1, act2, reg_param=1e-8):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    sigma = (sigma1 + sigma2) / 2
    det_sigma1 = np.linalg.det(sigma1) + reg_param
    det_sigma2 = np.linalg.det(sigma2) + reg_param
    det_sigma = np.linalg.det(sigma) + reg_param
    
    if np.linalg.det(sigma) != 0:
        sigma_inverted = np.linalg.inv(sigma)
    else:
        sigma_inverted = np.linalg.pinv(sigma)
    distance = 0.125 * np.dot(np.dot(np.transpose(mu1 - mu2), sigma_inverted),(mu1 - mu2)) + 0.5 * np.log(det_sigma / (np.sqrt(det_sigma1 * det_sigma2) + reg_param))
    distance = max(distance, 0)
    return np.exp(-distance)

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
    state_space = len(style_A["state_set"].union(style_B["state_set"]))
    valid_state_count = 0

    distances = []
    for s in intersection_state:
        distance = calculate_w2(np.asarray(style_A["actions"][s] + style_A["actions"][s]), np.asarray(style_B["actions"][s] + style_B["actions"][s]))
        distances.append(distance)
        valid_state_count += 1
    return distances, valid_state_count

def compute_similarity_int_bc(style_A, style_B):
    intersection_state = style_A["state_set"].intersection(style_B["state_set"])
    state_space = len(style_A["state_set"].union(style_B["state_set"]))
    valid_state_count = 0

    similar_probability = np.double(0.0)
    for s in intersection_state:
        similar_probability += calculate_bhattacharyya_coefficient(np.asarray(style_A["actions"][s] + style_A["actions"][s]), np.asarray(style_B["actions"][s] + style_B["actions"][s]))
        valid_state_count += 1
    if valid_state_count != 0:
        similar_probability /= valid_state_count
    return similar_probability

def get_compound_style(speed_styles, noise_styles):
    compound_styles = []
    for speed_style in speed_styles:
        for noise_style in noise_styles:
            compound_styles.append("Speed{}N{}".format(speed_style, noise_style))
    return compound_styles
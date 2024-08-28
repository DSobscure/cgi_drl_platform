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
    code_level = [-1]
    threshold_count = 1

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
    result_path += "code-2"
    # with open(result_path + "_playstyle_distance.csv", "w", newline="") as distance_csvfile, open(result_path + "_playstyle_distance_iou.csv", "w", newline="") as iou_csvfile:
    #     distance_writer = csv.writer(distance_csvfile)
    #     iou_writer = csv.writer(iou_csvfile)

    for i_sample_power in range(max_sample_count_power + 1):
        sample_count = 1024 # 2 ** i_sample_power

        model_accurate_list = []
        jaccard_index_list = []

        for i in range(repeat_count):
            model_accurate = 0
            print("Repated: {}/{}".format(i, repeat_count), end='\r')

            for test_style in all_styles:
                playstyle_distances = {}
                jaccard_indexes = []
                double_test_list = sample_a_list_without_replacement(playstyle_dataset[test_style], sample_count * 2)
                test_style_info = extract_style_info(double_test_list[sample_count:])
                all_style_infos = {}
                for candidate_style in all_styles:
                    if test_style == candidate_style:
                        all_style_infos[candidate_style] = extract_style_info(double_test_list[:sample_count])
                    else:
                        all_style_infos[candidate_style] = extract_style_info(sample_a_list_without_replacement(playstyle_dataset[candidate_style], sample_count))     
                    playstyle_distance, jaccard_index = compute_similarity(test_style_info, all_style_infos[candidate_style], threshold_count)
                    playstyle_distances[candidate_style] = playstyle_distance
                    jaccard_indexes.append(jaccard_index)
                sorted_by_similarity = sorted(playstyle_distances.items(), key=lambda d: d[1]) 
                if test_style == sorted_by_similarity[0][0]:
                    model_accurate += 1
                # print(test_style, "in", [s[0]for s in sorted_by_similarity])
            model_accurate_list.append(model_accurate / len(all_styles))
            jaccard_index_list.append(np.mean(jaccard_indexes))

        print()
        print("under {} sample size, accuracy: {:.2f}%, std:{:.2f}%, jaccard index: {:.2f}%".format(sample_count, np.mean(model_accurate_list) * 100, np.std(model_accurate_list) * 100,  np.mean(jaccard_index_list) * 100))
            # distance_writer.writerow(model_accurate_list)
            # iou_writer.writerow(jaccard_index_list)

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
    jaccard_index = valid_state_count / len(style_A["state_set"].union(style_B["state_set"]))
    return playstyle_distance, jaccard_index

def get_compound_style(speed_styles, noise_styles):
    compound_styles = []
    for speed_style in speed_styles:
        for noise_style in noise_styles:
            compound_styles.append("Speed{}N{}".format(speed_style, noise_style))
    return compound_styles
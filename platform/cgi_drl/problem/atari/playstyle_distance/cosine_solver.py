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

    playstyle_dataset = {}
    for demo_pair in demo_config["demo_pairs"]:
        demo_config["npz_folder"] = demo_pair[0]
        memory = AtariDemonstrationMemory(demo_config)
        playstyle_dataset[demo_pair[1]] = []

        for batch in memory.sample_all_batch(128):
            observation_batch, action_batch = batch[0], batch[1]
            states = encoder.get_continuous_latent_codes({"visual" : observation_batch["observation_2d"]}, code_level[0])
            for s in states:
                playstyle_dataset[demo_pair[1]].append(s)

    algorithm_styles = ["DQN", "C51", "Rainbow", "IQN"]
    # algorithm_styles = ["DQN"]
    all_styles = get_compound_style(algorithm_styles)

    result_path = problem_config["result_path"]
    # result_path += "code0"
    with open(result_path + "_cosine.csv", "w", newline="") as distance_csvfile:
        distance_writer = csv.writer(distance_csvfile)

        for i_sample_power in range(max_sample_count_power + 1):
            sample_count = 2 ** i_sample_power

            model_accurate_list = []

            for i in range(repeat_count):
                model_accurate = 0
                sub_counter = 0
                for test_style in all_styles:
                    print("Repated: {}-{}/{}".format(i, sub_counter, repeat_count), end='\r')
                    sub_counter += 1
                    playstyle_distances = {}
                    double_test_list = sample_a_list_without_replacement(playstyle_dataset[test_style], sample_count * 2)
                    test_style_info = double_test_list[sample_count:]
                    all_style_infos = {}
                    for candidate_style in all_styles:
                        if test_style == candidate_style:
                            all_style_infos[candidate_style] = double_test_list[:sample_count]
                        else:
                            all_style_infos[candidate_style] = sample_a_list_without_replacement(playstyle_dataset[candidate_style], sample_count)
                        playstyle_distance = calculate_cosine_similarity(test_style_info, all_style_infos[candidate_style])
                        playstyle_distances[candidate_style] = -playstyle_distance
                    sorted_by_similarity = sorted(playstyle_distances.items(), key=lambda d: d[1]) 
                    if test_style == sorted_by_similarity[0][0]:
                        model_accurate += 1
                model_accurate_list.append(model_accurate / len(all_styles))

            print()
            print("under {} sample size, accuracy: {:.2f}%, std:{:.2f}%".format(sample_count, np.mean(model_accurate_list) * 100, np.std(model_accurate_list) * 100))
            distance_writer.writerow(model_accurate_list)

def calculate_cosine_similarity(sample1, sample2):
    sample1 = np.asarray(sample1, dtype=np.float32)
    sample2 = np.asarray(sample2, dtype=np.float32)

    mu1 = sample1.mean(axis=0)
    mu2 = sample2.mean(axis=0)

    return np.dot(mu1, mu2)/(np.linalg.norm(mu1)*np.linalg.norm(mu2))

def sample_a_list_without_replacement(full_list, size):
    if len(full_list) < size:
        return random.sample(full_list, k=size//2) + random.sample(full_list, k=size//2)
    else:
        return random.sample(full_list, k=size)

def get_compound_style(style_names):
    indice = [1, 2, 3, 4, 5]
    compound_styles = []
    for style in style_names:
        for index in indice:
            compound_styles.append("{}_Model{}".format(style, index))
    return compound_styles
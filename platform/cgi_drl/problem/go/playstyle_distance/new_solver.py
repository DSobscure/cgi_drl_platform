import importlib
import numpy as np
import random
import math
import time

def launch(problem_config):
    load = problem_config["load_function"]

    # setup hsd
    hsd_config = load(*problem_config["hsd"])
    problem_config["hsd"] = hsd_config
    from cgi_drl.representation_model.hsd.encoder_trainer_torch_go import EncoderTrainer
    encoder = EncoderTrainer(hsd_config)
    
    encoder.load("versions/go_hsd_default/model/model_100.ckpt")

    def encoder_function(observations):
        return encoder.get_discrite_states(observations)

    # setup demo
    query_demo_config = load(*problem_config["query_demo"])
    problem_config["query_demo"] = query_demo_config
    candidate_demo_config = load(*problem_config["candidate_demo"])
    problem_config["candidate_demo"] = candidate_demo_config
    from cgi_drl.data_storage.demonstration_memory.go_demonstration_memory import GoDemonstrationMemory
    query_records = GoDemonstrationMemory(query_demo_config, encoder_function)
    candidate_records = GoDemonstrationMemory(candidate_demo_config, encoder_function)

    unit_game_count = [1, 5, 10, 25, 50, 75, 100]
    # unit_game_count = [100]
    query_styles = query_records.get_player_names()
    candidate_styles = candidate_records.get_player_names()

    for unit_of_query in unit_game_count:
        for unit_of_candidate in unit_game_count:
            accurate_count = 0
            top3_accurate_count = 0
            query_count = 0
            intersection_state_counts = []
            for query_style in query_styles:
                playstyle_similar_probabilities = {}
                jaccard_indexes = []

                records = query_records.get_records(query_style, unit_of_query)
                query_style_info = extract_style_info(records)

                all_style_distances = {}
                all_state_spaces = {}
                all_distances = []

                for candidate_style in candidate_styles:
                    records = candidate_records.get_records(candidate_style, unit_of_candidate)
                    candidate_style_info = extract_style_info(records)
                    distances, jaccard_index, state_space = compute_similarity(query_style_info, candidate_style_info)
                    intersection_state_counts.append(state_space)
                    all_style_distances[candidate_style] = distances
                    all_state_spaces[candidate_style] = state_space
                    all_distances = all_distances + distances
                    jaccard_indexes.append(jaccard_index)
                if len(all_distances) > 0:
                    distance_mean = np.mean(all_distances)
                    distance_std = np.std(all_distances)
                for candidate_style in candidate_styles:
                    similar_probability = np.double(0.0)
                    state_space = all_state_spaces[candidate_style]
                    for d in all_style_distances[candidate_style]:
                        d = d / (distance_mean + 1e-8)
                        similar_probability += np.exp(-d) / state_space
                    playstyle_similar_probabilities[candidate_style] = similar_probability
                sorted_by_similarity = sorted(playstyle_similar_probabilities.items(), key=lambda d: d[1], reverse=True) 
                if query_style == sorted_by_similarity[0][0]:
                    accurate_count += 1
                    top3_accurate_count += 1
                elif query_style == sorted_by_similarity[1][0]:
                    top3_accurate_count += 1
                elif query_style == sorted_by_similarity[2][0]:
                    top3_accurate_count += 1
                query_count += 1
                print("Progess: {}/200".format(query_count), end='\r')
            print("Unit of Query: {}, Unit of Candidate: {}, Accuracy: {}, Top3 Accuracy: {}, Average Intersection: {}".format(unit_of_query, unit_of_candidate, accurate_count / 200, top3_accurate_count / 200, np.mean(intersection_state_counts)))
            
def calculate_w2(action_list1, action_list2):
    mu1 = np.bincount(action_list1, minlength=362)
    mu2 = np.bincount(action_list2, minlength=362)

    mu1 = mu1 / len(action_list1)
    mu2 = mu2 / len(action_list2)

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
    threshold_count = 1

    distances = []
    for s in intersection_state:
        distance = calculate_w2(np.asarray(style_A["actions"][s] + style_A["actions"][s]), np.asarray(style_B["actions"][s] + style_B["actions"][s]))
        distances.append(distance)
        valid_state_count += 1
    jaccard_index = valid_state_count / state_space
    return distances, jaccard_index, state_space
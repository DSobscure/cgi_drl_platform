import numpy as np
import math
import csv

def launch(problem_config):
    load = problem_config["load_function"]

    # setup demo
    demo_config = load(*problem_config["demo"])
    problem_config["demo"] = demo_config
    from cgi_drl.data_storage.demonstration_memory.game2048_demonstration_memory import Game2048DemonstrationMemory
    buffer = Game2048DemonstrationMemory(demo_config)

    code_level = [-1]
    threshold_count = 1
    model_count = 10
    model_episdoe_count = 1000
    query_episdoe_count = 100
    reference_episdoe_count = 500

    reference_style_infos = {}
    query_style_infos = {}

    for i_model in range(model_count):
        reference_style_infos[i_model] = {
            "state_set" : set(),
            "state_count" : {},
            "actions" : {},
        }
        for i_episode in range(model_episdoe_count):
            if i_episode >= reference_episdoe_count:
                query_style_infos[i_model * model_episdoe_count + i_episode // query_episdoe_count] = {
                    "state_set" : set(),
                    "state_count" : {},
                    "actions" : {},
                }
            episode = buffer.collect_episodes(i_model, i_episode)
            for move in episode:
                state = move["state"]
                action = move["action"]
                if action < 0:
                    continue
                for level in code_level:
                    code = buffer.encode_state(state, level)
                    if i_episode < reference_episdoe_count:
                        style_name = i_model
                        reference_style_infos[style_name]["state_set"].add(code)
                        if code not in reference_style_infos[style_name]["state_count"]:
                            reference_style_infos[style_name]["state_count"][code] = 0
                        reference_style_infos[style_name]["state_count"][code] += 1
                        if code not in reference_style_infos[style_name]["actions"]:
                            reference_style_infos[style_name]["actions"][code] = np.zeros(4)
                        reference_style_infos[style_name]["actions"][code][action] += 1
                    else:
                        style_name = i_model * model_episdoe_count + i_episode // query_episdoe_count
                        query_style_infos[style_name]["state_set"].add(code)
                        if code not in query_style_infos[style_name]["state_count"]:
                            query_style_infos[style_name]["state_count"][code] = 0
                        query_style_infos[style_name]["state_count"][code] += 1
                        if code not in query_style_infos[style_name]["actions"]:
                            query_style_infos[style_name]["actions"][code] = np.zeros(4)
                        query_style_infos[style_name]["actions"][code][action] += 1

            print("Process model:{}, episdoe: {}".format(i_model, i_episode), end='\r')
    print()

            
    accurate_count = 0
    int_accurate_count = 0
    iou_accurate_count = 0
    ps_accurate_count = 0
    int_bc_accurate_count = 0
    ps_bc_accurate_count = 0
    test_count = 0
    jaccard_index_sum = 0
    jaccard_index_test_count = 0

    total_test_count = (model_episdoe_count - reference_episdoe_count) * model_count // query_episdoe_count

    for query_style in query_style_infos: 
        playstyle_metrics = {}
        playstyle_jaccrad_indeces = {}
        playstyle_intersection_similarities = {}
        playstyle_similarities = {}
        playstyle_bc_intersection_similarities = {}
        playstyle_bc_similarities = {}
        jaccard_indexes = []
        query_style_info = query_style_infos[query_style]

        all_style_distances = {}
        all_state_spaces = {}
        all_distances = []
        for reference_style in reference_style_infos:
            reference_style_info = reference_style_infos[reference_style]
            playstyle_distance, int_bc, jaccard_index, playstyle_distances, state_space = compute_playstyle(query_style_info, reference_style_info, threshold_count)
            playstyle_metrics[reference_style] = playstyle_distance
            all_style_distances[reference_style] = playstyle_distances
            all_state_spaces[reference_style] = state_space
            all_distances = all_distances + playstyle_distances
            playstyle_jaccrad_indeces[reference_style] = jaccard_index
            playstyle_bc_intersection_similarities[reference_style] = int_bc
            playstyle_bc_similarities[reference_style] = int_bc * jaccard_index
            jaccard_index_sum += jaccard_index
            jaccard_index_test_count += 1
        distance_mean = np.mean(all_distances)
        distance_std = np.std(all_distances)
        for reference_style in reference_style_infos:
            similar_probability = np.double(0.0)
            state_space = all_state_spaces[reference_style]
            for d in all_style_distances[reference_style]:
                d = d / (distance_mean + 1e-8)
                similar_probability += np.exp(-d) / state_space
            playstyle_intersection_similarities[reference_style] = similar_probability
            playstyle_similarities[reference_style] = similar_probability * playstyle_jaccrad_indeces[reference_style]
        sorted_by_similarity = sorted(playstyle_metrics.items(), key=lambda d: d[1]) 
        sorted_by_iou_similarity = sorted(playstyle_intersection_similarities.items(), key=lambda d: d[1], reverse=True) 
        sorted_by_int_similarity = sorted(playstyle_jaccrad_indeces.items(), key=lambda d: d[1], reverse=True) 
        sorted_by_ps_similarity = sorted(playstyle_similarities.items(), key=lambda d: d[1], reverse=True) 
        sorted_by_int_bc_similarity = sorted(playstyle_bc_intersection_similarities.items(), key=lambda d: d[1], reverse=True) 
        sorted_by_ps_bc_similarity = sorted(playstyle_bc_similarities.items(), key=lambda d: d[1], reverse=True) 
        if (query_style // model_episdoe_count) == sorted_by_similarity[0][0] :
            accurate_count += 1
        if (query_style // model_episdoe_count) == sorted_by_iou_similarity[0][0] :
            int_accurate_count += 1
        if (query_style // model_episdoe_count) == sorted_by_int_similarity[0][0] :
            iou_accurate_count += 1
        if (query_style // model_episdoe_count) == sorted_by_ps_similarity[0][0] :
            ps_accurate_count += 1
        if (query_style // model_episdoe_count) == sorted_by_int_bc_similarity[0][0] :
            int_bc_accurate_count += 1
        if (query_style // model_episdoe_count) == sorted_by_ps_bc_similarity[0][0] :
            ps_bc_accurate_count += 1
        test_count += 1
        print(test_count, "/", total_test_count, end="\r")

    print("Playstyle Distance - accuracy: {:.2f}%, jaccard index: {:.2f}%".format(accurate_count / test_count * 100,  jaccard_index_sum / jaccard_index_test_count * 100))
    print("Playstyle Intersection Similarity - accuracy: {:.2f}%, jaccard index: {:.2f}%".format(int_accurate_count / test_count * 100,  jaccard_index_sum / jaccard_index_test_count * 100))
    print("Playstyle Jaccard Index - accuracy: {:.2f}%, jaccard index: {:.2f}%".format(iou_accurate_count / test_count * 100,  jaccard_index_sum / jaccard_index_test_count * 100))
    print("Playstyle Similarity - accuracy: {:.2f}%, jaccard index: {:.2f}%".format(ps_accurate_count / test_count * 100,  jaccard_index_sum / jaccard_index_test_count * 100))
    print("Playstyle Intersection BC Similarity - accuracy: {:.2f}%, jaccard index: {:.2f}%".format(int_bc_accurate_count / test_count * 100,  jaccard_index_sum / jaccard_index_test_count * 100))
    print("Playstyle BC Similarity - accuracy: {:.2f}%, jaccard index: {:.2f}%".format(ps_bc_accurate_count / test_count * 100,  jaccard_index_sum / jaccard_index_test_count * 100))
    

def calculate_w2(act1, act2):
    mu1 = act1 / act1.sum()
    mu2 = act2 / act2.sum()
    return np.linalg.norm(mu1 - mu2, 2)

def calculate_bhattacharyya_coefficient(act1, act2):
    mu1 = act1 / act1.sum()
    mu2 = act2 / act2.sum()
    return np.sum(np.sqrt(mu1 * mu2), axis=0)

def compute_playstyle_distance(style_A, style_B, threshold_count):
    intersection_state = style_A["state_set"].intersection(style_B["state_set"])
    state_space = len(style_A["state_set"].union(style_B["state_set"]))
    overlapping_count_in_A, overlapping_count_in_B = 0, 0
    valid_state_count = 0

    for s in intersection_state:
        if style_A["state_count"][s] < threshold_count or style_B["state_count"][s] < threshold_count:
            continue
        overlapping_count_in_A += style_A["state_count"][s]
        overlapping_count_in_B += style_B["state_count"][s]

    playstyle_distance = np.double(0.0)
    similar_probability = np.double(0.0)
    for s in intersection_state:
        if style_A["state_count"][s] < threshold_count or style_B["state_count"][s] < threshold_count:
            continue
        distance = calculate_w2(style_A["actions"][s], style_B["actions"][s])
        playstyle_distance += (distance * style_A["state_count"][s] / overlapping_count_in_A + distance * style_B["state_count"][s] / overlapping_count_in_B) / 2
        similar_probability += calculate_bhattacharyya_coefficient(style_A["actions"][s], style_B["actions"][s])
        valid_state_count += 1
    
    if valid_state_count == 0:
        playstyle_distance = math.inf
    jaccard_index = valid_state_count / state_space
    return playstyle_distance, similar_probability / valid_state_count, jaccard_index

def compute_playstyle_intersection_similarity(style_A, style_B, threshold_count):
    intersection_state = style_A["state_set"].intersection(style_B["state_set"])
    state_space = len(style_A["state_set"].union(style_B["state_set"]))
    valid_state_count = 0

    distances = []
    for s in intersection_state:
        if style_A["state_count"][s] < threshold_count or style_B["state_count"][s] < threshold_count:
            continue
        distance = calculate_w2(style_A["actions"][s], style_B["actions"][s])
        distances.append(distance)
        valid_state_count += 1
    jaccard_index = valid_state_count / state_space
    return distances, jaccard_index, valid_state_count

def compute_playstyle(style_A, style_B, threshold_count):
    intersection_state = style_A["state_set"].intersection(style_B["state_set"])
    state_space = len(style_A["state_set"].union(style_B["state_set"]))
    overlapping_count_in_A, overlapping_count_in_B = 0, 0
    valid_state_count = 0

    for s in intersection_state:
        if style_A["state_count"][s] < threshold_count or style_B["state_count"][s] < threshold_count:
            continue
        overlapping_count_in_A += style_A["state_count"][s]
        overlapping_count_in_B += style_B["state_count"][s]

    playstyle_distance = np.double(0.0)
    similar_probability = np.double(0.0)
    distances = []
    for s in intersection_state:
        if style_A["state_count"][s] < threshold_count or style_B["state_count"][s] < threshold_count:
            continue
        distance = calculate_w2(style_A["actions"][s], style_B["actions"][s])
        distances.append(distance)
        playstyle_distance += (distance * style_A["state_count"][s] / overlapping_count_in_A + distance * style_B["state_count"][s] / overlapping_count_in_B) / 2
        similar_probability += calculate_bhattacharyya_coefficient(style_A["actions"][s], style_B["actions"][s])
        valid_state_count += 1
    
    if valid_state_count == 0:
        playstyle_distance = math.inf
        int_bc = 0
    else:
        int_bc = similar_probability / valid_state_count
    jaccard_index = valid_state_count / state_space
    return playstyle_distance, int_bc, jaccard_index, distances, valid_state_count
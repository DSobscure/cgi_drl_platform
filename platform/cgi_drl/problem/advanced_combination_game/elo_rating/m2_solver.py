from cgi_drl.problem.workflow_template.elo_rating_template import EloRatingSolver

def launch(problem_config):
    load = problem_config["load_function"]
    # setup model
    elo_rating_config = load(*problem_config["elo_rating"])
    problem_config["elo_rating"] = elo_rating_config
    from cgi_drl.measure_model.elo_rating.m2_trainer import EloRatingTrainer
    model = EloRatingTrainer(elo_rating_config)

    # setup record
    from cgi_drl.data_storage.match_records.advanced_combination_game_tabular_record import AdvancedCombinationGameMatchResultBuffer

    record_train_config = load(*problem_config["record_train"])
    problem_config["record_train"] = record_train_config
    result_buffer_train = AdvancedCombinationGameMatchResultBuffer(record_train_config)

    record_test_config = load(*problem_config["record_test"])
    problem_config["record_test"] = record_test_config
    result_buffer_test = AdvancedCombinationGameMatchResultBuffer(record_test_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = EloRatingSolver(problem_config)
    solver.model = model
    solver.result_buffer_train = result_buffer_train
    solver.result_buffer_test = result_buffer_test

    solver.train()

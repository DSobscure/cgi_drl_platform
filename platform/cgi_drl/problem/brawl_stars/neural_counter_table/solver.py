from cgi_drl.problem.workflow_template.neural_counter_table_template import NeuralCounterTableSolver

def launch(problem_config):
    load = problem_config["load_function"]
    # setup model
    nrt_config = load(*problem_config["nrt"])
    problem_config["nrt"] = nrt_config
    from cgi_drl.measure_model.neural_rating_table.trainer import NeuralRatingTableTrainer
    bt_model = NeuralRatingTableTrainer(nrt_config)

    nrt_config = load(*problem_config["nct"])
    problem_config["nct"] = nrt_config
    from cgi_drl.measure_model.neural_counter_table.trainer import NeuralCounterTableTrainer
    model = NeuralCounterTableTrainer(nrt_config)

    # setup record
    from cgi_drl.data_storage.match_records.brawl_stars_match_result_buffer import BrawlStarsMatchResultBuffer

    record_train_config = load(*problem_config["record_train"])
    problem_config["record_train"] = record_train_config
    result_buffer_train = BrawlStarsMatchResultBuffer(record_train_config)

    record_test_config = load(*problem_config["record_test"])
    problem_config["record_test"] = record_test_config
    result_buffer_test = BrawlStarsMatchResultBuffer(record_test_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = NeuralCounterTableSolver(problem_config)
    solver.bt_model = bt_model
    solver.model = model
    solver.result_buffer_train = result_buffer_train
    solver.result_buffer_test = result_buffer_test

    solver.train()

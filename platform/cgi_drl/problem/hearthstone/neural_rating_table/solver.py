from cgi_drl.problem.workflow_template.neural_rating_table_template import NeuralRatingTableSolver

def launch(problem_config):
    load = problem_config["load_function"]
    # setup model
    nrt_config = load(*problem_config["nrt"])
    problem_config["nrt"] = nrt_config
    from cgi_drl.measure_model.neural_rating_table.trainer import NeuralRatingTableTrainer
    model = NeuralRatingTableTrainer(nrt_config)

    # setup record
    from cgi_drl.data_storage.match_records.hearthstone_match_result_buffer import HearthstoneMatchResultBuffer

    record_train_config = load(*problem_config["record_train"])
    problem_config["record_train"] = record_train_config
    result_buffer_train = HearthstoneMatchResultBuffer(record_train_config)

    record_test_config = load(*problem_config["record_test"])
    problem_config["record_test"] = record_test_config
    result_buffer_test = HearthstoneMatchResultBuffer(record_test_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = NeuralRatingTableSolver(problem_config)
    solver.model = model
    solver.result_buffer_train = result_buffer_train
    solver.result_buffer_test = result_buffer_test

    solver.train()

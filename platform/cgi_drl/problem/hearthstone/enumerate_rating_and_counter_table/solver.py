from cgi_drl.problem.workflow_template.enumerate_rating_and_counter_table_template import EnumerateRatingAndCounterTableSolver

def launch(problem_config):
    load = problem_config["load_function"]
    # setup model
    nrt_config = load(*problem_config["nrt"])
    problem_config["nrt"] = nrt_config
    from cgi_drl.measure_model.neural_rating_table.trainer import NeuralRatingTableTrainer
    rating_table_model = NeuralRatingTableTrainer(nrt_config)

    nrt_config = load(*problem_config["nct"])
    problem_config["nct"] = nrt_config
    from cgi_drl.measure_model.neural_counter_table.trainer import NeuralCounterTableTrainer
    counter_table_model = NeuralCounterTableTrainer(nrt_config)

    # setup record
    from cgi_drl.data_storage.match_records.hearthstone_match_result_buffer import HearthstoneMatchResultBuffer

    record_config = load(*problem_config["record"])
    problem_config["record"] = record_config
    result_buffer = HearthstoneMatchResultBuffer(record_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = EnumerateRatingAndCounterTableSolver(problem_config)
    solver.rating_table_model = rating_table_model
    solver.counter_table_model = counter_table_model
    solver.result_buffer = result_buffer

    solver.train()


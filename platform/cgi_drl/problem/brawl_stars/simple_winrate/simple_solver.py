from cgi_drl.problem.workflow_template.simple_winrate_template import SimpleWinrateSolver

def launch(problem_config):
    load = problem_config["load_function"]
    # setup model
    winr_config = load(*problem_config["winr"])
    problem_config["winr"] = winr_config
    from cgi_drl.measure_model.simple_winrate.trainer import SimpleWinrateTrainer
    model = SimpleWinrateTrainer(winr_config)

    # setup record
    from cgi_drl.data_storage.match_records.brawl_stars_simple_match_result_buffer import BrawlStarsMatchResultBuffer

    record_train_config = load(*problem_config["record_train"])
    problem_config["record_train"] = record_train_config
    result_buffer_train = BrawlStarsMatchResultBuffer(record_train_config)

    record_test_config = load(*problem_config["record_test"])
    problem_config["record_test"] = record_test_config
    result_buffer_test = BrawlStarsMatchResultBuffer(record_test_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = SimpleWinrateSolver(problem_config)
    solver.model = model
    solver.result_buffer_train = result_buffer_train
    solver.result_buffer_test = result_buffer_test

    solver.train()

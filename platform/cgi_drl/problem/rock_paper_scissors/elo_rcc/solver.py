from cgi_drl.problem.workflow_template.elo_rcc_template import EloRccSolver

def launch(problem_config):
    load = problem_config["load_function"]
    # setup model
    elo_rcc_config = load(*problem_config["elo_rcc"])
    problem_config["elo_rcc"] = elo_rcc_config
    from cgi_drl.measure_model.elo_rcc.trainer import EloRccTrainer
    model = EloRccTrainer(elo_rcc_config)

    # setup record
    from cgi_drl.data_storage.match_records.rock_paper_scissors_tabular_record import RockPaperScissorsMatchResultBuffer

    record_train_config = load(*problem_config["record_train"])
    problem_config["record_train"] = record_train_config
    result_buffer_train = RockPaperScissorsMatchResultBuffer(record_train_config)

    record_test_config = load(*problem_config["record_test"])
    problem_config["record_test"] = record_test_config
    result_buffer_test = RockPaperScissorsMatchResultBuffer(record_test_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = EloRccSolver(problem_config)
    solver.model = model
    solver.result_buffer_train = result_buffer_train
    solver.result_buffer_test = result_buffer_test

    solver.train()

from cgi_drl.problem.workflow_template.pairwise_winrate_template import PairwiseWinrateSolver

def launch(problem_config):
    load = problem_config["load_function"]
    # setup model
    pairwise_winrate_config = load(*problem_config["pairwise_winrate"])
    problem_config["pairwise_winrate"] = pairwise_winrate_config
    from cgi_drl.measure_model.pairwise_winrate.tabular_trainer import PairwiseWinrateTrainer
    model = PairwiseWinrateTrainer(pairwise_winrate_config)

    # setup record
    from cgi_drl.data_storage.match_records.league_of_legends_tabular_match_result_buffer import LoLMatchResultBuffer

    record_train_config = load(*problem_config["record_train"])
    problem_config["record_train"] = record_train_config
    result_buffer_train = LoLMatchResultBuffer(record_train_config)

    record_test_config = load(*problem_config["record_test"])
    problem_config["record_test"] = record_test_config
    result_buffer_test = LoLMatchResultBuffer(record_test_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = PairwiseWinrateSolver(problem_config)
    solver.model = model
    solver.result_buffer_train = result_buffer_train
    solver.result_buffer_test = result_buffer_test

    solver.train()

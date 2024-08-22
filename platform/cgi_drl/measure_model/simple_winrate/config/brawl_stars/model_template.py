import tensorflow as tf

class DefaultTemplate(dict):
    def __init__(self, config):
        self["algorithm_scope"] = config.get("algorithm_scope", "WinR")

        def create_tensorflow_variables():
            self["learning_rate_placeholder"] = config.get("learning_rate_placeholder", tf.placeholder(dtype=tf.float32, name="learning_rate_placeholder"))
            self["optimizer"] = config.get("optimizer", tf.train.AdamOptimizer(self["learning_rate_placeholder"]))
        
            player1_combo_placeholder = tf.placeholder(shape = [None, 64 + 44 + 7], dtype = tf.float32, name = "player1_combo_placeholder")
            player2_combo_placeholder = tf.placeholder(shape = [None, 64 + 44 + 7], dtype = tf.float32, name = "player2_combo_placeholder")
            player_combo_input = tf.concat([player1_combo_placeholder, player2_combo_placeholder], axis=0) 

            self["observation_placeholders"] = config.get("observation_placeholders", {
                "player1_combo" : player1_combo_placeholder,
                "player2_combo" : player2_combo_placeholder,
            })   
            self["observation_inputs"] = config.get("observation_inputs", {
                "player_combo_input" : player_combo_input
            })          

            match_result_placeholder = tf.placeholder(shape = [None], dtype = tf.float32, name = "match_result_placeholder")
            match_result = tf.concat([match_result_placeholder, 1 - match_result_placeholder], axis=0) 
            match_result = tf.reshape(match_result, [-1, 1])

            self["match_result_placeholder"] = config.get("match_result_placeholder", match_result_placeholder)   
            self["match_result"] = config.get("match_result", match_result)    

        self["create_tensorflow_variables"] = create_tensorflow_variables

        self["model_define_path"] = config.get("model_define_path", "measure_model.simple_winrate.nn_model.shared.prediction_model")
        super().__init__(config)

class SimpleTemplate(dict):
    def __init__(self, config):
        self["algorithm_scope"] = config.get("algorithm_scope", "WinR")

        def create_tensorflow_variables():
            self["learning_rate_placeholder"] = config.get("learning_rate_placeholder", tf.placeholder(dtype=tf.float32, name="learning_rate_placeholder"))
            self["optimizer"] = config.get("optimizer", tf.train.AdamOptimizer(self["learning_rate_placeholder"]))
        
            player1_combo_placeholder = tf.placeholder(shape = [None, 64], dtype = tf.float32, name = "player1_combo_placeholder")
            player2_combo_placeholder = tf.placeholder(shape = [None, 64], dtype = tf.float32, name = "player2_combo_placeholder")
            player_combo_input = tf.concat([player1_combo_placeholder, player2_combo_placeholder], axis=0) 

            self["observation_placeholders"] = config.get("observation_placeholders", {
                "player1_combo" : player1_combo_placeholder,
                "player2_combo" : player2_combo_placeholder,
            })   
            self["observation_inputs"] = config.get("observation_inputs", {
                "player_combo_input" : player_combo_input
            })          

            match_result_placeholder = tf.placeholder(shape = [None], dtype = tf.float32, name = "match_result_placeholder")
            match_result = tf.concat([match_result_placeholder, 1 - match_result_placeholder], axis=0) 
            match_result = tf.reshape(match_result, [-1, 1])

            self["match_result_placeholder"] = config.get("match_result_placeholder", match_result_placeholder)   
            self["match_result"] = config.get("match_result", match_result)    

        self["create_tensorflow_variables"] = create_tensorflow_variables

        self["model_define_path"] = config.get("model_define_path", "measure_model.simple_winrate.nn_model.shared.prediction_model")
        super().__init__(config)
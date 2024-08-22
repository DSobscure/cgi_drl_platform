import tensorflow as tf

class DefaultTemplate(dict):
    def __init__(self, config):
        self["algorithm_scope"] = config.get("algorithm_scope", "PairWin")

        def create_tensorflow_variables():
            self["learning_rate_placeholder"] = config.get("learning_rate_placeholder", tf.placeholder(dtype=tf.float32, name="learning_rate_placeholder"))
            self["optimizer"] = config.get("optimizer", tf.train.AdamOptimizer(self["learning_rate_placeholder"]))
        
            player1_combo_placeholder = tf.placeholder(shape = [None, 64 + 44 + 7], dtype = tf.float32, name = "player1_combo_placeholder")
            player2_combo_placeholder = tf.placeholder(shape = [None, 64 + 44 + 7], dtype = tf.float32, name = "player2_combo_placeholder")
            player1_combo_input = player1_combo_placeholder
            player2_combo_input = player2_combo_placeholder

            self["observation_placeholders"] = config.get("observation_placeholders", {
                "player1_combo" : player1_combo_placeholder,
                "player2_combo" : player2_combo_placeholder,
            })   
            self["observation_inputs"] = config.get("observation_inputs", {
                "player1_combo_input" : player1_combo_input,
                "player2_combo_input" : player2_combo_input
            })          
        self["create_tensorflow_variables"] = create_tensorflow_variables
        self["vq_beta"] = config.get("vq_beta", 0.01)

        self["model_define_path"] = config.get("model_define_path", "measure_model.pairwise_winrate.nn_model.shared.prediction_model")
        super().__init__(config)

class SimpleTemplate(dict):
    def __init__(self, config):
        self["algorithm_scope"] = config.get("algorithm_scope", "PairWin")

        def create_tensorflow_variables():
            self["learning_rate_placeholder"] = config.get("learning_rate_placeholder", tf.placeholder(dtype=tf.float32, name="learning_rate_placeholder"))
            self["optimizer"] = config.get("optimizer", tf.train.AdamOptimizer(self["learning_rate_placeholder"]))
        
            player1_combo_placeholder = tf.placeholder(shape = [None, 64], dtype = tf.float32, name = "player1_combo_placeholder")
            player2_combo_placeholder = tf.placeholder(shape = [None, 64], dtype = tf.float32, name = "player2_combo_placeholder")
            player1_combo_input = player1_combo_placeholder
            player2_combo_input = player2_combo_placeholder

            self["observation_placeholders"] = config.get("observation_placeholders", {
                "player1_combo" : player1_combo_placeholder,
                "player2_combo" : player2_combo_placeholder,
            })   
            self["observation_inputs"] = config.get("observation_inputs", {
                "player1_combo_input" : player1_combo_input,
                "player2_combo_input" : player2_combo_input
            })          
        self["create_tensorflow_variables"] = create_tensorflow_variables
        self["vq_beta"] = config.get("vq_beta", 0.01)

        self["model_define_path"] = config.get("model_define_path", "measure_model.pairwise_winrate.nn_model.shared.prediction_model")
        super().__init__(config)
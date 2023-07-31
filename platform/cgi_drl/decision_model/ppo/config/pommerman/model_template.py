import tensorflow as tf

class DefaultTemplate(dict):
    def __init__(self, config):
        self["algorithm_scope"] = config.get("algorithm_scope", "PPO")
        self["value_head_count"] = config.get("value_head_count", 1)

        self["use_rnn"] = config.get("use_rnn", False)
        if self["use_rnn"]:
            self["rnn_sequence_length"] = config.get("rnn_sequence_length", 8)
            self["rnn_burn_in_length"] = config.get("rnn_burn_in_length", 2)
            self["memory_size"] = config.get("memory_size", 512)        
        branch_count = 1

        def create_tensorflow_variables():
            if self["use_rnn"]:
                self["rnn_sequence_length_placeholder"] = config.get("rnn_sequence_length_placeholder", tf.placeholder(shape=None, dtype=tf.int32, name="rnn_sequence_length_placeholder"))
            self["learning_rate_placeholder"] = config.get("learning_rate_placeholder", tf.placeholder(dtype=tf.float32, name="learning_rate_placeholder"))
            self["clip_epsilon_placeholder"] = config.get("clip_epsilon_placeholder", tf.placeholder(dtype=tf.float32, name="clip_epsilon_placeholder"))
            self["entropy_coefficient_placeholder"] = config.get("entropy_coefficient_placeholder", tf.placeholder(dtype=tf.float32, name="entropy_coefficient_placeholder")) 
            self["value_coefficient_placeholder"] = config.get("value_coefficient_placeholder", tf.placeholder(shape=[self["value_head_count"]], dtype=tf.float32, name="value_coefficient_placeholder")) 
            self["value_clip_range_placeholder"] = config.get("value_clip_range_placeholder", tf.placeholder(shape=[self["value_head_count"]], dtype=tf.float32, name="value_clip_range_placeholder")) 
            self["optimizer"] = config.get("optimizer", tf.keras.optimizers.Adam(self["learning_rate_placeholder"]))
        
            observation_visual_placeholder = tf.placeholder(shape = [None, 3, 8, 8, 12], dtype = tf.float32, name = "observation_visual_placeholder")
            observation_visual_input = tf.transpose(observation_visual_placeholder, [0,2,3,4,1])
            observation_visual_input = tf.reshape(observation_visual_input, [-1, 8, 8, 12 * 3])

            observation_vector_placeholder = tf.placeholder(shape = [None, 3, 4], dtype = tf.float32, name = "observation_vector_placeholder")
            observation_vector_input = tf.reshape(observation_vector_placeholder, [-1, 3 * 4])

            self["observation_placeholders"] = config.get("observation_placeholders", {
                "observation_visual" : observation_visual_placeholder,
                "observation_vector" : observation_vector_placeholder
            })   
            self["observation_inputs"] = config.get("observation_inputs", {
                "observation_visual_input" : observation_visual_input,
                "observation_vector_input" : observation_vector_input,
            })   

            if self["use_rnn"]:
                observation_previous_action_placeholder = tf.placeholder(shape =  [None, branch_count], dtype = tf.int32, name = "observation_previous_action_placeholder")
                observation_previous_reward_placeholder = tf.placeholder(shape =  [None], dtype = tf.float32, name = "observation_previous_reward_placeholder")
                observation_previous_reward_input = tf.reshape(observation_previous_reward_placeholder, [-1, 1])
                observation_memory_placeholder = tf.placeholder(shape =  [None, self["memory_size"]], dtype = tf.float32, name = "observation_memory_placeholder")

                self["observation_placeholders"]["observation_previous_action"] = observation_previous_action_placeholder
                self["observation_placeholders"]["observation_previous_reward"] = observation_previous_reward_placeholder
                self["observation_placeholders"]["observation_memory"] = observation_memory_placeholder

                self["observation_inputs"]["observation_previous_action_input"] = observation_previous_action_placeholder
                self["observation_inputs"]["observation_previous_reward_input"] = observation_previous_reward_input
                self["observation_inputs"]["observation_memory_input"] = observation_memory_placeholder
        
        self["create_tensorflow_variables"] = create_tensorflow_variables
        self["model_define_path"] = config.get("model_define_path", "decision_model.ppo.nn_model.pommerman.default_model")
        self["max_gradient_norm"] = config.get("max_gradient_norm", 10)

        def invertible_value_function(x, is_inverse):
            return x
        self["invertible_value_functions"] = config.get("invertible_value_functions", [invertible_value_function] * self["value_head_count"])

        super().__init__(config)
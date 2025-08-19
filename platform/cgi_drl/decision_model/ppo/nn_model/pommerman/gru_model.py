"""NNModels for pommerman"""
import numpy as np
import tensorflow as tf

class PolicyModel:
    def __init__(self, network_settings):
        self.rnn_sequence_length = network_settings["rnn_sequence_length"]
        self.rnn_burn_in_length = network_settings["rnn_burn_in_length"]
        self.memory_size = network_settings["memory_size"]
        self.action_space = network_settings["action_space"]
        self.value_head_count = network_settings["value_head_count"]

    def build_observation_encoder(self, observations, is_train_mode):
        x_visual = observations["observation_visual_input"]
        x_vector = observations["observation_vector_input"]
        x_memory = observations["observation_memory_input"]

        cnn_filters = [32, 64, 128, 256]
        cnn_filer_sizes = [(3, 3), (3, 3), (3, 3), (2, 2)]
        cnn_strides = [(1, 1), (1, 1), (1, 1), (1, 1)]

        for i in range(len(cnn_filters)):
            x_visual = tf.keras.layers.Conv2D(
                filters=cnn_filters[i], 
                kernel_size=cnn_filer_sizes[i], 
                strides=cnn_strides[i], 
                activation=tf.nn.relu)(x_visual)
        x_visual = tf.keras.layers.Flatten()(x_visual)
        x = tf.concat([x_visual, x_vector], axis=-1)

        return {
            "latent_code_x": x,
            "latent_code_memory": x_memory,
        }
        
    def build_recurrent_encoder(self, latent_codes, is_train_mode):
        x = latent_codes["latent_code_x"]
        memory = latent_codes["latent_code_memory"]
        state_size = x.get_shape().as_list()[-1]
        sequence_x = tf.reshape(x, shape=[-1, self.rnn_sequence_length, state_size])
        memory = tf.reshape(memory, shape=[-1, self.rnn_sequence_length, self.memory_size])
        memory = memory[:,0,:]
        memory = tf.reshape(memory, shape=[-1, self.memory_size])

        burn_in_sequence_x = tf.stop_gradient(sequence_x[:,:self.rnn_burn_in_length,:])
        non_burn_in_sequence_x = sequence_x[:,self.rnn_burn_in_length:,:]
        training_sequence = tf.concat([burn_in_sequence_x, non_burn_in_sequence_x], axis=1)

        sequence_x = tf.cond(is_train_mode, lambda: training_sequence, lambda: sequence_x)

        gru_cell = tf.keras.layers.GRU(self.memory_size, return_sequences=True, return_state=True)
        recurrent_output, final_state = gru_cell(sequence_x, initial_state=memory)
        recurrent_output = tf.reshape(recurrent_output, shape=[-1, self.memory_size])

        return {
            "latent_code_x": recurrent_output,
        }, final_state

    def build_policy(self, latent_codes, is_train_mode):
        x = latent_codes["latent_code_x"]

        x_value = x
        x_policy = x

        values = []
        for i in range(self.value_head_count):
            values.append(tf.keras.layers.Dense(1, activation=None)(x_value))
        branches_logit = [
            tf.keras.layers.Dense(branch, activation=None, use_bias=False)(x_policy)
            for branch in self.action_space
        ]
        models = [
            tf.distributions.Categorical(logits=branches_logit[i])
            for i in range(len(self.action_space))
        ]
        sample_action = tf.stack([model.sample(1) for model in models], axis=-1)
        sample_action = tf.reshape(sample_action, shape=[-1, len(self.action_space)])
        print("PPO Default sample action: {}".format(sample_action.get_shape()))
        max_action = tf.stack([tf.argmax(branches_logit[i], axis=-1) for i in range(len(self.action_space))], axis=-1)
        print("PPO Default max action: {}".format(max_action.get_shape()))

        _entropy = [model.entropy() for model in models]
        entropy = tf.reduce_sum(tf.stack(_entropy, axis=-1), axis=-1)

        policy_function = []
        for i in range(len(models)):
            policy_function.append(lambda act, k=i: models[k].prob(act) + 1e-8)

        return {
            "sample_action": sample_action,
            "max_action": max_action,
            "value": values,
            "entropy": entropy,
            "policy_function": policy_function,
        }

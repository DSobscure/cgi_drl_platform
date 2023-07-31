"""NNModels for pommerman"""
import numpy as np
import tensorflow as tf

EPSILON = 1e-7


def swish(x):
    """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
    return tf.multiply(x, tf.nn.sigmoid(x))


class PolicyModel:
    def __init__(self, network_settings):
        self.action_space = network_settings["action_space"]
        self.value_head_count = network_settings["value_head_count"]

    def build_observation_encoder(self, observations, is_train_mode):
        x_visual = observations["observation_visual_input"]
        x_vector = observations["observation_vector_input"]

        cnn_filters = [128, 64, 64]
        cnn_filer_sizes = [(3, 3), (3, 3), (3, 3)]
        cnn_strides = [(1, 1), (1, 1), (1, 1)]

        for i in range(len(cnn_filters)):
            x_visual = tf.keras.layers.Conv2D(
                filters=cnn_filters[i], 
                kernel_size=cnn_filer_sizes[i], 
                strides=cnn_strides[i], 
                padding="same",
                activation=tf.nn.relu)(x_visual)
        x_visual = tf.keras.layers.Flatten()(x_visual)
        x_vector = tf.keras.layers.Dense(256, activation=swish)(x_vector)
        x = tf.concat([x_visual, x_vector], axis=-1)

        return {
            "latent_code_x": x,
        }

    def build_policy(self, latent_codes, is_train_mode):
        x = latent_codes["latent_code_x"]

        x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)

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
        # Force nan to zero
        for i in range(len(self.action_space)):
            _entropy[i] = tf.where_v2(tf.is_nan(_entropy[i]), tf.zeros_like(_entropy[i]), _entropy[i])
        entropy = tf.reduce_sum(tf.stack(_entropy, axis=-1), axis=-1)

        policy_function = []
        for i in range(len(models)):
            policy_function.append(lambda act, k=i: models[k].prob(act))

        return {
            "sample_action": sample_action,
            "max_action": max_action,
            "value": values,
            "entropy": entropy,
            "policy_function": policy_function,
        }

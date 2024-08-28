
import tensorflow as tf

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

class PredictionModel():
    def __init__(self, network_settings):
        self.hidden_dimension = network_settings["hidden_dimension"]
        self.shared_layers = {}

    def build_winvalue_predictor(self, observations, is_train_mode):
        x = observations["player_combo_input"]
        for _ in range(4):
            x = tf.keras.layers.Dense(self.hidden_dimension, activation=leaky_relu)(x)
        x = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

        return x



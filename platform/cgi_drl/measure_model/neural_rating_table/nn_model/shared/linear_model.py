
import tensorflow as tf

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

class PredictionModel():
    def __init__(self, network_settings):
        self.hidden_dimension = network_settings["hidden_dimension"]
        self.shared_layers = {}

    def build_observation_rating_encoder(self, observations, is_train_mode):
        x_player1_combo = observations["player1_combo_input"]
        x_player2_combo = observations["player2_combo_input"]

        key = "rating_encoder_fc"
        self.shared_layers[key] = tf.keras.layers.Dense(self.hidden_dimension, activation=None)

        x_player1_combo = self.shared_layers[key](x_player1_combo)
        x_player2_combo = self.shared_layers[key](x_player2_combo)

        key = "rating_encoder_fc2"
        self.shared_layers[key] = tf.keras.layers.Dense(self.hidden_dimension, activation=None)

        x_player1_combo = self.shared_layers[key](x_player1_combo)
        x_player2_combo = self.shared_layers[key](x_player2_combo)

        key = "rating_encoder_fc3"
        self.shared_layers[key] = tf.keras.layers.Dense(self.hidden_dimension, activation=None)

        x_player1_combo = self.shared_layers[key](x_player1_combo)
        x_player2_combo = self.shared_layers[key](x_player2_combo)

        key = "rating_encoder_fc4"
        self.shared_layers[key] = tf.keras.layers.Dense(self.hidden_dimension, activation=None)

        x_player1_combo = self.shared_layers[key](x_player1_combo)
        x_player2_combo = self.shared_layers[key](x_player2_combo)


        key = "rating_encoder_fc5"
        self.shared_layers[key] = tf.keras.layers.Dense(1, activation=tf.math.exp)

        x_player1_combo = self.shared_layers[key](x_player1_combo)
        x_player2_combo = self.shared_layers[key](x_player2_combo)
        
        return x_player1_combo, x_player2_combo


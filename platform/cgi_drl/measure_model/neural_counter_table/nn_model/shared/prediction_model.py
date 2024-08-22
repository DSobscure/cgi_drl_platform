import tensorflow as tf

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

class PredictionModel():
    def __init__(self, network_settings):
        self.hidden_dimension = network_settings["hidden_dimension"]
        self.shared_layers = {}

    def build_observation_category_encoder(self, observations, is_train_mode):
        x_player1_combo = observations["player1_combo_input"]
        x_player2_combo = observations["player2_combo_input"]

        for i in range(2):
            key = f"category_encoder_fc{i}"
            self.shared_layers[key] = tf.keras.layers.Dense(self.hidden_dimension, activation=leaky_relu)
            x_player1_combo = self.shared_layers[key](x_player1_combo)
            x_player2_combo = self.shared_layers[key](x_player2_combo)

        key = "category_encoder_output"
        self.shared_layers[key] = tf.keras.layers.Dense(self.hidden_dimension, activation=None)

        x_player1_combo = self.shared_layers[key](x_player1_combo)
        x_player2_combo = self.shared_layers[key](x_player2_combo)
        
        return x_player1_combo, x_player2_combo

    def build_residual_winvalue_predictor(self, x_player1_combo, x_player2_combo, is_train_mode):
        x1 = tf.concat([x_player1_combo, x_player2_combo], axis=-1)
        x2 = tf.concat([x_player2_combo, x_player1_combo], axis=-1)
        for i in range(4):
            key = f"residual_encoder_fc{i}"
            self.shared_layers[key] = tf.keras.layers.Dense(self.hidden_dimension, activation=leaky_relu)
            x1 = self.shared_layers[key](x1)
            x2 = self.shared_layers[key](x2)
        key = "residual_encoder_output"
        self.shared_layers[key] = tf.keras.layers.Dense(1, activation=tf.nn.tanh)
        x1 = self.shared_layers[key](x1)
        x2 = self.shared_layers[key](x2)
        return (x1 - x2) / 2



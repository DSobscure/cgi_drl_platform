import tensorflow as tf
import importlib

class NeuralRatingTableTrainer():
    def __init__(self, config):
        self.algorithm_scope = config["algorithm_scope"]

        create_tensorflow_variables = config["create_tensorflow_variables"]
        with tf.variable_scope(self.algorithm_scope):  
            create_tensorflow_variables()
            PredictionModel = getattr(importlib.import_module(config["model_define_path"]), "PredictionModel")
            network_settings = config.get("network_settings", {})

            self.learning_rate_placeholder = config["learning_rate_placeholder"]
            self.optimizer = config["optimizer"]

            self.observation_placeholders = config["observation_placeholders"]
            self.observation_inputs = config["observation_inputs"]

            self.is_train_mode = tf.placeholder(dtype = tf.bool, name = 'is_train_mode')
            self.prediction_model = PredictionModel(network_settings)

            self.match_result_placeholder = tf.placeholder(shape = [None], dtype = tf.float32, name = "match_result_placeholder")
            self.match_result = tf.reshape(self.match_result_placeholder, [-1, 1])

            with tf.variable_scope("combo_rating_encoder") as combo_rating_encoder_parm_scope:
                player1_combo_rating, player2_combo_rating = self.prediction_model.build_observation_rating_encoder(self.observation_inputs, self.is_train_mode)
                bradley_terry_winvalue_prediction = player1_combo_rating / (player1_combo_rating + player2_combo_rating)
                self.player1_combo_rating, self.player2_combo_rating = tf.reshape(player1_combo_rating, [-1]), tf.reshape(player2_combo_rating, [-1])
                self.winvalue_prediction = tf.reshape(bradley_terry_winvalue_prediction, [-1])

            self.batch_size = tf.shape(self.match_result_placeholder)[0]

            bradley_terry_loss = tf.compat.v1.losses.mean_squared_error(self.match_result, bradley_terry_winvalue_prediction)
            self.winvalue_prediction_loss = bradley_terry_loss

            gradients = []

            combo_rating_encoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=combo_rating_encoder_parm_scope.name)
            combo_rating_encoder_grads = list(zip(tf.gradients(bradley_terry_loss, combo_rating_encoder_vars), combo_rating_encoder_vars))
            gradients += combo_rating_encoder_grads

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.train_op  = self.optimizer.apply_gradients(gradients)  

        self.saver = tf.train.Saver(var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.algorithm_scope))

    def set_session(self, session):
        self.session = session

    def update(self, observations, match_results, extra_settings = None):         
        feed_dict = {}
        feed_dict[self.learning_rate_placeholder] = extra_settings["learning_rate"]
        for key in self.observation_placeholders:
            feed_dict[self.observation_placeholders[key]] = observations[key]
        feed_dict[self.match_result_placeholder] = match_results
        feed_dict[self.is_train_mode] = True

        _, winvalue_prediction_loss = self.session.run([self.train_op, self.winvalue_prediction_loss], feed_dict)
        return winvalue_prediction_loss

    def get_predictions(self, observations, extra_settings = None):            
        feed_dict = {}
        for key in self.observation_placeholders:
            feed_dict[self.observation_placeholders[key]] = observations[key]
        feed_dict[self.is_train_mode] = False

        return self.session.run(self.winvalue_prediction, feed_dict = feed_dict)

    def get_rating_and_predictions(self, observations, extra_settings = None):            
        feed_dict = {}
        for key in self.observation_placeholders:
            feed_dict[self.observation_placeholders[key]] = observations[key]
        feed_dict[self.is_train_mode] = False

        return self.session.run([self.player1_combo_rating, self.player2_combo_rating, self.winvalue_prediction], feed_dict = feed_dict)

    def save(self, path, time_step):
        '''save NN model (give a directory name for the model) '''
        savePath = self.saver.save(self.session, path + "/model.ckpt", global_step=time_step)
        print("Model saved in file: %s" % savePath)

    def load(self, path):
        '''load NN model (give a directory name for the model) '''
        self.saver.restore(self.session, tf.train.latest_checkpoint(path))
        print("Model restored.")

import tensorflow as tf
import importlib
import numpy as np

class NeuralCounterTableTrainer():
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
            self.bt_prediction_placeholder = tf.placeholder(shape = [None], dtype = tf.float32, name = "bt_prediction_placeholder")
            self.bt_prediction = tf.reshape(self.bt_prediction_placeholder, [-1, 1])
            self.vq_embedding_size = config["vq_embedding_size"]
            self.vq_beta = config["vq_beta"]
            self.vq_mean_beta = config.get("vq_mean_beta", 0.25)

            with tf.variable_scope("combo_category_encoder") as combo_category_encoder_parm_scope:
                player1_combo_category_latent, player2_combo_category_latent = self.prediction_model.build_observation_category_encoder(self.observation_inputs, self.is_train_mode)

            with tf.variable_scope("combo_category_vq_embedding") as combo_category_vq_embedding_parm_scope:
                vq_embedding = tf.get_variable("vq_embedding", self.vq_embedding_size, initializer=tf.truncated_normal_initializer(stddev=0.02))
                vq_embedding_mean = tf.math.reduce_mean(vq_embedding, axis=0)
                vq_embedding_mean = tf.expand_dims(vq_embedding_mean, axis=0)

            def vq(latent_point, embedding_points):
                vq_distance = tf.norm(embedding_points - latent_point, axis=-1)
                k = tf.argmin(vq_distance, axis=-1, output_type=tf.int32)
                z_decoder = tf.gather(embedding_points, k)
                return k, z_decoder


            player1_embedding_k, player1_embedding_decoder_latent = vq(tf.expand_dims(player1_combo_category_latent, axis=-2), vq_embedding)
            player2_embedding_k, player2_embedding_decoder_latent = vq(tf.expand_dims(player2_combo_category_latent, axis=-2), vq_embedding)
            self.player1_embedding_k, self.player2_embedding_k = tf.reshape(player1_embedding_k, [-1]), tf.reshape(player2_embedding_k, [-1])
            

            with tf.variable_scope("residual_winvalue_predictor") as residual_winvalue_predictor_parm_scope:
                residual_winvalue_prediction = self.prediction_model.build_residual_winvalue_predictor(player1_embedding_decoder_latent, player2_embedding_decoder_latent, self.is_train_mode)
                self.winvalue_prediction = tf.reshape(self.bt_prediction + residual_winvalue_prediction, [-1])

            winvalue_prediction_loss = tf.compat.v1.losses.mean_squared_error(self.match_result - self.bt_prediction, residual_winvalue_prediction)
            # player1_vq_loss = tf.compat.v1.losses.mean_squared_error(tf.stop_gradient(player1_combo_category_latent), player1_embedding_decoder_latent)
            # player2_vq_loss = tf.compat.v1.losses.mean_squared_error(tf.stop_gradient(player2_combo_category_latent), player2_embedding_decoder_latent)
            # vq_loss = (player1_vq_loss + player2_vq_loss) / 2
            # player1_commit_loss = tf.compat.v1.losses.mean_squared_error(tf.stop_gradient(player1_embedding_decoder_latent), player1_combo_category_latent)
            # player2_commit_loss = tf.compat.v1.losses.mean_squared_error(tf.stop_gradient(player2_embedding_decoder_latent), player2_combo_category_latent)
            # commit_loss = (player1_commit_loss + player2_commit_loss) / 2

            # player1_vq_mean_loss = tf.compat.v1.losses.mean_squared_error(tf.stop_gradient(player1_combo_category_latent), vq_embedding_mean)
            # player2_vq_mean_loss = tf.compat.v1.losses.mean_squared_error(tf.stop_gradient(player2_combo_category_latent), vq_embedding_mean)
            # vq_loss += (player1_vq_mean_loss + player2_vq_mean_loss) / 2
            player1_vq_loss = tf.compat.v1.losses.mean_squared_error(player1_combo_category_latent, player1_embedding_decoder_latent)
            player2_vq_loss = tf.compat.v1.losses.mean_squared_error(player2_combo_category_latent, player2_embedding_decoder_latent)
            vq_loss = (player1_vq_loss + player2_vq_loss) / 2

            player1_vq_mean_loss = tf.compat.v1.losses.mean_squared_error(player1_combo_category_latent, vq_embedding_mean)
            player2_vq_mean_loss = tf.compat.v1.losses.mean_squared_error(player2_combo_category_latent, vq_embedding_mean)
            vq_mean_loss = (player1_vq_mean_loss + player2_vq_mean_loss) / 2

            self.winvalue_prediction_loss = winvalue_prediction_loss
            self.vq_loss = vq_loss
            self.commit_loss = vq_mean_loss

            gradients = []

            player1_grad_z_residual_winvalue_predictor = tf.gradients(winvalue_prediction_loss, player1_embedding_decoder_latent)[0]
            player2_grad_z_residual_winvalue_predictor = tf.gradients(winvalue_prediction_loss, player2_embedding_decoder_latent)[0]

            combo_category_encoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=combo_category_encoder_parm_scope.name)
            gradients += self.optimizer.compute_gradients(player1_combo_category_latent, combo_category_encoder_vars, grad_loss=player1_grad_z_residual_winvalue_predictor)
            gradients += self.optimizer.compute_gradients(player2_combo_category_latent, combo_category_encoder_vars, grad_loss=player2_grad_z_residual_winvalue_predictor)
            gradients += self.optimizer.compute_gradients(self.vq_beta * vq_loss, combo_category_encoder_vars)

            combo_category_vq_embedding_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=combo_category_vq_embedding_parm_scope.name)
            vq_embedding_grads = list(zip(tf.gradients(vq_loss + self.vq_mean_beta * vq_mean_loss, combo_category_vq_embedding_vars), combo_category_vq_embedding_vars))
            gradients += vq_embedding_grads

            residual_winvalue_predictor_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=residual_winvalue_predictor_parm_scope.name)
            residual_winvalue_predictor_grads = list(zip(tf.gradients(winvalue_prediction_loss, residual_winvalue_predictor_vars), residual_winvalue_predictor_vars))
            gradients += residual_winvalue_predictor_grads

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.train_op  = self.optimizer.apply_gradients(gradients)  

        self.saver = tf.train.Saver(var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.algorithm_scope))

    def set_session(self, session):
        self.session = session

    def update(self, observations, match_results, bt_predictions, extra_settings = None):         
        feed_dict = {}
        feed_dict[self.learning_rate_placeholder] = extra_settings["learning_rate"]
        for key in self.observation_placeholders:
            feed_dict[self.observation_placeholders[key]] = observations[key]
        feed_dict[self.match_result_placeholder] = match_results
        feed_dict[self.bt_prediction_placeholder] = bt_predictions
        feed_dict[self.is_train_mode] = True

        _, winvalue_prediction_loss, vq_loss, commit_loss, k1, k2 = self.session.run([self.train_op, self.winvalue_prediction_loss, self.vq_loss, self.commit_loss, self.player1_embedding_k, self.player2_embedding_k], feed_dict)
        return winvalue_prediction_loss, vq_loss, commit_loss

    def get_predictions(self, observations, bt_predictions, extra_settings = None):            
        feed_dict = {}
        for key in self.observation_placeholders:
            feed_dict[self.observation_placeholders[key]] = observations[key]
        feed_dict[self.is_train_mode] = False
        feed_dict[self.bt_prediction_placeholder] = bt_predictions

        return self.session.run([self.winvalue_prediction, self.player1_embedding_k, self.player2_embedding_k], feed_dict = feed_dict)

    def save(self, path, time_step):
        '''save NN model (give a directory name for the model) '''
        savePath = self.saver.save(self.session, path + "/model.ckpt", global_step=time_step)
        print("Model saved in file: %s" % savePath)

    def load(self, path):
        '''load NN model (give a directory name for the model) '''
        self.saver.restore(self.session, tf.train.latest_checkpoint(path))
        print("Model restored.")

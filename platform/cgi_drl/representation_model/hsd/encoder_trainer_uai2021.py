import tensorflow as tf
import numpy as np
import os, sys
import importlib

class EncoderTrainer():
    def __init__(self, config):
        self.cnn_embd_size = config["cnn_embd_size"]
        self.Ks = config["Ks"]
        self.D = config["D"]
        self.optimizer = config["optimizer"]
        self.latent_block_sizes = config["latent_block_sizes"]
        self.cnn_latent_block_sizes = config["cnn_latent_block_sizes"]
        self.output_dimensions = config["output_dimensions"]
        self.hierarchy_layer = config["hierarchy_layer"]
        self.learning_rate_placeholder = config["learning_rate_placeholder"]

        self.visual_observation_dimension = config["visual_observation_dimension"]
        self.compressed_visual_observation_dimension = config["compressed_visual_observation_dimension"]
        self.visual_observation_frame_count = config["visual_observation_frame_count"]
        self.action_count = config["action_count"]
        self.vq_alpha = config["vq_alpha"]
        self.vq_beta = config["vq_beta"]

        if "training_preprocessing_function" in config:
            self.training_preprocessing_function = config["training_preprocessing_function"]
        else:
            self.training_preprocessing_function = lambda _x: _x
        
        EncoderModel = getattr(importlib.import_module(config["model_define_path"]), "EncoderModel")

        if "network_settings" in config:
            network_settings = config["network_settings"]
        else:
            network_settings = {}
        network_settings["D"] = self.D
        network_settings["latent_block_sizes"] = self.latent_block_sizes
        network_settings["output_dimensions"] = self.output_dimensions
        network_settings["hierarchy_layer"] = self.hierarchy_layer
        network_settings["visula_observation_frame_count"] = self.visual_observation_frame_count 

        with tf.variable_scope("Hpssd2"):
            with tf.variable_scope("cnn_embd") as cnn_embd_parm_scope:
                cnn_embed = tf.get_variable("cnn_embd", self.cnn_embd_size, initializer=tf.truncated_normal_initializer(stddev=0.02))
            with tf.variable_scope("embd") as embd_parm_scope:
                embeds = []
                for i in range(self.hierarchy_layer):
                    embeds.append(tf.get_variable("embd_{}".format(i), [self.Ks[i], self.D // self.latent_block_sizes[i]], initializer=tf.truncated_normal_initializer(stddev=0.02)))
            
            self.visual_observation_placeholder = tf.placeholder(shape =  [None] + [self.visual_observation_frame_count] + self.visual_observation_dimension, dtype = tf.float32, name = "visual_observation_placeholder")
            self.compressed_visual_observation_placeholder = tf.placeholder(shape =  [None] + [self.visual_observation_frame_count] + self.compressed_visual_observation_dimension, dtype = tf.float32, name = "compressed_visual_observation_placeholder")

            self.is_train_mode = tf.placeholder(dtype = tf.bool, name = 'is_train_mode')
            self.action_placeholder = tf.placeholder(shape = [None, self.action_count], dtype = tf.float32, name = "action_placeholder")
            autoencoder = EncoderModel(network_settings)
            self.feature_map_sizes = autoencoder.feature_map_sizes

            visual_observation_input = tf.cond(self.is_train_mode, lambda: self.training_preprocessing_function(self.visual_observation_placeholder), lambda: self.visual_observation_placeholder)
            visual_observation_input = tf.transpose(visual_observation_input, [0,2,3,4,1])
            visual_observation_input = tf.reshape(visual_observation_input, [-1, self.visual_observation_dimension[0], self.visual_observation_dimension[1], self.visual_observation_dimension[2] * self.visual_observation_frame_count])

            compressed_visual_observation_input = self.compressed_visual_observation_placeholder
            compressed_visual_observation_input = tf.transpose(compressed_visual_observation_input, [0,2,3,4,1])
            compressed_visual_observation_input = tf.reshape(compressed_visual_observation_input, [-1, self.compressed_visual_observation_dimension[0], self.compressed_visual_observation_dimension[1], self.compressed_visual_observation_dimension[2] * self.visual_observation_frame_count])

            with tf.variable_scope("forward") as forward_scope:
                self.batch_size = tf.shape(self.visual_observation_placeholder)[0]

                def vq(latent_point, embedding_points, is_train_mode):
                    vq_distance = tf.norm(embedding_points - latent_point, axis=-1)
                    k = tf.argmin(vq_distance, axis=-1, output_type=tf.int32)
                    z_decoder = tf.gather(embedding_points, k)
                    return k, z_decoder

                with tf.variable_scope("cnn_encoder") as cnn_enc_parm_scope:
                    cnn_output = autoencoder.build_cnn_encoder(visual_observation_input, self.is_train_mode)
                    cnn_k, cnn_z_decoder = vq(tf.expand_dims(cnn_output, axis=-2), cnn_embed, self.is_train_mode)
                    self.cnn_latent_codes = cnn_output
                encoder_inputs = []
                encoder_outputs = []
                ks = []
                z_decoders = []
                self.hierarchy_usages = []
                self.continuous_latent_codes = []
                encoder_input = cnn_output
                enc_parm_scopes = []

                for i in range(self.hierarchy_layer):
                    with tf.variable_scope("encoder_h{}".format(i)) as enc_parm_scope:
                        enc_parm_scopes.append(enc_parm_scope)
                        hierarchy_usage = tf.placeholder(shape = [None], dtype = tf.float32, name = 'hierarchy_usage{}'.format(i))
                        self.hierarchy_usages.append(hierarchy_usage)
                        encoder_inputs.append(encoder_input)
                        encoder_output = autoencoder.build_hierarchy_encoder(encoder_input, i, self.is_train_mode)
                        encoder_outputs.append(encoder_output)
                        self.continuous_latent_codes.append(encoder_output)
                        _k, _z_decoder = vq(tf.expand_dims(encoder_output, axis=-2), embeds[i], self.is_train_mode)
                        ks.append(_k)
                        z_decoders.append(_z_decoder)
                        encoder_input = encoder_output

                dec_param_scopes = []
                decoder_input = z_decoders[-1]
                decoder_outputs = []
                
                for i in range(self.hierarchy_layer - 1, -1, -1):
                    with tf.variable_scope("decoder_h{}".format(i)) as dec_param_scope:
                        dec_param_scopes.append(dec_param_scope)
                        decoder_output = autoencoder.build_hierarchy_decoder(decoder_input, i, self.is_train_mode)
                        decoder_outputs.append(decoder_output)
                        usage = self.hierarchy_usages[i]
                        usage = tf.expand_dims(usage, axis=-1)
                        if i == 0:
                            decoder_input = tf.multiply(usage, tf.contrib.layers.flatten(decoder_output)) + tf.multiply(1 - usage, tf.contrib.layers.flatten(cnn_z_decoder))
                        else:
                            decoder_input = tf.multiply(usage, tf.contrib.layers.flatten(decoder_output)) + tf.multiply(1 - usage, tf.contrib.layers.flatten(z_decoders[i - 1]))

                dec_param_scopes.reverse()
                decoder_outputs.reverse()

                with tf.variable_scope("cnn_policy") as cnn_policy_parm_scope:
                    self.policy_output = autoencoder.build_policy(decoder_input, self.action_count, self.is_train_mode)
                    self.ac = tf.argmax(self.policy_output, axis=-1) 
                with tf.variable_scope("cnn_decoder") as cnn_dec_param_scope:  
                    self.compressed_x = autoencoder.build_cnn_decoder(decoder_input, self.is_train_mode)

                with tf.variable_scope("loss"):
                    policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.action_placeholder, logits=self.policy_output))
                    l2_loss = tf.losses.get_regularization_loss()
                    policy_loss += l2_loss
                    compressed_loss = tf.compat.v1.losses.huber_loss(compressed_visual_observation_input, self.compressed_x, reduction=tf.losses.Reduction.MEAN)
                    compressed_loss = compressed_loss
                    
                    # vector quantization loss
                    vq_loss = 0
                    cnn_vq_loss = tf.compat.v1.losses.huber_loss(tf.stop_gradient(cnn_output), cnn_z_decoder, reduction=tf.losses.Reduction.MEAN)
                    vq_loss += cnn_vq_loss
                    vq_losses = []
                    for i in range(self.hierarchy_layer):       
                        vq_losses.append(tf.compat.v1.losses.huber_loss(tf.stop_gradient(encoder_outputs[i]), z_decoders[i], reduction=tf.losses.Reduction.MEAN))     
                        vq_loss += vq_losses[i]
                    vq_loss /= self.hierarchy_layer + 1
                    # commit loss
                    commit_loss = 0
                    cnn_commit_loss = tf.compat.v1.losses.huber_loss(tf.stop_gradient(cnn_z_decoder), cnn_output, reduction=tf.losses.Reduction.MEAN)
                    commit_loss += cnn_commit_loss
                    commit_losses = []
                    for i in range(self.hierarchy_layer):
                        commit_losses.append(tf.compat.v1.losses.huber_loss(tf.stop_gradient(z_decoders[i]), encoder_outputs[i], reduction=tf.losses.Reduction.MEAN))
                        commit_loss += commit_losses[i]
                    commit_loss /= self.hierarchy_layer + 1
                    
                    self.policy_loss = policy_loss 
                    self.compressed_loss = compressed_loss
                    self.vq_loss = vq_loss
                    self.commit_loss = commit_loss
                    self.compressed_x = tf.reshape(self.compressed_x, [-1, self.compressed_visual_observation_dimension[0], self.compressed_visual_observation_dimension[1], self.compressed_visual_observation_dimension[2], self.visual_observation_frame_count])
                    self.compressed_x = tf.transpose(self.compressed_x, [0,4,1,2,3])

            with tf.variable_scope("backward"):
                alpha = self.vq_alpha
                beta = self.vq_beta
                gradients = []

                # encoder gradients
                print("Start compute encoder gradients")
                grad_z_policy = tf.gradients(policy_loss, cnn_z_decoder)[0]
                grad_z_decoder = tf.gradients(compressed_loss, cnn_z_decoder)[0]
                cnn_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, cnn_enc_parm_scope.name)
                cnn_encoder_grads = self.optimizer.compute_gradients(cnn_output, cnn_encoder_vars, grad_loss=grad_z_policy)
                cnn_encoder_grads += self.optimizer.compute_gradients(cnn_output, cnn_encoder_vars, grad_loss=grad_z_decoder)
                cnn_encoder_grads += self.optimizer.compute_gradients(beta * cnn_commit_loss, cnn_encoder_vars)
                gradients += cnn_encoder_grads

                # encoder_vars = cnn_encoder_vars
                for i in range(self.hierarchy_layer):
                    print("Start compute hierarchy encoder gradients")
                    grad_z_policy = tf.gradients(policy_loss, z_decoders[i])[0]
                    grad_z_decoder = tf.gradients(compressed_loss, z_decoders[i])[0]
                    encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, enc_parm_scopes[i].name)
                    encoder_grads = self.optimizer.compute_gradients(encoder_outputs[i], encoder_vars, grad_loss=grad_z_policy)
                    encoder_grads += self.optimizer.compute_gradients(encoder_outputs[i], encoder_vars, grad_loss=grad_z_decoder)
                    encoder_grads += self.optimizer.compute_gradients(beta * commit_losses[i], encoder_vars)
                    gradients += encoder_grads

                # embedding gradients
                print("Start compute embedding gradients")
                cnn_embed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, cnn_embd_parm_scope.name)
                cnn_embed_grads = list(zip(tf.gradients(cnn_vq_loss, cnn_embed_vars), cnn_embed_vars))
                gradients += cnn_embed_grads
                for i in range(self.hierarchy_layer):
                    embed_grads = list(zip(tf.gradients(alpha * vq_losses[i], embeds[i]), [embeds[i]]))
                    gradients += embed_grads 

                # policy gradients
                print("Start compute policy gradients")
                policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, cnn_policy_parm_scope.name)
                policy_grads = list(zip(tf.gradients(policy_loss, policy_vars), policy_vars))
                gradients += policy_grads
                for i in range(self.hierarchy_layer):
                    decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, dec_param_scopes[i].name)
                    decoder_grads = list(zip(tf.gradients(policy_loss, decoder_vars), decoder_vars))
                    gradients += decoder_grads

                # decoder gradients
                print("Start compute decoder gradients")
                cnn_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, cnn_dec_param_scope.name)
                cnn_decoder_grads = list(zip(tf.gradients(compressed_loss, cnn_decoder_vars), cnn_decoder_vars))
                gradients += cnn_decoder_grads
                for i in range(self.hierarchy_layer):
                    decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, dec_param_scopes[i].name)
                    decoder_grads = list(zip(tf.gradients(compressed_loss, decoder_vars), decoder_vars))
                    gradients += decoder_grads

        self.ks = ks
        self.cnn_k = cnn_k

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op  = self.optimizer.apply_gradients(gradients)  
                 
        self.saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Hpssd2"))

        correct_prediction = tf.equal(tf.argmax(self.policy_output,1), tf.argmax(self.action_placeholder,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def set_session(self, session):
        self.session = session

    def update_fixed_encoder(self):
        self.session.run(self.update_fixed_op)

    def _parse_state_code(self, ks, code_level):
        code = ""          
        for x in range(self.latent_block_sizes[code_level]):
            code = "{},{}".format(code, ks[code_level][x])
        return code

    def _parse_cnn_state_code(self, k):
        code = ""          
        for x in range(self.cnn_latent_block_sizes):
            code = "{},{}".format(code, k[x])
        return code

    def get_continuous_latent_codes(self, observations, code_level, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}
        if "feed_dict" not in extra_settings:
            feed_dict = {}
            extra_settings["feed_dict"] = feed_dict                 
        feed_dict = extra_settings["feed_dict"]
        feed_dict[self.is_train_mode] = False

        if self.visual_observation_dimension != None:
            feed_dict[self.visual_observation_placeholder] = observations["visual"]
        batch_size = len(observations["visual"])

        if code_level >= 0:
            codes = self.session.run(self.continuous_latent_codes[code_level], feed_dict = feed_dict)
        elif code_level == -1:
            codes = self.session.run(self.cnn_latent_codes, feed_dict = feed_dict)
        else:
            codes = []
            for i in range(len(observations["visual"])):
                codes.append("none")
        return np.reshape(codes, [batch_size, -1]) 

    def get_discrite_latent_codes(self, observations, code_level, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}
        if "feed_dict" not in extra_settings:
            feed_dict = {}
            extra_settings["feed_dict"] = feed_dict                 
        feed_dict = extra_settings["feed_dict"]
        feed_dict[self.is_train_mode] = False

        if self.visual_observation_dimension != None:
            feed_dict[self.visual_observation_placeholder] = observations["visual"]

        if code_level >= 0:
            ks = (self.session.run(self.ks, feed_dict = feed_dict))
            codes = []
            for i in range(len(observations["visual"])):
                _ks = []
                for j in range(self.hierarchy_layer):
                    _ks.append(ks[j][i])
                code = self._parse_state_code(_ks, code_level)
                codes.append(code)
        elif code_level == -1:
            cnn_k = self.session.run(self.cnn_k, feed_dict = feed_dict)
            codes = []
            for i in range(len(observations["visual"])):
                code = self._parse_cnn_state_code(cnn_k[i])
                codes.append(code)
        else:
            codes = []
            for i in range(len(observations["visual"])):
                codes.append("none")
        return codes

    def get_compressed_x(self, transitions, extra_settings = {}):
        observation_batch = transitions["observation"]
        if "feed_dict" not in extra_settings:
            feed_dict = {}
            extra_settings["feed_dict"] = feed_dict                 
        feed_dict = extra_settings["feed_dict"]

        if self.visual_observation_dimension != None:
            feed_dict[self.visual_observation_placeholder] = observation_batch["visual"]
        feed_dict[self.is_train_mode] = False

        for i in range(self.hierarchy_layer):
            feed_dict[self.hierarchy_usages[i]] = extra_settings["hierarchy_usages"][i]
                
        return self.session.run(self.compressed_x, feed_dict = feed_dict)

    def update(self, transitions, extra_settings = {}):
        observation_batch = transitions["observation"]
        compressed_observation_batch = transitions["compressed_observation"]
        action_batch = transitions["action"]
        if "feed_dict" not in extra_settings:
            feed_dict = {}
            extra_settings["feed_dict"] = feed_dict                 
        feed_dict = extra_settings["feed_dict"]

        if self.visual_observation_dimension != None:
            feed_dict[self.visual_observation_placeholder] = observation_batch["visual"]
        if self.visual_observation_dimension != None:
            feed_dict[self.compressed_visual_observation_placeholder] = compressed_observation_batch["visual"]
        feed_dict[self.is_train_mode] = True
        feed_dict[self.action_placeholder] = action_batch
        feed_dict[self.learning_rate_placeholder] = extra_settings["learning_rate"]
        batch_size = len(observation_batch["visual"])

        for i in range(self.hierarchy_layer):
            feed_dict[self.hierarchy_usages[i]] = np.random.uniform(0, 1, batch_size)

        _, policy_loss, compressed_loss, vq_loss, commit_loss = self.session.run([self.train_op, self.policy_loss, self.compressed_loss, self.vq_loss, self.commit_loss], feed_dict=feed_dict)
        return policy_loss, compressed_loss, vq_loss, commit_loss

    def get_accuracy(self, transitions, extra_settings = {}):
        observation_batch = transitions["observation"]
        action_batch = transitions["action"]
        if "feed_dict" not in extra_settings:
            feed_dict = {}
            extra_settings["feed_dict"] = feed_dict                 
        feed_dict = extra_settings["feed_dict"]

        if self.visual_observation_dimension != None:
            feed_dict[self.visual_observation_placeholder] = observation_batch["visual"]     
        feed_dict[self.is_train_mode] = False
        feed_dict[self.action_placeholder] = action_batch   

        for i in range(self.hierarchy_layer):
            feed_dict[self.hierarchy_usages[i]] = extra_settings["hierarchy_usages"][i]

        return self.session.run(self.accuracy, feed_dict = feed_dict)

    def save(self, path, time_step):
        '''save NN model (give a directory name for the model) '''
        savePath = self.saver.save(self.session, path + "/model.ckpt", global_step=time_step)
        print("Model saved in file: %s" % savePath)

    def load(self, path):
        '''load NN model (give a directory name for the model) '''
        self.saver.restore(self.session, tf.train.latest_checkpoint(path))
        print("Model restored.")
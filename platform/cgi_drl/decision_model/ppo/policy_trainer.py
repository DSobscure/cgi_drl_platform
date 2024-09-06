import tensorflow as tf
import importlib
import numpy as np
   
class PolicyTrainer():
    def __init__(self, config):
        self.algorithm_scope = config["algorithm_scope"]

        self.action_space = config["action_space"]
        if isinstance(self.action_space, int):
            self.action_space = [self.action_space]
        self.branch_count = len(self.action_space)
        self.value_head_count = config["value_head_count"]

        self.use_rnn = config["use_rnn"]
        self.max_gradient_norm = config["max_gradient_norm"]

        create_tensorflow_variables = config["create_tensorflow_variables"]
        with tf.name_scope(self.algorithm_scope):  
            create_tensorflow_variables()
            self.invertible_value_functions = config["invertible_value_functions"]
            PolicyModel = getattr(importlib.import_module(config["model_define_path"]), "PolicyModel")
            network_settings = config.get("network_settings", {})
            network_settings["action_space"] = self.action_space
            network_settings["value_head_count"] = self.value_head_count
            if self.use_rnn:
                self.memory_size = config["memory_size"]
                self.rnn_sequence_length_placeholder = config["rnn_sequence_length_placeholder"]
                self.rnn_sequence_length = config["rnn_sequence_length"]
                self.rnn_burn_in_length = config["rnn_burn_in_length"]
                if self.rnn_burn_in_length > self.rnn_sequence_length:
                    raise ValueError("rnn_burn_in_length:{} need to be smaller than rnn sequence length:{}.".format(self.rnn_burn_in_length, self.rnn_sequence_length))

                network_settings["rnn_sequence_length"] = self.rnn_sequence_length_placeholder
                network_settings["rnn_burn_in_length"] = self.rnn_burn_in_length
                network_settings["memory_size"] = config["memory_size"]

        
            self.learning_rate_placeholder = config["learning_rate_placeholder"]
            self.clip_epsilon_placeholder = config["clip_epsilon_placeholder"]
            self.entropy_coefficient_placeholder = config["entropy_coefficient_placeholder"]
            self.value_coefficient_placeholder = config["value_coefficient_placeholder"]
            self.optimizer = config["optimizer"]

            self.observation_placeholders = config["observation_placeholders"]
            self.observation_inputs = config["observation_inputs"]

            self.action_placeholder = tf.compat.v1.placeholder(shape = [None, self.branch_count], dtype = tf.int32, name = "action_placeholder")
            self.return_placeholder = tf.compat.v1.placeholder(shape = [None, self.value_head_count], dtype = tf.float32, name = "return_placeholder")
            self.advantage_placeholder = tf.compat.v1.placeholder(shape = [None], dtype = tf.float32, name = "advantage_placeholder")

            self.policy_scope = "policy"
            self.old_policy_scope = "old_policy"

            self.is_train_mode = tf.compat.v1.placeholder(dtype = tf.bool, name = 'is_train_mode')            
            self.policy_model = PolicyModel(network_settings)

            with tf.name_scope("forward") as forward_scope:
                with tf.name_scope(self.policy_scope) as policy_parm_scope:
                    encoder_output = self.policy_model.build_observation_encoder(self.observation_inputs, self.is_train_mode)
                    if self.use_rnn:
                        encoder_output, memory_output = self.policy_model.build_recurrent_encoder(encoder_output, self.is_train_mode)
                        self.memory_output = tf.identity(memory_output, name="memory_output")
                    encoder_output["action_placeholder"] = self.action_placeholder
                    policy_outouts = self.policy_model.build_policy(encoder_output, self.is_train_mode)
                    self.sample_action = policy_outouts["sample_action"]
                    self.max_action = policy_outouts["max_action"]
                    self.value = policy_outouts["value"]
                    self.entropy = policy_outouts["entropy"]
                    self.policy_function = policy_outouts["policy_function"]
                    # model
                    self.one_hot_action = tf.concat([
                        tf.one_hot(tf.cast(self.sample_action[:,i], tf.int32), branch)
                        for i, branch in enumerate(self.action_space)
                    ], axis=1)

                with tf.name_scope(self.old_policy_scope) as old_policy_parm_scope:
                    encoder_output = self.policy_model.build_observation_encoder(self.observation_inputs, tf.constant(False))
                    if self.use_rnn:
                        encoder_output, memory_output = self.policy_model.build_recurrent_encoder(encoder_output, tf.constant(False))
                    encoder_output["action_placeholder"] = self.action_placeholder
                    policy_outouts = self.policy_model.build_policy(encoder_output, tf.constant(False))
                    self.old_sample_action = policy_outouts["sample_action"]
                    self.old_max_action = policy_outouts["max_action"]
                    self.old_value = policy_outouts["value"]
                    self.old_entropy = policy_outouts["entropy"]
                    self.old_policy_function = policy_outouts["policy_function"]

                with tf.name_scope("loss"):
                    # value clipping
                    value_losses = []
                    for i in range(self.value_head_count):
                        transformed_return = self.invertible_value_functions[i](tf.reshape(self.return_placeholder[:,i], [-1, 1]), False)
                        value_loss = tf.losses.huber_loss(transformed_return, self.value[i], reduction=tf.losses.Reduction.MEAN)
                        value_losses.append(value_loss)
                    # Policy loss
                    ratio = tf.reduce_prod(
                        tf.stack([self.policy_function[branch](self.action_placeholder[:, branch]) for branch in range(self.branch_count)], axis=1), 
                        axis=1)
                    old_ratio = tf.reduce_prod(
                        tf.stack([self.old_policy_function[branch](self.action_placeholder[:, branch]) for branch in range(self.branch_count)], axis=1), 
                        axis=1)
                    total_ratio = ratio / old_ratio
                    total_ratio = tf.squeeze(total_ratio)

                    # PPO loss
                    p_opt_a = total_ratio * self.advantage_placeholder
                    p_opt_b = tf.clip_by_value(total_ratio, 1 - self.clip_epsilon_placeholder, 1 + self.clip_epsilon_placeholder) * self.advantage_placeholder
                    surrogate_loss = tf.math.reduce_mean(tf.minimum(p_opt_a, p_opt_b))

                    entropy = tf.math.reduce_mean(self.entropy)
                    self.loss = -surrogate_loss + tf.math.reduce_sum(tf.math.multiply(self.value_coefficient_placeholder, value_losses)) + self.entropy_coefficient_placeholder * -entropy
                    self.surrogate_loss, self.value_losses, self.entropy = surrogate_loss, value_losses, entropy

                    policy_no_clip_event = tf.equal(total_ratio, tf.clip_by_value(total_ratio, 1 - self.clip_epsilon_placeholder, 1 + self.clip_epsilon_placeholder))
                    self.policy_clip_event_ratio = tf.reduce_mean(1 - tf.cast(policy_no_clip_event, tf.float32))


            with tf.name_scope('backward'):
                #Policy Grads
                policy_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=policy_parm_scope)
                gradients = tf.gradients(self.loss, policy_vars)
                gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                gradients = list(zip(gradients, policy_vars))               

                policy_vars = tf.compat.v1.global_variables(scope=policy_parm_scope)
                old_policy_vars = tf.compat.v1.global_variables(scope=old_policy_parm_scope)

                update_old_policy_op = [oldp.assign(p) for p, oldp in zip(policy_vars, old_policy_vars)]
                self.update_old_policy_op = update_old_policy_op

            tf.identity(self.one_hot_action, name="one_hot_action")
            values = []
            for i in range(self.value_head_count):
                values.append(tf.reshape(self.invertible_value_functions[i](self.value[i], True), [-1]))
            self.value = values
            tf.identity(self.value, name="value")
            extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_ops):
                self.train_op  = self.optimizer.apply_gradients(gradients)   
        self.saver = tf.compat.v1.train.Saver(var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.algorithm_scope))

    def set_session(self, session):
        self.session = session

    def update(self, transitions, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}
        observations = transitions["observations"]
        actions = transitions["actions"]
        returns = transitions["returns"]
        advantages = transitions["advantages"]
        if "feed_dict" not in extra_settings:
            feed_dict = {}
            extra_settings["feed_dict"] = feed_dict   
        feed_dict = extra_settings["feed_dict"]
        feed_dict[self.learning_rate_placeholder] = extra_settings["learning_rate"]
        feed_dict[self.clip_epsilon_placeholder] = extra_settings["clip_epsilon"]
        feed_dict[self.entropy_coefficient_placeholder] = extra_settings["entropy_coefficient"]
        feed_dict[self.value_coefficient_placeholder] = extra_settings["value_coefficient"]

        for key in self.observation_placeholders:
            feed_dict[self.observation_placeholders[key]] = observations[key]
        feed_dict[self.action_placeholder] = actions
        feed_dict[self.return_placeholder] = returns
        feed_dict[self.advantage_placeholder] = advantages
        feed_dict[self.is_train_mode] = True

        if self.use_rnn:
            feed_dict[self.rnn_sequence_length_placeholder] = self.rnn_sequence_length
        _, loss, surrogate_loss, value_losses, entropy, policy_clip_event_ratio = self.session.run([self.train_op, self.loss, self.surrogate_loss, self.value_losses, self.entropy, self.policy_clip_event_ratio], feed_dict)
        return loss, surrogate_loss, value_losses, entropy, policy_clip_event_ratio

    def get_max_actions(self, observations, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}
        if "feed_dict" not in extra_settings:
            feed_dict = {}
            extra_settings["feed_dict"] = feed_dict                 
        feed_dict = extra_settings["feed_dict"]

        for key in self.observation_placeholders:
            feed_dict[self.observation_placeholders[key]] = observations[key]
            batch_size = len(observations[key])
        feed_dict[self.is_train_mode] = False
        feed_dict[self.action_placeholder] = np.zeros([batch_size, self.branch_count])

        if self.use_rnn:
            feed_dict[self.rnn_sequence_length_placeholder] = 1
            return self.session.run([self.max_action, self.memory_output], feed_dict = feed_dict)
        else:
            return self.session.run(self.max_action, feed_dict = feed_dict)

    def sample_actions(self, observations, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}
        if "feed_dict" not in extra_settings:
            feed_dict = {}
            extra_settings["feed_dict"] = feed_dict                 
        feed_dict = extra_settings["feed_dict"]

        for key in self.observation_placeholders:
            feed_dict[self.observation_placeholders[key]] = observations[key]
            batch_size = len(observations[key])
        feed_dict[self.is_train_mode] = False
        feed_dict[self.action_placeholder] = np.zeros([batch_size, self.branch_count])

        if self.use_rnn:
            feed_dict[self.rnn_sequence_length_placeholder] = 1
            return self.session.run([self.sample_action, self.memory_output], feed_dict = feed_dict)
        else:
            return self.session.run(self.sample_action, feed_dict = feed_dict)

    def sample_actions_and_get_values(self, observations, extra_settings = None):
        if extra_settings == None:
            extra_settings = {}
        if "feed_dict" not in extra_settings:
            feed_dict = {}
            extra_settings["feed_dict"] = feed_dict                 
        feed_dict = extra_settings["feed_dict"]

        for key in self.observation_placeholders:
            feed_dict[self.observation_placeholders[key]] = observations[key]
            batch_size = len(observations[key])
        feed_dict[self.is_train_mode] = False
        feed_dict[self.action_placeholder] = np.zeros([batch_size, self.branch_count])

        if self.use_rnn:
            feed_dict[self.rnn_sequence_length_placeholder] = 1
            return self.session.run([self.sample_action, self.value, self.memory_output], feed_dict = feed_dict)
        else:
            return self.session.run([self.sample_action, self.value], feed_dict = feed_dict)

    def update_old_policy(self):
        self.session.run(self.update_old_policy_op)

    def save(self, path, time_step):
        '''save NN model (give a directory name for the model) '''
        savePath = self.saver.save(self.session, path + "/model.ckpt", global_step=time_step)
        print("Model saved in file: %s" % savePath)

    def load(self, path):
        '''load NN model (give a directory name for the model) '''
        self.saver.restore(self.session, tf.train.latest_checkpoint(path))
        print("Model restored.")

    def save_to_agent_pool(self, agent_pool_path, time_step=None):
        existing_checkpoints = self.saver._last_checkpoints
        self.saver.set_last_checkpoints_with_time([])
        savePath = self.saver.save(self.session, agent_pool_path + "/model.ckpt", global_step=time_step)
        print("Model saved to an agent pool: %s" % savePath)
        self.saver.set_last_checkpoints_with_time(existing_checkpoints)



import tensorflow as tf
import tensorflow.contrib.layers as layers


class EncoderModel():
    def __init__(self, network_settings):
        if "visula_observation_frame_count" in network_settings:
            self.visula_observation_frame_count = network_settings["visula_observation_frame_count"]
        else:
            self.visula_observation_frame_count = 1
        self.D = network_settings["D"]
        self.latent_block_sizes = network_settings["latent_block_sizes"]
        self.output_dimensions = network_settings["output_dimensions"]
        self.hierarchy_layer = network_settings["hierarchy_layer"]
        self.feature_map_sizes = []
        for i in range(self.hierarchy_layer):
            self.feature_map_sizes.append([self.latent_block_sizes[i],1])

    def build_cnn_encoder(self, x, is_train_mode):
        cnn_filters = [16, 32]
        cnn_filer_sizes = [(8, 8), (4, 4)]
        cnn_strides = [(4, 4), (2, 2)]
        last_index = len(cnn_filters) - 1
        for i in range(len(cnn_filters)):
            if i != last_index:
                x = tf.contrib.layers.conv2d(x, cnn_filters[i], cnn_filer_sizes[i], cnn_strides[i], activation_fn=tf.nn.relu, scope='cnn_'+str(i))
            else:
                x = tf.contrib.layers.conv2d(x, cnn_filters[i], cnn_filer_sizes[i], cnn_strides[i], activation_fn=tf.nn.tanh, scope='cnn_'+str(i))
        x = tf.reshape(x, [-1, 121, 32])
        return  x

    def build_cnn_decoder(self, x, is_train_mode):
        x = tf.reshape(x, [-1, 11, 11, 32])
        dcnn_kernel_sizes = [[4,4,16,32],[8,8,1 * self.visula_observation_frame_count,16]]
        dcnn_strides = [(1, 2, 2, 1),(1, 4, 4, 1)]
        batch_size = tf.shape(x)[0]
        dcnn_output_shape = [[batch_size,21,21,16],[batch_size,84,84,1 * self.visula_observation_frame_count]]
        last_index = len(dcnn_kernel_sizes) - 1
        for i in range(len(dcnn_kernel_sizes)):
            weight = tf.get_variable("dcnn_weight_{}".format(i), shape=dcnn_kernel_sizes[i], initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable("dcnn_bias_{}".format(i), initializer=tf.zeros([dcnn_kernel_sizes[i][-2]]))
            x = tf.nn.conv2d_transpose(x, weight, dcnn_output_shape[i], dcnn_strides[i])
            x = tf.add(x, bias)
            if i != last_index:
                x = tf.nn.relu(x)
        x = tf.nn.sigmoid(x)
        return x

    def build_policy(self, x, action_count, is_train_mode):
        x = layers.flatten(x)
        x = layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = layers.fully_connected(x, action_count, activation_fn=tf.nn.tanh)
        return x

    def build_hierarchy_encoder(self, x, hierarchy_index, is_train_mode):
        x = layers.flatten(x)
        D = self.D // self.latent_block_sizes[hierarchy_index]
        x = layers.fully_connected(x, self.D, activation_fn=tf.nn.relu)
        x = layers.fully_connected(x, self.D, activation_fn=tf.nn.tanh)
        x = tf.reshape(x, [-1, self.latent_block_sizes[hierarchy_index], D])
        return x

    def build_hierarchy_decoder(self, x, hierarchy_index, is_train_mode):
        x = layers.flatten(x)
        x = layers.fully_connected(x, self.D, activation_fn=tf.nn.relu)
        output_dimension = 1
        for d in self.output_dimensions[hierarchy_index]:
            output_dimension *= d
        x = layers.fully_connected(x, output_dimension, activation_fn=tf.nn.tanh)
        x = tf.reshape(x, [-1] + self.output_dimensions[hierarchy_index])
        return x
    


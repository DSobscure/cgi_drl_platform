import tensorflow as tf

class UAI2021Template(dict):
    def __init__(self, config):
        self["learning_rate_placeholder"] = config.get("learning_rate_placeholder", tf.placeholder(dtype = tf.float32))
        self["optimizer"] = config.get("optimizer", tf.train.AdamOptimizer(self["learning_rate_placeholder"]))
        self["visual_observation_dimension"] = config.get("visual_observation_dimension", [72, 128, 3])
        self["compressed_visual_observation_dimension"] = config.get("compressed_visual_observation_dimension", [72, 128, 3])
        self["visual_observation_frame_count"] = config.get("visual_observation_frame_count", 4)
        self["model_define_path"] = config.get("model_define_path", "representation_model.hsd.nn_model.rgsk.encoder_model_uai2021")
        self["cnn_embd_size"] = config.get("cnn_embd_size", [256, 32])
        self["Ks"] = config.get("Ks", [2])
        self["D"] = config.get("D", 500)
        self["latent_block_sizes"] = config.get("latent_block_sizes", [20])
        self["cnn_latent_block_sizes"] = config.get("cnn_latent_block_sizes", 144)
        self["output_dimensions"] = config.get("output_dimensions", [[144, 32]])
        self["hierarchy_layer"] = config.get("hierarchy_layer", 1)
        self["vq_alpha"] = config.get("vq_alpha", 1)
        self["vq_beta"] = config.get("vq_beta", 0.25)
        self["training_preprocessing_function"] = config.get("training_preprocessing_function", lambda x : x + tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=4.0, dtype=tf.float32) / 255)
        super().__init__(config)

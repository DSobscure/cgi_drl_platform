import tensorflow as tf
import os, sys
import numpy as np
import math
import copy

class TorcsDemonstrationMemory(object):
    '''This object maintain TORCS demonstration observation action pairs
        and provide sample mini batch and data augmentation functions.
    '''
    def __init__(self, session, batch_size, config):
        self.demonstration_directory_prefix = config["demonstration_directory_prefix"]
        self.demonstration_directory = config["demonstration_directory"]
        self.is_use_internal_state = config["is_use_internal_state"]
        self.visual_observation_frame_count = config["visual_observation_frame_count"]
        self.batch_size = batch_size
        self.session = session

        observations, labels = [], []
        self.style_count = 0
        self.total_frame_count = 0
        self.total_round_count = 0

        if not os.path.isdir(self.demonstration_directory):
            print("## TorcsDemonstrationMemory: data path not a folder({})".format(self.demonstration_directory))
            exit(0)
        all_label_directory = os.listdir(self.demonstration_directory)
        for label_directory in all_label_directory:
            # label_dir will represent a kind of style
            if not os.path.isdir(os.path.join(self.demonstration_directory, label_directory)):
                continue

            self.style_count += 1
            average_time = 0.0
            average_speed = 0.0
            current_round = 0
            print("Expending {}".format(label_directory))

            for filename in os.listdir(os.path.join(self.demonstration_directory, label_directory)):
                if filename[-4:] != '.txt':
                    continue
                # extract information
                infos = filename[:-4].split('_')
                absolute_time = infos[0]
                elapsed_time = float(infos[1][7:])
                speed = float(infos[2][8:])
                average_time += elapsed_time
                average_speed += speed
                self.total_round_count += 1
                current_round += 1
                if not self.is_use_internal_state:
                    filename_queue = []

                with open(os.path.join(self.demonstration_directory, label_directory, filename)) as label_file:
                    lines = label_file.read().split('\n')
                    for line in lines:
                        fields = line.split(' ')
                        if self.is_use_internal_state:
                            ##################################################
                            # Data fields                                    #
                            # fields[0] is imgpath                           #
                            # fields[2] is label "speedX"                    #
                            # fields[3] is label "angle"                     #
                            # fields[4] is label "trackPos"                  #
                            # fields[5] is label "brake"                     #
                            # fields[6] is label "accel"                     #
                            # fields[7] is label "steer"                     #
                            ##################################################
                            if len(fields) != 7 or not os.path.isfile(fields[0]):
                                continue
                    
                            fields = [float(x) for x in fields[1:]]
                            observations.append([fields[0]/100.0]+fields[1:3])
                            labels.append([np.clip(fields[5],-1,1), np.clip(fields[4],0,1)]) # (steer, accel)
                        else:
                            ##################################################
                            # Data fields                                    #
                            # fields[0] is imgpath                           #
                            # fields[1] is label "speedX"                    #
                            # fields[2] is label "angle"                     #
                            # fields[3] is label "trackPos"                  #
                            # fields[4] is label "brake"                     #
                            # fields[5] is label "accel"                     #
                            # fields[6] is label "steer"                     #
                            ##################################################
                            if len(fields) != 7 or not os.path.isfile(self.demonstration_directory_prefix + fields[0]):
                                filename_queue.clear()
                                continue

                            filename_queue.append(self.demonstration_directory_prefix + fields[0])
                            if len(filename_queue) == self.visual_observation_frame_count:                    
                                observations.append(filename_queue[:])
                                filename_queue.pop(0)
                                fields = [float(x) for x in fields[1:]]
                                labels.append([np.clip(fields[5],-1,1), np.clip(fields[4],0,1)]) # (steer, accel)

                        
                        self.total_frame_count += 1
            
            average_time = average_time / current_round
            average_speed = average_speed / current_round
            print("    average elapsed time: {:.1f} average speed: {:.1f}".format(average_time, average_speed))
        if self.total_frame_count == 0:
            print("## TorcsDemonstrationMemory: Data not found({})".format(self.demonstration_directory))
            exit(0)

        ############################################################
        #       Build input_producer                               #
        ############################################################
        assert(len(observations) == len(labels))

        # do shuffling
        np.random.seed(0)
        shuffle_idx = np.arange(len(observations))
        np.random.shuffle(shuffle_idx)
        observations = np.asarray(observations)[shuffle_idx]
        labels = np.asarray(labels)[shuffle_idx]

        # split into train and validation set
        observations_train = observations[:]
        labels_train = labels[:]

        self.data_size = len(observations_train)

        tf_observations_train, tf_labels_train = tf.constant(observations_train), tf.constant(labels_train)
        observations_fifo_train, label_fifo_train = tf.train.slice_input_producer(
            [tf_observations_train, tf_labels_train],
            shuffle=True,
            capacity=50*batch_size
            )
        if not self.is_use_internal_state:
            observations_fifo_train = self.image_loading(observations_fifo_train)

        # make batch
        observations_batch_train, label_batch_train = tf.train.shuffle_batch(
            [observations_fifo_train, label_fifo_train],
            batch_size=batch_size,
            num_threads=2,
            capacity=10*batch_size,
            min_after_dequeue=2*batch_size)

        print("="*18 + " Data loader " + "="*19)
        print("Total styles:", self.style_count)
        print("Total rounds:", self.total_round_count)
        print("Total frames:", self.total_frame_count)
        print("="*50)

        self.observations_batch = observations_batch_train
        self.label_batch = label_batch_train

    def size(self):
        '''return the size of inner deque size'''
        return self.data_size

    def sample_mini_batch(self):
        state_batch, action_batch = self.session.run([self.observations_batch, self.label_batch])
        state_batch = np.reshape(state_batch, [-1, self.visual_observation_frame_count] + [64, 64, 3])
        return state_batch, action_batch


    def image_loading(self, fifo, noisy=True):
        all_img = fifo
        result_img = []
        for i in range(self.visual_observation_frame_count):
            img = tf.image.decode_jpeg(tf.read_file(all_img[i]))
            img = tf.image.resize_images(img, (64, 64))
            img.set_shape([64, 64, 3])
            result_img.append(img)
        return result_img


# test
if __name__ == '__main__':
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    memory = TorcsDemonstrationMemory(sess, 64, {
        "demonstration_directory_prefix" : "/root/playstyle_uai2021_demos/torcs/testing/Speed80N4/",
        "demonstration_directory" : "/root/playstyle_uai2021_demos/torcs/testing/Speed80N4/demo_data/label/",
        "is_use_internal_state" : False,
        "visual_observation_frame_count" : 4,
    })

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)


    state_batch, action_batch = memory.sample_mini_batch()
    print(action_batch)

    coord.request_stop()
    coord.join(threads)
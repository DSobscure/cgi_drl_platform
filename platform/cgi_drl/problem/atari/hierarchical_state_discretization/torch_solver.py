from cgi_drl.problem.supervised_learning_trainer import SupervisedLearningTrainer
import importlib
import numpy as np
import tensorflow as tf
import sys, os
import time

def launch(problem_config):
    load = problem_config["load_function"]
    # setup encoder
    hsd_config = load(*problem_config["hsd"])
    problem_config["hsd"] = hsd_config
    from cgi_drl.representation_model.hsd.encoder_trainer_torch import EncoderTrainer
    encoder = EncoderTrainer(hsd_config)

    # setup demo
    training_demo_config = load(*problem_config["training_demo"])
    problem_config["training_demo"] = training_demo_config
    from cgi_drl.data_storage.demonstration_memory.atari_demonstration_memory import AtariDemonstrationMemory
    demonstration_memory = AtariDemonstrationMemory(training_demo_config)

    # setup environment (simple version)
    evaluation_env_config = load(*problem_config["evaluation_environment"])
    problem_config["evaluation_environment"] = evaluation_env_config
    from cgi_drl.environment.atari.atari_environment_wrapper import AtariEnvironmentWrapper
    eval_env = AtariEnvironmentWrapper(evaluation_env_config)

    # setup observation preprocessor
    preprocessor_config = load(*problem_config["observation_preprocessor"])
    problem_config["observation_preprocessor"] = preprocessor_config
    from cgi_drl.environment.atari.atari_observation_preprocessor import AtariObservationPreprocessor
    preprocessor = AtariObservationPreprocessor(preprocessor_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = AtariHsdSolver(problem_config)
    solver.encoder = encoder
    solver.demonstration_memory = demonstration_memory
    solver.eval_env = eval_env
    solver.eval_env_preprocessor = preprocessor

    solver.train()

class AtariHsdSolver(SupervisedLearningTrainer):
    def initialize(self):
        self.summary_writer = tf.summary.FileWriter(self.log_path)
        self.log_file = open(self.log_path + "/log.txt", 'a', 1)
        self.batch_size = self.solver_config["batch_size"]

        if "initial_learning_rate" in self.solver_config:
            self.initial_learning_rate = self.solver_config["initial_learning_rate"]
        elif "fixed_learning_rate" in self.solver_config:
            self.fixed_learning_rate = self.solver_config["fixed_learning_rate"]

        if self.solver_config["is_load_encoder"]:
            self.encoder.load(path=self.model_path)

        print("=" * 18 + " Setup model " + "=" * 19)
        print("mode: training with HSD")
        print("=" * 50)

        self.update_times = self.demonstration_memory.size() // self.batch_size
        self.evaluation_episode_count = self.solver_config["evaluation_episode_count"]

    def terminate(self):
        pass

    def on_epoch(self):
        self.update()
        self.evaluate()

    def update(self):
        total_policy_loss = 0
        total_compression_loss = 0
        total_vq_loss = 0
        total_vq_mean_loss = 0
        start_train_time = time.time()

        if "initial_learning_rate" in self.solver_config:
            learning_rate = self.initial_learning_rate * (1 - self.epoch_number/(self.solver_config["end_epoch_number"] + 1 - self.solver_config["start_epoch_number"]))
        elif "fixed_learning_rate" in self.solver_config:
            learning_rate = self.fixed_learning_rate

        i_counter = 0
        for observation_batch, action_batch in self.demonstration_memory.random_sample_all_batch(self.batch_size):
            policy_loss, compression_loss, vq_loss, vq_mean_loss = self.encoder.update(
                observations=observation_batch,
                compressed_observations=observation_batch, 
                actions=action_batch, 
                batch_size=len(action_batch), 
                learning_rate=learning_rate
            )
            total_policy_loss += policy_loss
            total_compression_loss += compression_loss
            total_vq_loss += vq_loss
            total_vq_mean_loss += vq_mean_loss
            i_counter += 1
            print("update:{}/{}, policy_loss:{:.5f}, compression_loss:{:.5f}, vq_loss:{:.5f}, vq_mean_loss:{:.5f}".format(i_counter, self.update_times, policy_loss, compression_loss, vq_loss, vq_mean_loss), end='\r')

        total_policy_loss = total_policy_loss / self.update_times
        total_compression_loss = total_compression_loss / self.update_times
        total_vq_loss = total_vq_loss / self.update_times
        total_vq_mean_loss = total_vq_mean_loss / self.update_times
        print()
        traning_time = time.time() - start_train_time
        log_str = "Epoch {}, traning_time: {}, policy_loss: {}, compression_loss:{}, vq_loss: {}, vq_mean_loss: {}\n".format(self.epoch_number, traning_time, total_policy_loss, total_compression_loss, total_vq_loss, total_vq_mean_loss)

        each_iter_summary = tf.Summary(value=[
            tf.Summary.Value(tag="Traning Time", simple_value=traning_time),
            tf.Summary.Value(tag="Policy Loss", simple_value=total_policy_loss),
            tf.Summary.Value(tag="Compression Loss", simple_value=total_compression_loss),
            tf.Summary.Value(tag="VQ Loss", simple_value=total_vq_loss),
            tf.Summary.Value(tag="VQ Mean Loss", simple_value=total_vq_mean_loss),
        ])
        print(log_str, end='')
        self.log_file.write(log_str)
        self.summary_writer.add_summary(each_iter_summary, self.epoch_number)
        self.encoder.save(self.model_path, self.epoch_number)

    def evaluate(self):
        # dataset evaluation
        sample_count = 0
        base_accuracy_sum = 0
        hierarchy_accuracy_sum = 0

        for observation_batch, action_batch in self.demonstration_memory.sample_all_batch(self.batch_size):
            batch_size = len(action_batch)
            _, base_policy, _ = self.encoder.get_compressed_observation_and_policy(observations=observation_batch, hierarchy_usages=np.ones([batch_size, 1]))
            _, hierarchy_policy, _ = self.encoder.get_compressed_observation_and_policy(observations=observation_batch, hierarchy_usages=np.zeros([batch_size, 1]))
            sample_count += batch_size
            for i_batch in range(batch_size):
                if np.argmax(action_batch) == np.argmax(base_policy[i_batch]):
                    base_accuracy_sum += 1
                if np.argmax(action_batch) == np.argmax(hierarchy_policy[i_batch]):
                    hierarchy_accuracy_sum += 1
        base_accuracy = base_accuracy_sum / sample_count
        hierarchy_accuracy = hierarchy_accuracy_sum / sample_count
        log_str = "Epoch {}, base_accuracy: {}, hierarchy_accuracy: {}\n".format(self.epoch_number, base_accuracy, hierarchy_accuracy)

        each_iter_summary = tf.Summary(value=[
            tf.Summary.Value(tag="Base Accuracy", simple_value=base_accuracy),
            tf.Summary.Value(tag="Hierarchy Accuracy", simple_value=hierarchy_accuracy),
        ])
        print()
        print(log_str, end='')
        self.log_file.write(log_str)
        self.summary_writer.add_summary(each_iter_summary, self.epoch_number)

        # online policy evaluation
        self.scores = []
        self.best_episode_raw_observations = None
        self.best_episode_decoded_base_observations = None
        self.best_episode_decoded_hierarchy_observations = None
        
        for i in range(self.evaluation_episode_count):
            self.episode_initiate()
            while True:
                done = self.on_time_step()
                if done:
                    self.episode_terminate()
                    break
        self.eval_env_preprocessor.create_video({"observation_2d":self.best_episode_raw_observations}, self.video_path + '/epoch{}_best_score_{}_raw.mp4'.format(self.epoch_number, max(self.scores)))
        self.eval_env_preprocessor.create_video({"observation_2d":self.best_episode_decoded_base_observations}, self.video_path + '/epoch{}_best_score_{}_base.mp4'.format(self.epoch_number, max(self.scores)))
        self.eval_env_preprocessor.create_video({"observation_2d":self.best_episode_decoded_hierarchy_observations}, self.video_path + '/epoch{}_best_score_{}_hierarchy.mp4'.format(self.epoch_number, max(self.scores)))
            

    def episode_initiate(self):
        self.episode_score = 0
        self.observation = self.eval_env_preprocessor.process(self.eval_env.reset()[0])
        for key in self.observation:
            self.observation[key] = self.eval_env_preprocessor.observation_aggregator(key, None, self.observation[key], True)
        self.episode_raw_observations = []
        self.episode_decoded_base_observations = []
        self.episode_decoded_hierarchy_observations = []

    def episode_terminate(self):
        if len(self.scores) == 0 or self.episode_score > max(self.scores):
            self.best_episode_raw_observations = self.episode_raw_observations
            self.best_episode_decoded_base_observations = self.episode_decoded_base_observations
        self.best_episode_decoded_hierarchy_observations = self.episode_decoded_hierarchy_observations
        self.scores.append(self.episode_score)
        print("Epoch {}, score: {}".format(self.epoch_number, self.episode_score))

    def on_time_step(self):
        base_reconstructed_observation, base_policy, base_actions = self.encoder.get_compressed_observation_and_policy(self.observation, np.ones([1, 1]))
        hierarchy_reconstructed_observation, hierarchy_policy, hierarchy_actions = self.encoder.get_compressed_observation_and_policy(self.observation, np.zeros([1, 1]))

        next_observation, rewards, dones, infos = self.eval_env.step([base_actions[0]]) 
        self.next_observation = self.eval_env_preprocessor.process(next_observation[0])
        for key in self.next_observation:
            self.next_observation[key] = self.eval_env_preprocessor.observation_aggregator(key, self.observation[key], self.next_observation[key])
        self.episode_score += rewards[0]

        self.episode_raw_observations.append(self.observation["observation_2d"])
        self.episode_decoded_base_observations.append(base_reconstructed_observation["observation_2d"][0])
        self.episode_decoded_hierarchy_observations.append(hierarchy_reconstructed_observation["observation_2d"][0])

        self.observation = self.next_observation
        return dones[0]

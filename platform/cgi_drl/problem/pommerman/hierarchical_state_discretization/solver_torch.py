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
    from cgi_drl.data_storage.demonstration_memory.simple_demonstration_memory import SimpleDemonstrationMemory
    demonstration_memory = SimpleDemonstrationMemory(training_demo_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = PommermanHsdSolver(problem_config)
    solver.encoder = encoder
    solver.demonstration_memory = demonstration_memory

    solver.train()

class PommermanHsdSolver(SupervisedLearningTrainer):
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
        for batches in self.demonstration_memory.random_sample_all_batch(self.batch_size):
            policy_loss, compression_loss, vq_loss, vq_mean_loss = self.encoder.update(
                observations={
                    "observation_visual" : batches["observation_visual"],
                    "observation_vector" : batches["observation_vector"]
                },
                compressed_observations={
                    "observation_visual" : batches["observation_visual"],
                    "observation_vector" : batches["observation_vector"]
                }, 
                actions=batches["action"], 
                batch_size=len(batches["action"]), 
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
        base_state_set = set()
        hierarchy_state_set = set()

        i_counter = 0
        for batches in self.demonstration_memory.sample_all_batch(self.batch_size):
            batch_size = len(batches["action"])
            _, base_policy, _ = self.encoder.get_compressed_observation_and_policy(
                observations={
                    "observation_visual" : batches["observation_visual"],
                    "observation_vector" : batches["observation_vector"]
                }, hierarchy_usages=np.ones([batch_size, 1])
            )
            _, hierarchy_policy, _ = self.encoder.get_compressed_observation_and_policy(
                observations={
                    "observation_visual" : batches["observation_visual"],
                    "observation_vector" : batches["observation_vector"]
                }, hierarchy_usages=np.zeros([batch_size, 1])
            )
            base_state, hierarchy_state = self.encoder.get_discrite_states(
                observations={
                    "observation_visual" : batches["observation_visual"],
                    "observation_vector" : batches["observation_vector"]
                }
            )
            sample_count += batch_size
            for i_batch in range(batch_size):
                if batches["action"][i_batch][0] == np.argmax(base_policy[i_batch]):
                    base_accuracy_sum += 1
                if batches["action"][i_batch][0] == np.argmax(hierarchy_policy[i_batch]):
                    hierarchy_accuracy_sum += 1
                base_state_set.add(",".join(map(str,base_state[i_batch])))
                hierarchy_state_set.add(",".join(map(str,hierarchy_state[i_batch])))
            i_counter += 1
            print("evaluate:{}/{}".format(i_counter, self.update_times), end='\r')
        base_accuracy = base_accuracy_sum / sample_count
        hierarchy_accuracy = hierarchy_accuracy_sum / sample_count
        log_str = "Epoch {}, base_accuracy: {}, hierarchy_accuracy: {}, base_size: {}, hierarchy_size: {}\n".format(self.epoch_number, base_accuracy, hierarchy_accuracy, len(base_state_set), len(hierarchy_state_set))

        each_iter_summary = tf.Summary(value=[
            tf.Summary.Value(tag="Base Accuracy", simple_value=base_accuracy),
            tf.Summary.Value(tag="Hierarchy Accuracy", simple_value=hierarchy_accuracy),
            tf.Summary.Value(tag="Base Used State", simple_value=len(base_state_set)),
            tf.Summary.Value(tag="Hierarchy Used State", simple_value=len(hierarchy_state_set)),
        ])
        print()
        print(log_str, end='')
        self.log_file.write(log_str)
        self.summary_writer.add_summary(each_iter_summary, self.epoch_number)

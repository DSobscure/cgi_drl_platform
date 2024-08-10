from cgi_drl.problem.supervised_learning_trainer import SupervisedLearningTrainer
import numpy as np
import time
from tensorboardX import SummaryWriter

def launch(problem_config):
    load = problem_config["load_function"]
    # setup hsd
    hsd_config = load(*problem_config["hsd"])
    problem_config["hsd"] = hsd_config
    from cgi_drl.representation_model.hsd.encoder_trainer_torch import EncoderTrainer
    model = EncoderTrainer(hsd_config)

    # setup data_loader
    data_loader_config = load(*problem_config["data_loader"])
    problem_config["data_loader"] = data_loader_config
    from cgi_drl.data_storage.demonstration_memory.minizero_data_loader import MinizeroDadaLoader
    data_loader = MinizeroDadaLoader(data_loader_config)
    
    # setup solver
    problem_config["solver"] = problem_config
    solver = GoHsdSolver(problem_config)
    solver.model = model
    solver.data_loader = data_loader

    solver.train()

class GoHsdSolver(SupervisedLearningTrainer):
    def initialize(self):
        self.summary_writer = SummaryWriter(self.log_path)
        self.log_file = open(self.log_path + "/log.txt", 'a', 1)

        if "initial_learning_rate" in self.solver_config:
            self.initial_learning_rate = self.solver_config["initial_learning_rate"]
        elif "fixed_learning_rate" in self.solver_config:
            self.fixed_learning_rate = self.solver_config["fixed_learning_rate"]

        print("=" * 18 + " Setup model " + "=" * 19)
        print("mode: training with HSD")
        print("=" * 50)

        self.update_times = self.solver_config["update_steps_per_epoch"]

    def terminate(self):
        pass

    def on_epoch(self):
        total_policy_loss = 0
        total_value_loss = 0
        total_vq_loss = 0
        total_vq_mean_loss = 0
        total_correct_count = 0
        total_sample_count = 0
        total_used_state_count = np.zeros(self.model.hierarchy_count)
        start_train_time = time.time()

        if "initial_learning_rate" in self.solver_config:
            learning_rate = self.initial_learning_rate * (1 - self.epoch_number/(self.solver_config["end_epoch_number"] + 1 - self.solver_config["start_epoch_number"]))
        elif "fixed_learning_rate" in self.solver_config:
            learning_rate = self.fixed_learning_rate

        for i in range(self.update_times):
            features, _, label_policy, label_value, _, loss_scale, _ = self.data_loader.sample_data(self.model.device)
            policy_loss, value_loss, vq_loss, vq_mean_loss, correct_count, sample_count, used_state_count = self.model.update(features, label_policy, label_value, loss_scale, learning_rate)

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_vq_loss += vq_loss
            total_vq_mean_loss += vq_mean_loss
            total_correct_count += correct_count
            total_sample_count += sample_count
            total_used_state_count += used_state_count
            print("update:{}/{}, policy:{:.4f}, value:{:.4f}, vq:{:.4f}, vq_mean:{:.4f}, acc:{:.4f}, state:{}".format(
                i, self.update_times, policy_loss, value_loss, vq_loss, vq_mean_loss, correct_count / sample_count, used_state_count), end='\r'
            )

        total_policy_loss /= self.update_times
        total_value_loss /= self.update_times
        total_vq_loss /= self.update_times
        total_vq_mean_loss /= self.update_times
        total_used_state_count /= self.update_times
        print()
        traning_time = time.time() - start_train_time
        log_str = "Epoch {}, traning_time: {}, policy: {}, value:{}, vq: {}, vq_mean: {}, acc: {}, state: {}\n".format(
            self.epoch_number, traning_time, total_policy_loss, total_value_loss, total_vq_loss, total_vq_mean_loss, total_correct_count/ total_sample_count, total_used_state_count
        )

        self.summary_writer.add_scalar('Traning Time', traning_time, self.epoch_number)
        self.summary_writer.add_scalar('Policy Loss', total_policy_loss, self.epoch_number)
        self.summary_writer.add_scalar('Value Loss', total_value_loss, self.epoch_number)
        self.summary_writer.add_scalar('VQ Loss', total_vq_loss, self.epoch_number)
        self.summary_writer.add_scalar('VQ Mean Loss', total_vq_mean_loss, self.epoch_number)
        self.summary_writer.add_scalar('Accuracy', total_correct_count/ total_sample_count, self.epoch_number)
        for i_hierarchy in range(self.model.hierarchy_count):
            self.summary_writer.add_scalar('H{} Used State Count'.format(i_hierarchy), total_used_state_count[i_hierarchy], self.epoch_number)
        print(log_str, end='')
        self.log_file.write(log_str)
        self.model.save(self.model_path, self.epoch_number)


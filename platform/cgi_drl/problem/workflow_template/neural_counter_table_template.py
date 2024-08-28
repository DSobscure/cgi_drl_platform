from cgi_drl.problem.supervised_learning_trainer import SupervisedLearningTrainer
import tensorflow as tf

class NeuralCounterTableSolver(SupervisedLearningTrainer):
    def initialize(self):
        self.summary_writer = tf.summary.FileWriter(self.log_path)
        self.log_file = open(self.log_path + "/log.txt", 'a', 1)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()
        self.model.set_session(self.sess)
        self.bt_model.set_session(self.sess)

        self.batch_size = self.solver_config["batch_size"]

        if "initial_learning_rate" in self.solver_config:
            self.initial_learning_rate = self.solver_config["initial_learning_rate"]
        elif "fixed_learning_rate" in self.solver_config:
            self.fixed_learning_rate = self.solver_config["fixed_learning_rate"]

        self.bt_model_path = self.solver_config["bt_model_path"]

        self.sess.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ])

        self.bt_model.load(self.bt_model_path)

        print("=" * 18 + " Setup model " + "=" * 19)
        print("mode: training")
        print("=" * 50)

    def terminate(self):
        self.model.save(self.model_path, self.epoch_number)

    def update(self):
        total_winvalue_prediction_loss = 0
        total_table_loss = 0
        total_vq_loss = 0
        total_vq_mean_loss = 0

        if "initial_learning_rate" in self.solver_config:
            extra_settings = {
                "learning_rate" : self.initial_learning_rate * (1 - self.epoch_number/(self.solver_config["end_epoch_number"] + 1 - self.solver_config["start_epoch_number"]))
            }
        elif "fixed_learning_rate" in self.solver_config:
            extra_settings = {
                "learning_rate" : self.fixed_learning_rate
            }

        i = 0
        update_times = self.result_buffer_train.size() // self.batch_size
        for records in self.result_buffer_train.random_sample_all_batch(self.batch_size):
            bt_winvalue_prediction = self.bt_model.get_predictions({"player1_combo":records["player1_combos"], "player2_combo":records["player2_combos"]})
            winvalue_prediction_loss, vq_loss, vq_mean_loss = self.model.update(
                {"player1_combo":records["player1_combos"], "player2_combo":records["player2_combos"]},
                records["match_results"], 
                bt_winvalue_prediction,
                extra_settings
            )
            total_winvalue_prediction_loss += winvalue_prediction_loss
            total_vq_loss += vq_loss
            total_vq_mean_loss += vq_mean_loss
            print("update:{}/{}, prediction_loss:{:.5f}, vq_loss:{:.5f}, vq_mean_loss:{:.5f}".format(i, update_times, winvalue_prediction_loss, vq_loss, vq_mean_loss), end='\r')
            i += 1

        total_winvalue_prediction_loss = total_winvalue_prediction_loss / i
        total_table_loss = total_table_loss / i
        total_vq_loss = total_vq_loss / i
        total_vq_mean_loss = total_vq_mean_loss / i
        print()

        log_str = "Epoch {}, prediction_loss: {}, vq_loss: {}, vq_mean_loss: {}\n".format(self.epoch_number, total_winvalue_prediction_loss, total_vq_loss, total_vq_mean_loss)

        each_iter_summary = tf.Summary(value=[
            tf.Summary.Value(tag="Win Value Prediction Loss", simple_value=total_winvalue_prediction_loss),
            tf.Summary.Value(tag="VQ Loss", simple_value=total_vq_loss),
            tf.Summary.Value(tag="VQ Mean Loss", simple_value=total_vq_mean_loss),
        ])
        print(log_str, end='')
        self.log_file.write(log_str)
        self.summary_writer.add_summary(each_iter_summary, self.epoch_number)
        # self.model.save(self.model_path, self.epoch_number)

    def eval(self):
        winvalue_accurate_count = 0
        table_winvalue_accurate_count = 0
        sample_counter = 0
        comp_set = set()

        for records in self.result_buffer_train.sample_all_batch(self.batch_size):
            bt_winvalue_prediction = self.bt_model.get_predictions({"player1_combo":records["player1_combos"], "player2_combo":records["player2_combos"]})
            winvalue_prediction, player1_embedding_k, player2_embedding_k = self.model.get_predictions({"player1_combo":records["player1_combos"], "player2_combo":records["player2_combos"]}, bt_winvalue_prediction)
            comp_set.update(player1_embedding_k)
            comp_set.update(player2_embedding_k)
            sample_count = len(records["match_results"])
            sample_counter += sample_count
            for i in range(sample_count):
                match_result = records["match_results"][i]
                if match_result > 0.501: # win
                    if winvalue_prediction[i] > 0.501:
                        winvalue_accurate_count += 1
                elif match_result < 0.499: # lose
                    if winvalue_prediction[i] < 0.499:
                        winvalue_accurate_count += 1
                else: # tie
                    if winvalue_prediction[i] >= 0.499 and winvalue_prediction[i] <= 0.501:
                        winvalue_accurate_count += 1
        
        winvalue_accuracy = winvalue_accurate_count / sample_counter
        table_winvalue_accuracy = table_winvalue_accurate_count / sample_counter

        log_str = "Epoch {}, train - accuracy: {}, comp count: {}\n".format(self.epoch_number, winvalue_accuracy, len(comp_set))

        each_iter_summary = tf.Summary(value=[
            tf.Summary.Value(tag="Train/Win Value Accuracy", simple_value=winvalue_accuracy),
        ])
        print()
        print(log_str, end='')
        self.log_file.write(log_str)
        self.summary_writer.add_summary(each_iter_summary, self.epoch_number)

        winvalue_accurate_count = 0
        table_winvalue_accurate_count = 0
        sample_counter = 0

        for records in self.result_buffer_test.sample_all_batch(self.batch_size):
            bt_winvalue_prediction = self.bt_model.get_predictions({"player1_combo":records["player1_combos"], "player2_combo":records["player2_combos"]})
            winvalue_prediction, player1_embedding_k, player2_embedding_k = self.model.get_predictions({"player1_combo":records["player1_combos"], "player2_combo":records["player2_combos"]}, bt_winvalue_prediction)
            sample_count = len(records["match_results"])
            sample_counter += sample_count
            for i in range(sample_count):
                match_result = records["match_results"][i]
                if match_result > 0.501: # win
                    if winvalue_prediction[i] > 0.501:
                        winvalue_accurate_count += 1
                elif match_result < 0.499: # lose
                    if winvalue_prediction[i] < 0.499:
                        winvalue_accurate_count += 1
                else: # tie
                    if winvalue_prediction[i] >= 0.499 and winvalue_prediction[i] <= 0.501:
                        winvalue_accurate_count += 1
        
        winvalue_accuracy = winvalue_accurate_count / sample_counter

        log_str = "Epoch {}, test - accuracy: {}\n".format(self.epoch_number, winvalue_accuracy)

        each_iter_summary = tf.Summary(value=[
            tf.Summary.Value(tag="Test/Win Value Accuracy", simple_value=winvalue_accuracy),
        ])
        print()
        print(log_str, end='')
        self.log_file.write(log_str)
        self.summary_writer.add_summary(each_iter_summary, self.epoch_number)

    def on_epoch(self):
        self.update()
        self.eval()

class NeuralCounterTableTorchSolver(NeuralCounterTableSolver):
    def initialize(self):
        self.summary_writer = tf.summary.FileWriter(self.log_path)
        self.log_file = open(self.log_path + "/log.txt", 'a', 1)

        self.batch_size = self.solver_config["batch_size"]

        if "initial_learning_rate" in self.solver_config:
            self.initial_learning_rate = self.solver_config["initial_learning_rate"]
        elif "fixed_learning_rate" in self.solver_config:
            self.fixed_learning_rate = self.solver_config["fixed_learning_rate"]

        self.bt_model_path = self.solver_config["bt_model_path"]
        self.bt_model.load(self.bt_model_path)

        print("=" * 18 + " Setup model " + "=" * 19)
        print("mode: training")
        print("=" * 50)
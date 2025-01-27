from cgi_drl.problem.supervised_learning_trainer import SupervisedLearningTrainer
import tensorflow as tf

class EloRccSolver(SupervisedLearningTrainer):
    def initialize(self):
        self.summary_writer = tf.summary.FileWriter(self.log_path)
        self.log_file = open(self.log_path + "/log.txt", 'a', 1)
        self.model.set_session(None)

        self.batch_size = self.solver_config["batch_size"]

        print("=" * 18 + " Setup model " + "=" * 19)
        print("mode: training")
        print("=" * 50)

    def terminate(self):
        pass

    def update(self):
        total_winvalue_prediction_loss = 0
        total_residual_prediction_loss = 0
        total_cross_entropy_loss = 0

        i = 0
        update_times = self.result_buffer_train.size() // self.batch_size
        for records in self.result_buffer_train.random_sample_all_batch(self.batch_size):
            winvalue_prediction_loss, residual_prediction_loss, cross_entropy_loss = self.model.update(
                {"player1_combo":records["player1_combos"], "player2_combo":records["player2_combos"]},
                records["match_results"], 
                None
            )
            total_winvalue_prediction_loss += winvalue_prediction_loss
            total_residual_prediction_loss += residual_prediction_loss
            total_cross_entropy_loss += cross_entropy_loss
            print("update:{}/{}, prediction_loss:{:.5f}, residual_prediction_loss:{:.5f}, cross_entropy_loss:{:.5f}".format(i, update_times, winvalue_prediction_loss, residual_prediction_loss, cross_entropy_loss), end='\r')
            i += 1

        total_winvalue_prediction_loss = total_winvalue_prediction_loss / i
        total_residual_prediction_loss = total_residual_prediction_loss / i
        total_cross_entropy_loss = total_cross_entropy_loss / i
        print()

        log_str = "Epoch {}, prediction_loss: {}, residual_prediction_loss: {}, cross_entropy_loss: {}\n".format(self.epoch_number, total_winvalue_prediction_loss, total_residual_prediction_loss, total_cross_entropy_loss)

        each_iter_summary = tf.Summary(value=[
            tf.Summary.Value(tag="Win Value Prediction Loss", simple_value=total_winvalue_prediction_loss),
            tf.Summary.Value(tag="Win Value Residual Prediction Loss", simple_value=total_residual_prediction_loss),
            tf.Summary.Value(tag="Category Cross Entropy Loss", simple_value=total_cross_entropy_loss),
        ])
        print(log_str, end='')
        self.log_file.write(log_str)
        self.summary_writer.add_summary(each_iter_summary, self.epoch_number)

    def eval(self):
        # print(self.model.expected_residual_table)
        # print(self.model.rating_table)
        winvalue_accurate_count = 0
        sample_counter = 0

        for records in self.result_buffer_train.sample_all_batch(self.batch_size):
            winvalue_prediction = self.model.get_predictions({"player1_combo":records["player1_combos"], "player2_combo":records["player2_combos"]})
            sample_count = len(records["match_results"])
            sample_counter += sample_count
            for i in range(sample_count):
                match_result = records["match_results"][i]
                if match_result > 0.501: # win
                    if winvalue_prediction[i] >= 0 and winvalue_prediction[i] > 0.501:
                        winvalue_accurate_count += 1
                elif match_result < 0.499: # lose
                    if winvalue_prediction[i] >= 0 and winvalue_prediction[i] < 0.499:
                        winvalue_accurate_count += 1
                else: # tie
                    if winvalue_prediction[i] >= 0 and winvalue_prediction[i] >= 0.499 and winvalue_prediction[i] <= 0.501:
                        winvalue_accurate_count += 1
        
        winvalue_accuracy = winvalue_accurate_count / sample_counter

        log_str = "Epoch {}, train - accuracy: {}\n".format(self.epoch_number, winvalue_accuracy)

        each_iter_summary = tf.Summary(value=[
            tf.Summary.Value(tag="Train/Win Value Accuracy", simple_value=winvalue_accuracy),
        ])
        print()
        print(log_str, end='')
        self.log_file.write(log_str)
        self.summary_writer.add_summary(each_iter_summary, self.epoch_number)

        winvalue_accurate_count = 0
        sample_counter = 0

        for records in self.result_buffer_test.sample_all_batch(self.batch_size):
            winvalue_prediction = self.model.get_predictions({"player1_combo":records["player1_combos"], "player2_combo":records["player2_combos"]})
            sample_count = len(records["match_results"])
            sample_counter += sample_count
            for i in range(sample_count):
                match_result = records["match_results"][i]
                if match_result > 0.501: # win
                    if winvalue_prediction[i] >= 0 and winvalue_prediction[i] > 0.501:
                        winvalue_accurate_count += 1
                elif match_result < 0.499: # lose
                    if winvalue_prediction[i] >= 0 and winvalue_prediction[i] < 0.499:
                        winvalue_accurate_count += 1
                else: # tie
                    if winvalue_prediction[i] >= 0 and winvalue_prediction[i] >= 0.499 and winvalue_prediction[i] <= 0.501:
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
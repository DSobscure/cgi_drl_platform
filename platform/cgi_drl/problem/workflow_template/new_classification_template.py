from cgi_drl.problem.supervised_learning_trainer import SupervisedLearningTrainer
import numpy as np
import tensorflow as tf
import time

class NewClassificationTorchTrainer(SupervisedLearningTrainer):
    def initialize(self):
        self.summary_writer = tf.summary.FileWriter(self.log_path)
        self.log_file = open(self.log_path + "/log.txt", 'a', 1)

        self.batch_size = self.solver_config["batch_size"]
        self.learning_rate_scheduler = self.solver_config["learning_rate_scheduler"]

        print("=" * 18 + " Setup model " + "=" * 19)
        print("mode: training")
        print("=" * 50)

    def terminate(self):
        self.model.save(self.model_path, self.epoch_number)

    def update(self):
        total_prediction_loss, total_ensemble_loss = 0, 0
        start_train_time = time.time()

        learning_rate = self.learning_rate_scheduler(self.epoch_number)

        epoch_steps = self.dataset.size() // self.batch_size
        for i in range(epoch_steps):
            image_batch, label_batch, _ = self.dataset.sample_training_mini_batch(self.batch_size)
            prediction_loss, ensemble_loss = self.model.update(image_batch, label_batch, learning_rate)
            total_prediction_loss += prediction_loss
            total_ensemble_loss += ensemble_loss
            print("update:{}/{}, prediction_loss:{}, ensemble_loss: {}".format(i, epoch_steps, prediction_loss, ensemble_loss), end='\r')

        total_prediction_loss = total_prediction_loss / epoch_steps
        total_ensemble_loss = total_ensemble_loss / epoch_steps
        print()
        traning_time = time.time() - start_train_time
        log_str = "Epoch {}, traning_time: {}, prediction_loss:{}, ensemble_loss: {}\n".format(self.epoch_number, traning_time, total_prediction_loss, total_ensemble_loss)

        each_iter_summary = tf.Summary(value=[
            tf.Summary.Value(tag="TraningTime", simple_value=traning_time),
            tf.Summary.Value(tag="Prediction Loss", simple_value=total_prediction_loss),
            tf.Summary.Value(tag="Ensemble Loss", simple_value=total_prediction_loss),
        ])
        print(log_str, end='')
        self.log_file.write(log_str)
        self.summary_writer.add_summary(each_iter_summary, self.epoch_number)

    def eval(self):
        training_accuracy, testing_accuracy = 0.0, 0.0 
        training_accuracy_best, testing_accuracy_best = 0.0, 0.0 
        training_accuracy_ensemble, testing_accuracy_ensemble = 0.0, 0.0 
        
        for i in range(1,501):
            images, labels = self.dataset.get_training_mini_batch(i, 100)
            _accuracy, _accuracy_best, _accuracy_ensemble = self.model.get_accuracy(images, labels)
            training_accuracy += _accuracy
            training_accuracy_best += _accuracy_best
            training_accuracy_ensemble += _accuracy_ensemble
        training_accuracy /= 500
        training_accuracy_best /= 500
        training_accuracy_ensemble /= 500
        
        for i in range(1,101):
            images, labels = self.dataset.get_testing_mini_batch(i, 100)
            _accuracy, _accuracy_best, _accuracy_ensemble = self.model.get_accuracy(images, labels)
            testing_accuracy += _accuracy
            testing_accuracy_best += _accuracy_best
            testing_accuracy_ensemble += _accuracy_ensemble
        testing_accuracy /= 100
        testing_accuracy_best /= 100
        testing_accuracy_ensemble /= 100
        
        log_str = "{}/{} Training Accuracy: {}/{}/{}, Testing Accuracy: {}/{}/{}\n".format(
            self.epoch_number, self.solver_config["end_epoch_number"],
            training_accuracy, training_accuracy_best, training_accuracy_ensemble, testing_accuracy, testing_accuracy_best, testing_accuracy_ensemble)
        summary = [
            tf.Summary.Value(tag="Training Accuracy", simple_value=training_accuracy),
            tf.Summary.Value(tag="Testing Accuracy", simple_value=testing_accuracy),
            tf.Summary.Value(tag="Training Specific Accuracy", simple_value=training_accuracy_best),
            tf.Summary.Value(tag="Testing Specific Accuracy", simple_value=testing_accuracy_best),
            tf.Summary.Value(tag="Training Ensemble Accuracy", simple_value=training_accuracy_ensemble),
            tf.Summary.Value(tag="Testing Ensemble Accuracy", simple_value=testing_accuracy_ensemble),
        ]
        each_iter_summary = tf.Summary(value=summary)
        print(log_str, end='')
        self.log_file.write(log_str)
        self.summary_writer.add_summary(each_iter_summary, self.epoch_number)

    def on_epoch(self):
        self.update()
        self.eval()

        
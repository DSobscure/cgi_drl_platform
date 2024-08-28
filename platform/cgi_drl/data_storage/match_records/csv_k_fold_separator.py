import csv
import os
import numpy as np
from sklearn.model_selection import KFold

class CsvKFoldSeparator(object):
    def __init__(self, config):
        self.dataset_path = config["dataset_path"]
        self.k = config["k"]
        self.header = []

    def read_csv(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            self.header = next(reader)
            return list(reader)

    def separate(self):
        data = self.read_csv()
        kf = KFold(n_splits=self.k, shuffle=True)

        base_path, file_name = os.path.split(self.dataset_path)
        file_prefix, _ = os.path.splitext(file_name)

        fold = 1
        for train_index, test_index in kf.split(data):
            train_data = [self.header] + [data[i] for i in train_index]
            test_data = [self.header] + [data[i] for i in test_index]

            train_file = os.path.join(base_path, f"{file_prefix}_{fold}_train.csv")
            test_file = os.path.join(base_path, f"{file_prefix}_{fold}_test.csv")

            with open(train_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(train_data)

            with open(test_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(test_data)
            print("End ", fold)
            fold += 1

if __name__ == '__main__':
    replay = CsvKFoldSeparator({
        "dataset_path" : "/root/balance/Hearthstone/hearthstone_ranking_gold.csv",
        "k" : 5
    })
    replay.separate()

import pandas as pd
import time
import json
from itertools import product
from collections import namedtuple
from collections import OrderedDict

class RunBuilder():
    # generate multiple runs with varying parameters,
    # each set of parameters will be used in each run, until all combinations are covered
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


class RunManager():
    # start, manage and end each run
    # tracking important data - running time, loss results, with option to save/print the results
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        # self.epoch_num_correct = 0 # for classification problems
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None


    def begin_run(self, run, network, loader):
        # start run with set of parameters
        self.run_params = run
        self.run_start_time = time.time()
        self.run_count += 1

    def end_run(self):
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        # self.epoch_num_correct = 0 # for classification problems

    def end_epoch(self): #saving epoch results
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        loss = self.epoch_loss

        #tracking results:
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results['train_loss'] = self.train_loss
        results['test_loss'] = self.test_loss

        # results["accuracy"] = accuracy # for classification problems
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

    def track_loss(self, loss):
        self.epoch_loss = loss.item()

    def track_train_loss(self, loss):
        self.train_loss = loss.item()

    def track_test_loss(self, loss):
        self.test_loss = loss.item()

    ## for clasification problems
    # def track_num_correct(self, preds, labels):
    #     self.epoch_num_correct += self._get_num_correct(preds, labels)
    #
    # @torch.no_grad()
    # def _get_num_correct(self, preds, labels):
    #     return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):
        #save to csv
        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'{fileName}.csv')
        #save to json
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

    def print_results(self):
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        print("results - train and test loss calculated in each 10 epochs")
        print(df.to_string()) #print full table
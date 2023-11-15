import random
import numpy as np
from collections import defaultdict
import time
import os
import json
import copy

from nlp.bert.model import MultiLabelBertBase
from nlp.bert.data import BertDataset
from nlp.bert.training import get_model_config
from nlp.bert.utils import set_seeds, classification, print_results


class ActiveLearning:
    def __init__(self, iterations=5, sample_size=100, strategy='random',
                 warm_start=800, dataset_size=3233, test_size=300, language='dutch',
                 motivations=None, training_config=None, save_results=False, verbose=True):
        seed_val = 42
        set_seeds(seed_val)

        self.warm_start    = warm_start
        self.iterations    = iterations
        self.sample_size   = sample_size
        self.dataset_size  = dataset_size
        self.test_size     = test_size
        self.strategy_name = strategy
        self.save_results  = save_results
        self.verbose       = verbose

        self.select_strategy()

        self.train_config = get_model_config(language, training_config)

        self.full_dataset = BertDataset(motivations=motivations, shuffle=True,
                            label_names=self.train_config.get('label_names'),
                            max_size=dataset_size, language=self.train_config.get('language'),
                            model_name=self.train_config.get('model_name'),
                            max_seq_length=self.train_config.get('max_seq_length'))


    def initialize_indices(self, unlabeled_indices=None, test_indices=None):
        self.labeled_indices = np.array([], dtype=int)

        if unlabeled_indices is not None and test_indices is not None:
            self.unlabeled_indices = np.array(unlabeled_indices)
            self.test_indices      = np.array(test_indices)
        else:
            all_indices = self.full_dataset.ids.copy()
            random.shuffle(all_indices)
            self.unlabeled_indices = np.array(all_indices[self.test_size:])
            self.test_indices      = np.array(all_indices[:self.test_size])

        return
    

    def update_labeled_indices(self, indices):
        self.labeled_indices = np.append(self.labeled_indices,
                                         self.unlabeled_indices[indices])
        self.unlabeled_indices = np.delete(self.unlabeled_indices, indices)


    def make_train_test_datasets(self):
        train_dataset = self.full_dataset.make_dataset_from_indices(self.labeled_indices)
        test_dataset  = self.full_dataset.make_dataset_from_indices(self.test_indices)

        return train_dataset, test_dataset


    def warm_up(self, num_fold=None, warmup_indices=None):
        # Use warm start as a percentage of the dataset.
        if warmup_indices is None:
            # Pop first indices from unlabeled indices list (which has been shuffled)
            warmup_indices = list(range(round(self.warm_start * len(self.unlabeled_indices))))

        self.update_labeled_indices(warmup_indices)

        train_dataset, test_dataset = self.make_train_test_datasets()
        self.model.train(train_dataset, test_dataset, validation=False, verbose=self.verbose)

        if self.save_results:
            self.update_performance_results(test_dataset, num_fold)

        return


    def train(self, train_indices=None, test_indices=None, num_fold=None):
        self.model = MultiLabelBertBase(config=self.train_config)

        self.initialize_indices(unlabeled_indices=train_indices, test_indices=test_indices)

        if self.warm_start > 0:
            self.warm_up(num_fold=num_fold)

        for _ in range(self.iterations):
            self.strategy()
            train_dataset, test_dataset = self.make_train_test_datasets()
            self.model.train(train_dataset, test_dataset, validation=False, verbose=self.verbose)
            
            if self.save_results:
                self.update_performance_results(test_dataset, num_fold)

        if self.save_results:
            self.save_performance_results(num_fold)

        return classification(self.model, test_dataset,
                              labels=self.train_config.get('label_names'),
                              verbose=self.verbose)


    def update_performance_results(self, test_dataset, num_fold):
        clf_report, _ = classification(self.model, test_dataset,
                                      labels=self.train_config.get('label_names'),
                                      verbose=self.verbose)

        self.performance_results[num_fold]['micro'].append(clf_report['micro avg']['f1-score'])
        self.performance_results[num_fold]['macro'].append(clf_report['macro avg']['f1-score'])
        self.performance_results[num_fold]['motivations'].append(
            100 * (len(self.labeled_indices)) / (len(self.labeled_indices) + len(self.unlabeled_indices)))
        
        return

    
    def save_performance_results(self, num_fold):
        performances_dir = f'./performance_results_{self.strategy_name}'
        if not os.path.isdir(performances_dir):
            os.makedirs(performances_dir)

        with open(os.path.join(performances_dir, f"{num_fold}_fold.json"), 'w') as file:
            json.dump(self.performance_results[num_fold], file)

        return


    def select_strategy(self):
        if self.strategy_name == 'random':
            self.strategy = self.random_strategy
        elif self.strategy_name == 'uncertainty':
            self.strategy = self.uncertainty_strategy
        else:
            raise ValueError(f"Strategy {self.strategy_name} is not implemented.")
        print(f"Using {self.strategy_name} strategy.")

        return
    

    def random_strategy(self):
        """ Get the next batch of random samples. Since the dataset was shuffled,
        it is enough to retrieve the next batch of unlabeled data.
        """
        random_indices = list(range(self.sample_size))
        self.update_labeled_indices(random_indices)

        return


    def uncertainty_strategy(self):
        unlabeled_dataset = self.full_dataset.make_dataset_from_indices(self.unlabeled_indices)

        probabilities, _ = self.model.predict_prob(unlabeled_dataset)

        distance_to_middle = np.abs(probabilities - 0.5)
        certainty = np.mean(distance_to_middle, axis=1)
        most_uncertain_indices = np.argsort(certainty)[:self.sample_size]

        self.update_labeled_indices(most_uncertain_indices)

        return


    def k_fold(self, k=10):
        f1_scores = defaultdict(list)
        all_classifications = []
        durations = []

        if self.save_results:
            performance_result = {'strategy': self.strategy_name, 
                                  'micro': [], 'macro': [], 'profiles': [],
                                  'confusion': [], 'motivations': [], 'users': []}
            self.performance_results = {fold:copy.deepcopy(performance_result) for fold in range(k)}

        for train_indices, test_indices in self.full_dataset.kfold(k=k, only_ids=True):
            num_fold = len(durations)
            start_time = time.time()
            clf_report, classifications = self.train(train_indices, test_indices, num_fold)
            end_time = time.time()
            
            if self.verbose:
                print(f'Time spent training: {end_time - start_time}')
            
            durations.append(end_time - start_time)
            all_classifications.extend(classifications)

            for label in self.train_config.get('label_names') + ['micro avg', 'macro avg', 'weighted avg']:
                if label in clf_report:
                    f1_scores[label].append(clf_report[label]['f1-score'])

        print_results(f1_scores, self.train_config['label_names'])
        print(f'Average time spent training {sum(durations) / len(durations):.2f}')

        return

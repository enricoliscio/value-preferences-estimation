import datetime
from sklearn.metrics import classification_report
import statistics
import random
from decimal import Decimal

import torch
import pandas as pd
import numpy as np


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def set_seeds(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def classification_info(y_true, y_predicted, output_dict=False, target_names=None):
    # in case of a single target name add it's counterpart to the list
    if target_names is not None and len(target_names) == 1:
        label = target_names[0]
        target_names = [f'non-{label}', label]

    return classification_report(y_true, y_predicted, output_dict=output_dict,
                                 target_names=target_names, zero_division=1)


# Create a classification report and return a list of json objects with predictions.
def classification(model, dataset, labels, verbose=True):
    y_predicted, y_true = model.predict(dataset)
    if verbose:
        print('======== Statistics ========')
        print(classification_report(y_true, y_predicted, target_names=labels, digits=3), '\n')
    objects = make_objects(dataset.text, dataset.ids, y_predicted, y_true, labels)
    return classification_report(y_true, y_predicted, target_names=labels,
                                 output_dict=True, zero_division=1), objects


# Create a list of json objects with predictions
def make_objects(texts, ids, y_predicted, y_true, labels):
    list_of_objects = []
    for i in range(len(texts)):
        predicted_labels = [labels[j] for j, x in enumerate(y_predicted[i]) if x == 1]
        true_labels = [labels[j] for j, x in enumerate(y_true[i]) if x == 1]
        list_of_objects.append(
            {
                'id': ids[i].item(),
                'text': texts[i],
                'predicted': predicted_labels,
                'actual': true_labels
            }
        )
    return list_of_objects


# Summarize the f1 results in a table
def f1_results(f1_scores, labels):
    values = np.array([[label, round(Decimal(statistics.mean(f1_scores[label])), 2),
                        round(Decimal(statistics.stdev(f1_scores[label])), 2)] for label in
                       labels])
    classification_table = pd.DataFrame(values, columns=['Labels', 'Mean', 'SD'])
    print(classification_table)
    mean_f1 = round(Decimal(statistics.mean(classification_table["Mean"])), 2)
    return mean_f1


# Print the results of the experiment.
def print_results(f1_scores, labels):
    print("Classification table")
    f1 = f1_results(f1_scores, labels)
    print(f'\nAverage F1 Source: {f1}')

    print(f'\nAverage Micro F1: {statistics.mean(f1_scores["micro avg"]):.2f}')
    print(f'Average Macro F1: {statistics.mean(f1_scores["macro avg"]):.2f}')
    print(f'Average Weighted F1: {statistics.mean(f1_scores["weighted avg"]):.2f}')


def get_motivation_index_from_user_index(motivation_ids, user_ids):
    index = []
    for user_id in user_ids:
        user_motivation_ids = [id for id in motivation_ids if f'_{user_id}' in id]
        index.extend([np.argwhere(motivation_ids == mot_id)[0][0] for mot_id in user_motivation_ids])

    return index


def shuffle_indices(indices):
    idx = np.arange(indices.shape[0])
    np.random.shuffle(idx)
    return indices[idx]

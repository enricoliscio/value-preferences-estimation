import json
from collections import defaultdict
import time
import itertools
import statistics
import torch
import os
from transformers import AutoTokenizer

from nlp.bert.data import BertDataset
from nlp.bert.model import MultiLabelBertBase, DEFAULT_CONFIG
from nlp.bert.utils import set_seeds, classification, print_results
from value_profiles.nlp import load_motivations


def grid_search(motivations=None, hyperparameters_json=None, language='dutch',
                do_kfold=True, dataset_size=3233, verbose=True):
    
    with open(hyperparameters_json, 'r') as f:
        hyperparameters = json.load(f)

    all_hyperparameters   = [values for values in hyperparameters.values()]
    hyperparameters_names = hyperparameters.keys()

    for combination_index, combination in enumerate(itertools.product(*all_hyperparameters)):
        training_config = {}
        for name, value in zip(hyperparameters_names, combination):
            training_config[name] = value

        f1_scores = evaluate_bert(motivations=motivations, language=language, do_kfold=do_kfold,
                      dataset_size=dataset_size, training_config=training_config, verbose=verbose)

        training_config['micro_avg'] = statistics.mean(f1_scores["micro avg"])
        training_config['macro_avg'] = statistics.mean(f1_scores["macro avg"])

        with open(f"./hyperparameters_results/{combination_index}.json", 'w') as f_out:
            json.dump(training_config, f_out, indent=4)
    return


def get_model_config(language='dutch', training_config=None):
    # Initialize model_config with default config.
    model_config = DEFAULT_CONFIG

    # Set language.
    model_config['language'] = language
    try:
        model_config['name'] = model_config[language]
    except:
        raise ValueError(f"Language '{language}' not supported. Choose between dutch and english.")

    if training_config:
        for config_name, config_value in training_config.items():
            model_config[config_name] = config_value

    return model_config


def evaluate_bert(motivations=None, language='dutch', do_kfold=True, dataset_size=3233,
                  training_config=None, verbose=True, dump_results=False, save_model=False,
                  save_path='./'):
    # Get model config.
    model_config = get_model_config(language, training_config)

    print("Using the following MODEL CONFIG:")
    print(model_config)

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    set_seeds(seed_val)

    BertDataset.tokenizer = AutoTokenizer.from_pretrained(model_config.get('name'))

    dataset = BertDataset(motivations=motivations, label_names=model_config.get('label_names'),
                          max_size=dataset_size, language=model_config.get('language'),
                          model_name=model_config.get('name'),
                          max_seq_length=model_config.get('max_seq_length'),
                          shuffle=True)

    if do_kfold:
        f1_scores = defaultdict(list)
        all_classifications = []

        durations = []
        for train_dataset, test_dataset in dataset.kfold(10):
            bert = MultiLabelBertBase(config=model_config)

            start_time = time.time()
            bert.train(train_dataset, test_dataset, validation=False, verbose=verbose)
            end_time = time.time()
            
            clf_report, classifications = classification(bert, test_dataset,
                                          labels=model_config.get('label_names'), verbose=verbose)
            if verbose:
                print(f'Time spent training: {end_time - start_time}')
            
            durations.append(end_time - start_time)
            all_classifications.extend(classifications)

            for label in model_config.get('label_names') + ['micro avg', 'macro avg', 'weighted avg']:
                if label in clf_report:
                    f1_scores[label].append(clf_report[label]['f1-score'])

        print_results(f1_scores, model_config['label_names'])
        print(f'Average time spent training {sum(durations) / len(durations):.2f}')

        # Write all predictions to json
        if dump_results:
            with open(f'bert_SWF.json', 'w') as file:
                json.dump(all_classifications, file)

        if save_model:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(bert.model, os.path.join(save_path, 'model.pt'))

        return f1_scores


def predict_all_labels(language='dutch', model_path=None):
    model_config = get_model_config(language)
    loaded_model = torch.load(model_path)
    model = MultiLabelBertBase(config=model_config, model=loaded_model)

    motivations = load_motivations()
    dataset = BertDataset(motivations=motivations, label_names=model_config.get('label_names'),
                      language=model_config.get('language'), model_name=model_config.get('name'),
                      max_seq_length=model_config.get('max_seq_length'))

    y_predicted, y_true = model.predict(dataset)

    return y_predicted, y_true
import argparse
import warnings
import os
from transformers.utils import logging

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.set_verbosity_error()

from nlp.bert.training import evaluate_bert, grid_search
from value_profiles.nlp import load_motivations

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fold',            type=bool, default=True)
parser.add_argument('--size',            type=int,  default=3233)
parser.add_argument('--language',        type=str,  default="dutch")
parser.add_argument('--grid-search',     type=bool, default=False)
parser.add_argument('--hyperparameters', type=str,  default='./hyperparameters_testing/tested_hyperparameters.json')
parser.add_argument('--save-model',      type=bool, default=False)
parser.add_argument('--model-save-dir',  type=str,  default='./trained_model')

args = parser.parse_args()
motivations = load_motivations()

if args.grid_search:
    grid_search(motivations=motivations, hyperparameters_json=args.hyperparameters, language=args.language,
                do_kfold=args.fold, dataset_size=args.size, verbose=False)
else:
    _ = evaluate_bert(motivations=motivations, language=args.language, do_kfold=args.fold,
                      dataset_size=args.size, save_model=args.save_model, save_path=args.model_save_dir)

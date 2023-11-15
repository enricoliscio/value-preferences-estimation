import argparse
import warnings
import os
from transformers.utils import logging

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.set_verbosity_error()

from nlp.active_learning.training import ActiveLearning
from value_profiles.nlp import load_motivations

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--size',        type=int,  default=3233)
parser.add_argument('--language',    type=str,  default='dutch')
parser.add_argument('--iterations',  type=int,  default=5)
parser.add_argument('--sample-size', type=int,  default=146)
parser.add_argument('--test-size',   type=int,  default=300)
parser.add_argument('--strategy',    type=str,  default='uncertainty')
parser.add_argument('--warm-start',  type=int,  default=291)
parser.add_argument('--save-results',type=bool, default=True)
parser.add_argument('--verbose',     type=bool, default=False)
parser.add_argument('--k-fold',      type=int,  default=10)

args = parser.parse_args()
motivations = load_motivations()

active_learning = ActiveLearning(iterations=args.iterations, sample_size=args.sample_size,
               strategy=args.strategy, warm_start=args.warm_start,
               dataset_size=args.size, test_size=args.test_size,
               language=args.language, motivations=motivations,
               save_results=args.save_results, verbose=args.verbose)

if args.k_fold:
    active_learning.k_fold(k=args.k_fold)
else:
    active_learning.train()
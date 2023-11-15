import numpy as np
import json
import glob
import os
import matplotlib.pyplot as plt
import itertools
import pandas as pd


def load_results(folder_path):
    all_results = {}

    for file in glob.glob(os.path.join(folder_path, "*.json")):
        with open(file, 'r') as f_in:
            file_name = os.path.split(file)[-1]
            num_fold = int(file_name.split('_')[0])
            all_results[num_fold] = json.load(f_in)

    return all_results


def retrieve_iteration_steps(all_results, iterations, step_type):
    # Average the size of the users/motivations in each iteration step.
    iteration_step = [[round(score) for score in scores[step_type]] for scores in all_results.values()]
    iterations[step_type] = np.array(iteration_step).mean(axis=0).round().astype(int)

    if not all([a == b for a, b in itertools.combinations(iteration_step, 2)]):
       print(f"Not all folds used the same amount of {step_type} in warm-up and iteration steps.")
       print(iteration_step)

    return iterations


def retrieve_iterations(all_results, f1_type='micro'):
    assert len(set([len(scores[f1_type]) for scores in all_results.values()])) == 1,\
        f"Not all folds have the same amount of F1-scores."

    num_iterations = len(all_results[1][f1_type])
    iterations = {i:{'results':[]} for i in range(num_iterations)}

    iterations = retrieve_iteration_steps(all_results, iterations, 'motivations')
    iterations = retrieve_iteration_steps(all_results, iterations, 'users')

    for iteration in range(num_iterations):
        for results in all_results.values():
            iterations[iteration]['results'].append(results[f1_type][iteration])

        iterations[iteration]['average'] = np.mean(iterations[iteration]['results'])
        iterations[iteration]['std']     = np.std(iterations[iteration]['results'])

    return iterations


def make_plot_data(iterations, x_axis_label):
    x = iterations.pop(x_axis_label)

    folds = range(len(iterations)-1) # ignore the key that has not been popped
    y = [iterations[i]['average'] for i in folds]
    y_err = [iterations[i]['std'] for i in folds]

    return x, y, y_err


def plot_iterations(iterations, x_axis_label, color, label):
    x, y, y_err = make_plot_data(iterations, x_axis_label)
    plt.errorbar(x, y, yerr=y_err, color=color, label=label)

    return x, y, y_err


def plot_performances(folder_path, metric, x_axis_label, color='b', label=''):
    all_results = load_results(folder_path)
    iterations  = retrieve_iterations(all_results, f1_type=metric)
    return plot_iterations(iterations, x_axis_label, color, label)


def plot_one_result(folder_path, metric='profiles', topline=None, x_axis_label='users'):
    plot_performances(folder_path, metric, x_axis_label)
    if topline is not None:
        plt.axhline(y=topline, color='m', linestyle='--')


def dump_dat(x, y, std, save_path):
    results = pd.DataFrame(columns=['x', 'y', 'std'])
    results['x'] = x
    results['y'] = y
    results['std'] = std
    results.to_csv(save_path, index=False, sep='\t')


def plot_all_results(folder_path='./results', metric='profiles', topline=None,
                     save_path='', save_dat_folder='./results/plots'):
    plt.rc('font', size=15)

    save_dat_folder = os.path.join(save_dat_folder, metric)
    x, y, y_err = plot_performances(
        os.path.join(folder_path, 'performance_results_random_users'),
        metric, 'motivations', 'g', 'random')
    dump_dat(x, y, y_err, os.path.join(save_dat_folder, 'random.dat'))
    x, y, y_err = plot_performances(
        os.path.join(folder_path, 'performance_results_uncertainty'),
        metric, 'motivations', 'r', 'uncertainty')
    dump_dat(x, y, y_err, os.path.join(save_dat_folder, 'uncertainty.dat'))
    x, y, y_err = plot_performances(
        os.path.join(folder_path, 'performance_results_value_profiles'),
        metric, 'motivations', 'b', 'disambiguation')
    dump_dat(x, y, y_err, os.path.join(save_dat_folder, 'disambiguation.dat'))

    if topline is not None:
        plt.axhline(y=topline, color='m', linestyle='--')

    plt.legend(framealpha=1, frameon=True)

    if len(save_path) > 0:
        plt.savefig(save_path, dpi=300)

    plt.show()

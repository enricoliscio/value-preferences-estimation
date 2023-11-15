from .nlp import value_labels
from .utils import get_value_ranking, methods

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

name_columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']


def initialize_analysis():
    sheet_to_df_map = {}
    cont = 0
    results_eval = {}
    results_eval_index = {}
    results_eval_ID = {}
    v1 = {}  # Values (v1) annotated by reviewer X
    v2 = {}  # Values (v2) annotated by reviewer X

    return sheet_to_df_map, cont, results_eval, results_eval_index, results_eval_ID, v1, v2


def analyse_results_eval(xls, name_columns, sheet_to_df_map, cont, results_eval,
                         results_eval_index, results_eval_ID, v1, v2):
    for sheet_name in xls.sheet_names:  # For all tabs of the spreadsheeet
        if sheet_name == 'Instructions':
            continue
        sheet_to_df_map[sheet_name] = xls.parse(sheet_name)
        results_eval_index[cont] = np.where(pd.isna(sheet_to_df_map[sheet_name][9:13][name_columns]) == False)[0]
        results_eval[cont] = np.where(pd.isna(sheet_to_df_map[sheet_name][9:13][name_columns]) == False)[1]
        results_eval_ID[cont] = float(sheet_name)

        aux_v1 = {}
        aux_v2 = {}
        for it in range(0, len(results_eval_index[cont])):
            txt = sheet_to_df_map[sheet_name].iloc[9 + it, 0]  # 9 is the initial row of values
            x = re.findall(r'(^v1\s=\s)([A-z]+)', txt)[0][1]
            # r: regular expression; ^:start of the string; \s:empty space; [A-z]: any letter from A to z;
            # +: Continuation of the previous group; () means a caputre group
            aux_v1[it] = value_labels.index(x)
            x = re.findall(r'(\sv2\s=\s)([A-z]+)', txt)[0][1]
            aux_v2[it] = value_labels.index(x)

        v1[cont] = aux_v1
        v2[cont] = aux_v2
        cont += 1

    return v1, v2, results_eval_index, results_eval, results_eval_ID, cont


def get_evaluator_data(evaluator=1):
    sheet_to_df_map, cont, results_eval, results_eval_index, results_eval_ID, v1, v2 = initialize_analysis()

    #Analyse Review 1 from Evaluator 1
    xls = pd.ExcelFile(f'./value_profiles/data/evaluation/Reviewer{evaluator}_Evaluation_1.xlsx')
    v1, v2, results_eval_index, results_eval, results_eval_ID, cont = analyse_results_eval(
    xls, name_columns, sheet_to_df_map, cont, results_eval, results_eval_index, results_eval_ID, v1, v2)

    #Analyse Review 2 from Evaluator 1
    xls = pd.ExcelFile(f'./value_profiles/data/evaluation/Reviewer{evaluator}_Evaluation_2.xlsx')
    v1, v2, results_eval_index, results_eval, results_eval_ID, _ = analyse_results_eval(
    xls, name_columns, sheet_to_df_map, cont, results_eval, results_eval_index, results_eval_ID, v1, v2)

    return v1, v2, results_eval_index, results_eval, results_eval_ID


def get_plot_data(v1, v2, results_eval_index, results_eval, results_eval_ID, complete_results):
    support_method_ev = {methods[0]: 0, methods[1]: 0, methods[2]: 0, methods[3]: 0, methods[4]: 0, methods[5]: 0}
    cont_spt = 0

    for i in range(0, len(results_eval_ID)): #For this experiments, the IDs for reviewer 1 and 2 are the same
        current_ID = results_eval_ID[i]
        for l in range(0, 4 - 1): #We have up to 4 questions for each ID
            if any(l == results_eval_index[i]): #Both reviewers answered this questions?
                response = results_eval[i][l] #Which is the same as reviewer
                if response != 3: #If the response is NOT "I dont know"
                    cont_spt+= 1
                    for m in methods:
                        if response == 0: #v1 > v2 -> Here v1 and v2 for both reviewers are the same. Thus we use v11 and v12 from here on
                            if complete_results[current_ID][m][v1[i][l]] > complete_results[current_ID][m][v2[i][l]]:
                                #Is the response provided by the method v1 > v2?
                                support_method_ev[m]+=1
                        elif response == 1: #v2 > v1
                            if complete_results[current_ID][m][v1[i][l]] < complete_results[current_ID][m][v2[i][l]]:
                                #Is the response provided by the method v1 > v2?
                                support_method_ev[m]+=1
                        elif response == 2: #v2 ~ v1
                            if complete_results[current_ID][m][v1[i][l]] == complete_results[current_ID][m][v2[i][l]]:
                                #Is the response provided by the method v1 > v2?
                                support_method_ev[m]+=1

    support_method_ev_pct = [x *100 / cont_spt for x in list(support_method_ev.values())]

    return support_method_ev_pct


def get_plot_data_agree(v1, v2, results_eval_index, results_eval1, results_eval2, results_eval_ID, complete_results):
    support_method_ev = {methods[0]: 0, methods[1]: 0, methods[2]: 0, methods[3]: 0, methods[4]: 0, methods[5]: 0}
    cont_spt = 0

    for i in range(0, len(results_eval_ID)): #For this experiments, the IDs for reviewer 1 and 2 are the same
        current_ID = results_eval_ID[i]
        for l in range(0, 4 - 1): #We have up to 4 questions for each ID
            if any(l == results_eval_index[i]): #Both reviewers answered this questions?
                #Did the reviewers agree on the answer?
                if (results_eval1[i].size > 0) and (results_eval2[i].size > 0): #Check if the reviewers agreed 
                    if results_eval1[i][l] == results_eval2[i][l]: #Check if the reviewers agreed 
                        response = results_eval1[i][l] #Which is the same as reviewer 2
                        if response != 3: #If the response is NOT "I dont know"
                            cont_spt+= 1
                            for m in methods:
                                # results_eval: 0: v1 > v2; 1: v2 > v1; 2: v1 ~ v2
                                if response == 0: #v1 > v2 -> Here v1 and v2 for both reviewers are the same. Thus we use v11 and v12 from here on
                                    if complete_results[current_ID][m][v1[i][l]] > complete_results[current_ID][m][v2[i][l]]:
                                        #Is the response provided by the method v1 > v2?
                                        support_method_ev[m]+=1
                                elif response == 1: #v2 > v1
                                    if complete_results[current_ID][m][v1[i][l]] < complete_results[current_ID][m][v2[i][l]]:
                                        #Is the response provided by the method v1 > v2?
                                        support_method_ev[m]+=1
                                elif response == 2: #v2 ~ v1
                                    if complete_results[current_ID][m][v1[i][l]] == complete_results[current_ID][m][v2[i][l]]:
                                        #Is the response provided by the method v1 > v2?
                                        support_method_ev[m]+=1

    support_method_ev_pct = [x *100 / cont_spt for x in list(support_method_ev.values())]

    return support_method_ev_pct


def plot_agreement(evaluation_percent, save_path=None):
    evaluation_percent = [evaluation_percent[0], evaluation_percent[5], evaluation_percent[1], evaluation_percent[2], evaluation_percent[3], evaluation_percent[4]]
    labels = evaluation_percent

    fig = plt.figure(figsize=(5,3.5))
    fig.tight_layout()
    plt.rc('font', size=14)

    plt.bar(["$R_C$","$R_{M}$", "$R_{TB}$", "$R_{MC}$","$R_{MO}$", "$R_{MO+MC+TB}$"], evaluation_percent,
        zorder=3, width=0.8,  color='#89bedc',alpha=0.8, edgecolor='black')
    plt.xlabel('Methods')
    plt.ylabel('Performance (%)')
    plt.xticks(rotation=30)

    for i, _ in enumerate(labels):
        plt.text(i-.3, labels[i]-(max(labels)*0.2), "%.1f" %labels[i]+'%', fontsize=10, color = 'black')

    plt.tick_params(direction='out', length=3, width=1, grid_alpha=0.5, which='both', top=0, bottom=0, right=1)

    plt.show()

    if save_path is not None:
        fig.savefig(save_path, format='pdf')


def get_average_changes(complete_results, methods):
    changed = np.zeros((6,5))
    results = {}

    for cont_m, m in enumerate(methods):
        results[m]  = []
        changed_aux = [0, 0, 0, 0, 0]

        for id in complete_results.columns:
            rank_base = get_value_ranking(complete_results[id]['ER_C'])
            rank_m    = get_value_ranking(complete_results[id][m])
            changed_aux = changed_aux + abs(rank_m - rank_base)
            results[m].append(sum(abs(rank_m - rank_base)))
        changed[cont_m] = changed_aux / len(complete_results.columns)

    return pd.DataFrame.from_dict(results).melt()


def plot_changes(complete_results, save_path=None, plot_M=False):
    plot_order = ['ER_C_TB', 'ER_C_MC', 'ER_C_MO', 'ER_C_MO_MC_TB']
    if plot_M:
        plot_order.append('ER_M')

    changes = get_average_changes(complete_results, plot_order)

    fig, ax = plt.subplots(figsize=(5,3.5))
    plt.rc('font', size=14)

    plt.tick_params(direction='out', length=3, width=1, grid_alpha=0.5, which='both', top=0, bottom=0, right=1)
    sns.boxplot(x="variable", y="value", data=changes, color='lightskyblue', ax=ax, order=plot_order)

    ax.set_xticklabels(["$R_{TB}$", "$R_{MC}$","$R_{MO}$", "$R_{MO+MC+TB}$"])
    plt.xlabel('Method')
    plt.ylabel('Changes from $R_C$')

    plt.show()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format='pdf')

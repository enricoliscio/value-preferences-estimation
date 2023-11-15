import numpy as np
import pandas as pd
from itertools import product, combinations
from scipy.stats import rankdata, kendalltau
from scipy.spatial.distance import pdist, squareform

from .nlp import load_motivations

projects      = ['project_228', 'project_230', 'project_231', 'project_232', 'project_233', 'project_234']
project_codes = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']
value_labels  = ['kosten', 'land', 'leiding', 'samen', 'zelf']
methods       = ['ER_C','ER_C_TB','ER_C_MC','ER_C_MO','ER_C_MO_MC_TB','ER_M']


def dump_participant_choices(path='./value_profiles/data/participants_choices.csv'):
	""" Dump participant choices to csv.
		Use only selected values and only participants who provided a motivation.
	"""
	xls = pd.ExcelFile('./value_profiles/data/20200525_5_LK.xlsx')
	choices_participants = pd.read_excel(xls, '20200519_4e sessie LK')
	choices_participants['id'] = choices_participants['id'].astype('Int64')
	choices_participants.set_index('id', inplace=True)

	# Load annotated motivations
	df = load_motivations()

	# Retrieve only the participant IDs that wrote a motivation
	participant_ids = np.unique(df['participant_id']).tolist()
	participant_ids.remove(118373)  # this participants does not appear in 20200525_5_LK.xlsx

	# Retrieve point distribution of the selected participant IDs
	points = [choices_participants[projects].loc[participant_id].to_list() for participant_id in participant_ids]
	df_points = pd.DataFrame(np.array(points), index=participant_ids, columns=project_codes)
	df_points.to_csv(path, index_label='id')

	return


def load_choices(path='./value_profiles/data/participants_choices.csv'):
	return pd.read_csv(path, index_col='id')


def load_correct_rankings(use_predicted=False):
	if use_predicted:
		return pd.read_csv('./value_profiles/data/complete_predicted_rankings.csv', index_col='id')
	else:
		return pd.read_csv('./value_profiles/data/complete_rankings.csv', index_col='id')


def get_allocated_points(participant_id, choices):
    """Retrieve the points that a participant allocated to the 6 project questions.
	"""
    return choices.loc[participant_id, :].to_numpy()


def get_participant_motivations(motivations, participant_id):
	""" Retrieve the participant's motivations (ad a DataFrame).
	"""
	return motivations[motivations['participant_id'] == participant_id]


def get_value_ranking(weighted_values):
	""" Given weighted values, return value rankings.
	"""
	# We use a minus sign to rank as 1st the highest value and last the lowest value
	return rankdata(-weighted_values, method='min')


def get_motivation_values(motivation):
	""" Given a motivation (DataFrame row), return the indices of the annotated values.
	"""
	return np.flatnonzero(motivation[value_labels])


def get_VO_matrix(motivations, threshold=20):
	""" Calculate binary VO matrix, with a 1 for each value that has been annotated
	  at least <threshold> times for a project question.
	"""
	# Get total amount of annotations for a value for a project question
	df_total_annotations = pd.DataFrame(columns=range(1, len(project_codes)+1), index=value_labels)
	for project_code, value_name in product(range(1, len(project_codes)+1), value_labels):
		annotations = motivations[motivations['question_id'] == project_code]
		df_total_annotations.at[value_name, project_code] = annotations[value_name].sum()
    
	df_total_annotations.columns = project_codes

	# Return binary matrix with 1 where the number of annotations is larger than threshold
	return (df_total_annotations >= threshold).astype(int)


def get_VO_values(VO_matrix, question_index):
	""" Given a VO matrix and the index of a project question,
		return the indices of the values relevant to the question in VO.
	"""
	return np.flatnonzero(VO_matrix[question_index])


def check(it, num):
    it = iter(it)
    return all(any(it) for _ in range(num))


def kendall_tau_from_rankings(rank1, rank2):
    return kendalltau(rank1, rank2)[0]


def kemenydesign(X):
    # Compute the design matrix to compute Kemeny distance   
    M = X.shape[1]
    N = X.shape[0]
    indice = list(combinations(range(M), 2))
    KX = np.zeros((N, int(M*(M-1)/2)))

    for j in range(len(indice)):
        KX[:, j] = np.sign(X[:, indice[j][0]] - X[:, indice[j][1]]) * -1

    return KX


def kemenyd(X, Y):
    ## Kemeny Distance
    X = kemenydesign(np.matrix(X))
    Y = kemenydesign(np.matrix(Y))
    d = pdist(np.vstack([X,Y]), "cityblock")
    d = squareform(d)[len(X):, :len(X)]

    return d[0][0]


def make_full_results_df(ids):
	""" Create DataFrame to store value scores.
		Index=[participant_ids], Columns = [methods, values].
	"""
	# Create NaN matrix of data.
	data = np.empty((len(ids), len(methods) * len(value_labels)))
	data[:] = np.NaN

	# Create column indices.
	values_indices  = value_labels * len(methods)
	methods_indices = [m for method in methods for m in [method] * len(value_labels)]

	# Create columns as MultiIndex and index as participants' ID.
	columns = pd.MultiIndex.from_tuples(
		list(zip(methods_indices, values_indices)), names=['method', 'value'])
	index   = pd.Index(ids, name='participant_id')

	# Create multi-dimensional DataFrame.
	df = pd.DataFrame(data=data, columns=columns, index=index)

	# Transpose the dataframe for easier indexing.
	return df.T

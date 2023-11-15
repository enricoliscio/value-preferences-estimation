import numpy as np
from nlp.bert.utils import get_motivation_index_from_user_index


def get_user_index_from_motivation_index(motivation_ids, indices):
    user_motivation_ids = motivation_ids[indices]
    # Return unique user_ids that provided the motivations.
    return np.unique([idx.split('_')[1] for idx in user_motivation_ids])


def match_indices(target_indices, indices):
    return [np.argwhere(target_indices == index)[0][0] for index in indices \
            if index in target_indices]


def get_labeled_indices_from_user_indices(ids, target_indices, users):
    # Find the index of the motivations that where written by the users.
    motivation_indices = get_motivation_index_from_user_index(ids, users)
    # Match the index of the motivations to the index of target_indices.
    return match_indices(target_indices, motivation_indices)

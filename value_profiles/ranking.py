import numpy as np

from .utils import *


def method_C(id_evaluation, VO_matrix, choices):
    """ Calculate the ranking from participants from their choices.
    """
    points_projects = get_allocated_points(id_evaluation, choices)

    # Calculate value ranking as VO * points
    # Note: In some cases the participants distributed less than 100 points
    weighted_values = np.matmul(VO_matrix.values, points_projects)

    ranking_values = get_value_ranking(weighted_values)

    return weighted_values, ranking_values


def method_M(id_evaluation, motivations):
    """ Calculate the number of values mentioned in the motivations.
    """
    motivations_participant = get_participant_motivations(motivations, id_evaluation)
    weighted_values = motivations_participant[value_labels].sum(axis=0)

    ranking_values = get_value_ranking(weighted_values)

    return weighted_values, ranking_values


def method_TB(id_evaluation, weighted_values, motivations, increase_value=0.5):
    """ Break ties from already calculated rankings (weighted_values).
    """
    weighted_values_updated = weighted_values.copy().astype(float)
    num_ties = 0

    motivations_participant = get_participant_motivations(motivations, id_evaluation)

    for _, motivation in motivations_participant.iterrows():
        # Get index of supported values.
        supported_values_idx = get_motivation_values(motivation)

        # If those value(s) have a score equal to another value, add increase_value to them.
        for value_idx in supported_values_idx:
            if check(weighted_values_updated[value_idx] == weighted_values, 2):
                # Compare to the original ranking to avoid giving priority to first projects.
                weighted_values_updated[value_idx] += increase_value
                num_ties += 1

    ranking_values = get_value_ranking(weighted_values_updated)

    return weighted_values_updated, ranking_values, num_ties


def method_MC(id_evaluation, VO_matrix, weighted_values, choices, motivations):
    """ Address inconsistencies between choices and motivations, favoring the motivations.
    """
    VO_matrix_updated = VO_matrix.copy()
    motivations_participant = get_participant_motivations(motivations, id_evaluation)

    for _, motivation in motivations_participant.iterrows():
        # Get index of supported values.
        supported_values_idx = get_motivation_values(motivation)

        # Refer to Algorithm 2 in the paper for an explanation.
        for value_idx in supported_values_idx:
            other_value_indices = list(range(len(value_labels)))
            other_value_indices.remove(value_idx)
            for other_value_idx in other_value_indices:
                if weighted_values[value_idx] <= weighted_values[other_value_idx]:
                    VO_matrix_updated.at[
                        value_labels[other_value_idx],
                        project_codes[motivation['question_id'] - 1] # q_id starts from 1
                        ] = 0

    weighted_values_updated, ranking_values = method_C(id_evaluation, VO_matrix_updated, choices)

    return weighted_values_updated, ranking_values, VO_matrix_updated


def method_MO(id_evaluation, VO_matrix, choices, motivations):
    """ Address inconsistencies between values provided for two different motivations,
        selectively updating VO.
    """
    VO_matrix_updated = VO_matrix.copy()
    motivations_participant = get_participant_motivations(motivations, id_evaluation)

    # See Algorithm 3 in paper for loop and variable names
    for idx_a, m_a in motivations_participant.iterrows():
        values_m_a  = get_motivation_values(m_a)
        values_VO_a = get_VO_values(VO_matrix, f"q{m_a['question_id']}")
        V_alpha = [value_idx for value_idx in values_VO_a if value_idx not in values_m_a]
        for _, m_b in motivations_participant.drop(idx_a).iterrows():
            values_m_b = get_motivation_values(m_b)
            for v_x in V_alpha:
                if v_x in values_m_b:
                    for v_y in values_m_a:
                        values_VO_b = get_VO_values(VO_matrix, f"q{m_b['question_id']}")
                        V_beta = [value_idx for value_idx in values_VO_b if value_idx not in values_m_b]
                        if v_y in V_beta:
                            VO_matrix_updated.at[
                                value_labels[v_x],
                                f"q{m_a['question_id']}"
                                ] = 0

    weighted_values_updated, ranking_values = method_C(id_evaluation, VO_matrix_updated, choices)

    return weighted_values_updated, ranking_values, VO_matrix_updated


def method_MO_MC_TB(id_evaluation, VO_matrix, choices, motivations):
    """ Concatenate the three methods.
    """
    weighted_values_MO, _, VO_matrix_MO = method_MO(id_evaluation, VO_matrix, choices, motivations)

    weighted_values_MC, _, VO_matrix_MC = method_MC(id_evaluation, VO_matrix_MO, weighted_values_MO, choices, motivations)

    weighted_values_TB, ranking_values_TB, num_ties = method_TB(id_evaluation, weighted_values_MC, motivations)

    return weighted_values_TB, ranking_values_TB, VO_matrix_MC, num_ties

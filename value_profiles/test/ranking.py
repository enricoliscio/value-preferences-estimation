import unittest

from value_profiles.nlp import load_motivations
from value_profiles.ranking import *
from value_profiles.utils import *


class TestValueRanking(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestValueRanking, self).__init__(*args, **kwargs)
        self.value_labels  = ['kosten', 'land', 'leiding', 'samen', 'zelf']
        self.project_codes = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']
        self.id_eval = 117924
        self.choices = load_choices()
        self.motivations = load_motivations()
        self.VO_matrix = get_VO_matrix(self.motivations, threshold=20)

    def test_VO_matrix(self):
        VO_matrix = np.array([[1, 1, 1, 1, 1, 1],
                              [1, 1, 0, 1, 1, 1],
                              [1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 0, 0, 1],
                              [1, 1, 0, 0, 0, 0]])
        VO_matrix = pd.DataFrame(VO_matrix,
                                 columns=self.project_codes, index=self.value_labels)

        self.assertTrue(VO_matrix.equals(self.VO_matrix))

    def test_get_allocated_points(self):
        points = np.array([40, 30, 20,  0, 10,  0])
        self.assertTrue(np.array_equal(
            points, get_allocated_points(self.id_eval, self.choices)))
        
    def test_get_participant_motivations(self):
        motivations_participant = get_participant_motivations(self.motivations, self.id_eval)
        retrieved_annotations = motivations_participant[self.value_labels]

        annotations = [[0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]]
        points = get_allocated_points(self.id_eval, self.choices)
        answered_questions = np.argwhere(points != 0)
        index = [f"{idx[0]+1}_{self.id_eval}" for idx in answered_questions]
        annotations = pd.DataFrame(annotations, index=index, columns=self.value_labels)

        self.assertTrue(annotations.equals(retrieved_annotations))

    def test_method_C(self):
        weighted_values = [100, 80, 90, 90, 70]
        ranking_values  = [1, 4, 2, 2, 5]
        weighted_values = pd.Series(weighted_values, index=self.value_labels)
        weighted_values_C, ranking_values_C = method_C(
            self.id_eval, self.VO_matrix, self.choices)
        
        self.assertTrue(weighted_values.equals(weighted_values_C))
        self.assertTrue((ranking_values == ranking_values_C).all())

    def test_method_M(self):
        weighted_values_M, ranking_values_M = method_M(
            self.id_eval, self.motivations)
        
        weighted_values = [0, 0, 1, 0, 1]
        weighted_values = pd.Series(weighted_values, index=self.value_labels)
        ranking_values  = [3, 3, 1, 3, 1]

        self.assertTrue(weighted_values.equals(weighted_values_M))
        self.assertTrue((ranking_values == ranking_values_M).all())

    def test_method_TB(self):
        weighted_values_C, _ = method_C(self.id_eval, self.VO_matrix, self.choices)
        weighted_values_TB, ranking_values_TB, num_ties_TB = method_TB(
            self.id_eval, weighted_values_C, self.motivations, increase_value=0.8)

        weighted_values = [100, 80, 90.8, 90, 70]
        weighted_values = pd.Series(weighted_values, index=self.value_labels)
        ranking_values  = [1, 4, 2, 3, 5]
        num_ties = 1

        self.assertTrue(weighted_values.equals(weighted_values_TB))
        self.assertTrue((ranking_values == ranking_values_TB).all())
        self.assertEqual(num_ties, num_ties_TB)

    def test_method_MC(self):
        weighted_values_C, _ = method_C(self.id_eval, self.VO_matrix, self.choices)
        weighted_values_MC, ranking_values_MC, VO_matrix_MC = method_MC(
            self.id_eval, self.VO_matrix, weighted_values_C, self.choices, self.motivations)
        
        VO_matrix = np.array([[0, 0, 1, 1, 1, 1],
                              [1, 0, 0, 1, 1, 1],
                              [1, 0, 1, 0, 0, 0],
                              [1, 0, 1, 0, 0, 1],
                              [1, 1, 0, 0, 0, 0]])
        VO_matrix = pd.DataFrame(VO_matrix,
                                 columns=self.project_codes, index=self.value_labels)
        weighted_values = [30, 50, 60, 60, 70]
        weighted_values = pd.Series(weighted_values, index=self.value_labels)
        ranking_values  = [5, 4, 2, 2, 1]
        
        self.assertTrue(VO_matrix.equals(VO_matrix_MC))
        self.assertTrue(weighted_values.equals(weighted_values_MC))
        self.assertTrue((ranking_values == ranking_values_MC).all())

    def test_method_MO(self):
        weighted_values_MO, ranking_values_MO, VO_matrix_MO = method_MO(
            self.id_eval, self.VO_matrix, self.choices, self.motivations)
        
        VO_matrix = np.array([[1, 1, 1, 1, 1, 1],
                              [1, 1, 0, 1, 1, 1],
                              [1, 0, 1, 0, 0, 0],
                              [1, 1, 1, 0, 0, 1],
                              [0, 1, 0, 0, 0, 0]])
        VO_matrix = pd.DataFrame(VO_matrix,
                                 columns=self.project_codes, index=self.value_labels)
        weighted_values = [100, 80, 60, 90, 30]
        weighted_values = pd.Series(weighted_values, index=self.value_labels)
        ranking_values  = [1, 3, 4, 2, 5]

        self.assertTrue(VO_matrix.equals(VO_matrix_MO))
        self.assertTrue(weighted_values.equals(weighted_values_MO))
        self.assertTrue((ranking_values == ranking_values_MO).all())

    def test_method_MO_MC_TB(self):
        weighted_values_MO_MC_TB, ranking_values_MO_MC_TB, VO_matrix_MO_MC_TB, num_ties_MO_MC_TB \
        = method_MO_MC_TB(self.id_eval, self.VO_matrix, self.choices, self.motivations)
        
        VO_matrix = np.array([[0, 0, 1, 1, 1, 1],
                              [0, 0, 0, 1, 1, 1],
                              [1, 0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0, 1],
                              [0, 1, 0, 0, 0, 0]])
        VO_matrix = pd.DataFrame(VO_matrix,
                                 columns=self.project_codes, index=self.value_labels)
        weighted_values = [30, 10, 60, 20, 30.5]
        weighted_values = pd.Series(weighted_values, index=self.value_labels)
        ranking_values  = [3, 5, 1, 4, 2]
        num_ties = 1

        self.assertTrue(VO_matrix.equals(VO_matrix_MO_MC_TB))
        self.assertTrue(weighted_values.equals(weighted_values_MO_MC_TB))
        self.assertTrue((ranking_values == ranking_values_MO_MC_TB).all())
        self.assertEqual(num_ties, num_ties_MO_MC_TB)

if __name__ == '__main__':
    unittest.main()
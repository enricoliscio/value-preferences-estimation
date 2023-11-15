from .training import ActiveLearning
from value_profiles.ranking import *
from nlp.bert.data import ValueDataset
from .utils import *
from nlp.bert.utils import shuffle_indices


class ValueLearning(ActiveLearning):
    def __init__(self, iterations=5, sample_size=100, strategy='random', warm_start=0.1,
                 dataset_size=3233, test_size=300, language='dutch', choices=None,
                 motivations=None, correct_rankings=None, training_config=None,
                 save_results=False, verbose=True):
        assert warm_start < 1, "Warm start should be passed as percentage."

        super().__init__(iterations, sample_size, strategy, warm_start, dataset_size, test_size,
                         language, motivations, training_config, save_results, verbose)

        self.correct_rankings = correct_rankings
        self.full_dataset = ValueDataset(motivations=motivations, choices=choices,
                            label_names=self.train_config.get('label_names'), shuffle=True,
                            max_size=dataset_size, language=self.train_config.get('language'),
                            model_name=self.train_config.get('model_name'),
                            max_seq_length=self.train_config.get('max_seq_length'))


    def initialize_indices(self, unlabeled_indices=None, test_indices=None):
        super().initialize_indices(unlabeled_indices, test_indices)

        self.labeled_users   = np.array([], dtype=str)
        self.unlabeled_users = get_user_index_from_motivation_index(self.full_dataset.ids,
                                                                    unlabeled_indices)
        self.test_users      = get_user_index_from_motivation_index(self.full_dataset.ids,
                                                                    test_indices)

        self.unlabeled_users = shuffle_indices(self.unlabeled_users)
        self.test_users      = shuffle_indices(self.test_users)


    def update_user_labeled_indices(self, users, update_labeled_indices=False):
        self.labeled_users   = np.append(self.labeled_users, users)
        self.unlabeled_users = np.setdiff1d(self.unlabeled_users, users) # remove users from unlabeled_users

        if update_labeled_indices:
            labeled_indices = get_labeled_indices_from_user_indices(
                self.full_dataset.ids, self.unlabeled_indices, self.labeled_users)
            super().update_labeled_indices(labeled_indices)


    def warm_up(self, num_fold=None):
        num_warmup_users = round(self.warm_start * len(self.unlabeled_users))
        warmup_users     = self.unlabeled_users[:num_warmup_users]  # unlabeled_users was shuffled
        self.update_user_labeled_indices(warmup_users)

        # Find the index of the motivations that where written by the users,
        # and match the motivation index to self.unlabeled_idices
        labeled_indices = get_labeled_indices_from_user_indices(
            self.full_dataset.ids, self.unlabeled_indices, self.labeled_users)
        super().warm_up(num_fold=num_fold, warmup_indices=labeled_indices)


    def select_strategy(self):
        if self.strategy_name == 'value_profiles':
            self.strategy = self.value_strategy
            print(f"Using {self.strategy_name} strategy.")
        else:
            super().select_strategy()


    def random_strategy(self):
        """ Get a batch of random users and related motivations.
        """
        random_users = self.unlabeled_users[:self.sample_size]
        self.update_user_labeled_indices(random_users, update_labeled_indices=True)


    def predict_motivations(self, users):
        motivations_users = self.full_dataset.motivations_df[
            self.full_dataset.motivations_df['participant_id'].isin(users)]

        # Predict labels on unlabeled user motivations.
        user_ids = get_motivation_index_from_user_index(self.full_dataset.ids, users)
        users_dataset = self.full_dataset.make_dataset_from_indices(user_ids)
        y_predicted, _ = self.model.predict(users_dataset)

        # Replace annotated labels with predicted labels.
        for mot_id, y_pred in zip(users_dataset.ids, y_predicted.astype(int)):
            motivations_users.loc[mot_id, value_labels] = y_pred

        return motivations_users


    def compute_VO_matrix(self, motivations_unlabeled_users=None, motivations_test_users=None):
        """ Compute VO matrix. By default use only labeled users.
            If requested, use also unlabeled and test users.
        """
        # Retrieve labeled users motivations dataframe.
        labeled_users = [int(user) for user in self.labeled_users]
        motivations_labeled_users = self.full_dataset.motivations_df[
            self.full_dataset.motivations_df['participant_id'].isin(labeled_users)]

        motivations = motivations_labeled_users
        threshold   = 0.025*len(labeled_users)
        if motivations_unlabeled_users is not None:
            motivations = pd.concat([motivations, motivations_unlabeled_users])
            threshold   = 20
        if motivations_test_users is not None:
            motivations = pd.concat([motivations, motivations_test_users])
            threshold   = 20

        return get_VO_matrix(motivations, threshold=threshold)


    def get_ranking_distances(self, VO_matrix, users, motivations_users):
        """ Get Kemeny distance between rankings from method_C and method_M for given users.
        """
        distances = []
        for id_eval in users:
            _, ranking_C = method_C(id_eval, VO_matrix, self.full_dataset.choices_df)
            _, ranking_M = method_M(id_eval, motivations_users)
            distance = kemenyd(ranking_C, ranking_M)
            distances.append((id_eval, distance))

        return distances


    def value_strategy(self):
        unlabeled_users = [int(user) for user in self.unlabeled_users]
        motivations_unlabeled_users = self.predict_motivations(unlabeled_users)

        VO_matrix = self.compute_VO_matrix(motivations_unlabeled_users=motivations_unlabeled_users)
        distances = self.get_ranking_distances(VO_matrix, unlabeled_users, motivations_unlabeled_users)

        sorted_distances = sorted(distances, key=lambda tup: tup[1])
        most_confused_users = [d[0] for d in sorted_distances[:self.sample_size]]
        self.update_user_labeled_indices(most_confused_users, update_labeled_indices=True)


    def get_value_profiles_distances(self, VO_matrix, users, motivations_users):
        """ Return Kemeny distance between predicted and ground truth value profiles.
        """
        distances = []
        for id_eval in users:
            _, ranking_predicted, _, _ = method_MO_MC_TB(
                id_eval, VO_matrix, self.full_dataset.choices_df, motivations_users)
            ranking_correct = self.correct_rankings.loc[id_eval].to_list()

            distances.append(kemenyd(ranking_predicted, ranking_correct))

        return distances


    def update_performance_results(self, test_dataset, num_fold):
        super().update_performance_results(test_dataset, num_fold)

        unlabeled_users = [int(user) for user in self.unlabeled_users]
        motivations_unlabeled_users = self.predict_motivations(unlabeled_users)
        test_users = [int(user) for user in self.test_users]
        motivations_test_users = self.predict_motivations(test_users)

        VO_matrix = self.compute_VO_matrix(
            motivations_unlabeled_users=motivations_unlabeled_users,
            motivations_test_users=motivations_test_users)

        # Save distance between predicted and ground truth value rankings for users in test set.
        profile_distances = self.get_value_profiles_distances(
            VO_matrix, test_users, motivations_test_users)
        self.performance_results[num_fold]['profiles'].append(np.mean(profile_distances))

        # Save distance between method_C and method_M for users in test set.
        confusion_distances = self.get_ranking_distances(
            VO_matrix, test_users, motivations_test_users)
        confusion_distances = [d[1] for d in confusion_distances]
        self.performance_results[num_fold]['confusion'].append(np.mean(confusion_distances))

        # Save percentage of users used for training in this iteration.
        self.performance_results[num_fold]['users'].append(
            100 * (len(self.labeled_users)) / (len(self.labeled_users) + len(self.unlabeled_users)))

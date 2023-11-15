import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import KFold
from .model import DEFAULT_CONFIG
from .utils import get_motivation_index_from_user_index, shuffle_indices


class PVEDataset(Dataset):
    def __init__(self,  motivations=None, label_names=None, texts=None, labels=None,
                 ids=None, max_size=None, language='dutch', shuffle=False):
        self.label_names = label_names
        self.language    = language
        self.ids         = ids
        self.text        = texts
        self.labels      = labels

        if motivations is not None:
            self.ids    = motivations.index.to_numpy(dtype=str)[:max_size]
            self.text   = motivations[language].to_list()[:max_size]
            self.labels = motivations[label_names].to_numpy(dtype=int)[:max_size]

        if shuffle:
            self.shuffle_data()

    def __getitem__(self, index):
        return {'id'    : self.ids[index],
                'text'  : self.text[index],
                'labels': self.labels[index]}

    def __len__(self):
        return len(self.data)
    
    def shuffle_data(self):
        indices = np.arange(self.ids.shape[0])
        np.random.shuffle(indices)
        self.ids    = self.ids[indices]
        self.text   = np.array(self.text)[indices].tolist()
        self.labels = self.labels[indices]


class BertDataset(PVEDataset):
    def __init__(self, motivations=None, label_names=None, texts=None, labels=None, shuffle=False,
                 ids=None, max_size=None, language='dutch', model_name=None, max_seq_length=64):
        super().__init__(motivations, label_names, texts, labels, ids, max_size, language, shuffle)

        self.model_name = model_name if model_name is not None else DEFAULT_CONFIG['dutch']

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.encodings = self.tokenizer(self.text, truncation=True, padding=True,
                                               max_length=max_seq_length, return_tensors='pt')
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        item['id'] = self.ids[idx]
        return item

    def __len__(self):
        return len(self.labels)

    def kfold(self, k=10, only_ids=False):
        kf = KFold(n_splits=k)

        for train_index, test_index in kf.split(X=self.text, y=self.labels):
            if only_ids:
                yield train_index, test_index
            else:
                train_dataset = self.make_dataset_from_indices(train_index)
                test_dataset  = self.make_dataset_from_indices(test_index)
                yield train_dataset, test_dataset

    def make_dataset_from_indices(self, indices):
        ids = self.ids[indices]
        X   = np.array(self.text)[indices]
        y   = self.labels[indices]

        return BertDataset(motivations=None, texts=X.tolist(), labels=y, ids=ids,
                           label_names=self.label_names, model_name=self.model_name)

    def to(self, device):
        self.encodings = self.encodings.to(device)
        self.labels = self.labels.to(device)


class ValueDataset(BertDataset):
    def __init__(self, motivations=None, choices=None, label_names=None, texts=None,
                 labels=None, shuffle=False, ids=None, max_size=None, language='dutch',
                 model_name=None, max_seq_length=64):
        super().__init__(motivations, label_names, texts, labels, shuffle, ids,
                         max_size, language, model_name, max_seq_length)

        self.make_questions_users_ids(choices, shuffle)
        self.motivations_df = motivations
        self.choices_df = choices

    def make_questions_users_ids(self, choices, shuffle):
        self.question_ids = list(sorted(set([idx.split('_')[0] for idx in self.ids])))
        self.user_ids     = np.array(choices.index.astype(str).to_list())

        # Check that user_ids in motivations and choices files coincide.
        motivation_ids = set([idx.split('_')[1] for idx in self.ids])
        motivation_ids.remove('118373') # not present in choices file
        assert motivation_ids == set(self.user_ids),\
            "Participant IDs in motivations and choices files do not coincide."

        if shuffle:
            self.user_ids = shuffle_indices(self.user_ids)


    def kfold(self, k=10, only_ids=False):
        """ Do K-fold with user IDs instead of motivation IDs.
        """
        kf = KFold(n_splits=k)

        # Split by user IDs, return the indices of texts and labels corresponding to the user IDs
        for user_train_index, user_test_index in kf.split(self.user_ids):
            user_ids_train = self.user_ids[user_train_index]
            user_ids_test  = self.user_ids[user_test_index]
            train_index = get_motivation_index_from_user_index(self.ids, user_ids_train)
            test_index  = get_motivation_index_from_user_index(self.ids, user_ids_test)
            
            if only_ids:
                yield train_index, test_index
            else:
                train_dataset = self.make_dataset_from_indices(train_index)
                test_dataset  = self.make_dataset_from_indices(test_index)
                yield train_dataset, test_dataset

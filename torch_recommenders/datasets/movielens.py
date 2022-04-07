import random
import numpy as np 
import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieLens_Dataset(Dataset):
    
    def __init__(self, items, targets):
        super(MovieLens_Dataset, self).__init__()
        self.items = items
        self.targets = targets

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]


class MovieLens20M_Data(object):
    """
    Preprocessing for MovieLens
    """
    
    def __init__(self, preprocessed, config):
        self.path = config['path']
        self.preprocessed = preprocessed
        self.preprocessed_path = config['preprocessed_path']
        self.min_rating = int(config['min_rating'])
        self.num_neg = int(config['num_neg'])
        self.num_neg_test = int(config['num_neg_test'])
        self.field_dims = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        
        self.__preprocessing()
        
    def _load_data(self, path, sep=',', engine='c', header='infer'):
        return pd.read_csv(path, sep=sep, engine=engine, header=header)
    
    def __load_preprocessed_data(self):
        train_dataset = torch.load(f'{self.preprocessed_path}train_ratings.pt')
        valid_dataset = torch.load(f'{self.preprocessed_path}valid_ratings.pt')
        test_dataset = torch.load(f'{self.preprocessed_path}test_ratings.pt')
        return train_dataset, valid_dataset, test_dataset
        
    def __save_preprocessed_data(self):
        torch.save(self.train_dataset, f'{self.preprocessed_path}train_ratings.pt')
        torch.save(self.valid_dataset, f'{self.preprocessed_path}valid_ratings.pt')
        torch.save(self.test_dataset, f'{self.preprocessed_path}test_ratings.pt')

    def __reindex(self, data):
        """
        Process dataset to reindex userId and movieId
        """
        user_list = list(data['userId'].unique())
        user2id = {userId: index for index, userId in enumerate(user_list)}

        item_list = list(data['movieId'].unique())
        item2id = {movieId: index for index, movieId in enumerate(item_list)}

        data['userId'] = data['userId'].apply(lambda x: user2id[x])
        data['movieId'] = data['movieId'].apply(lambda x: item2id[x])
        return data
    
    def __preprocess_target(self, data):
        """
        Set rating as binary feedback
        """
        data['rating'] = data['rating'].map(lambda x: 1 if x > self.min_rating else 0)
        return data

    def __leave_one_out(self, data):
        """
        leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
        """
        data['rank_latest'] = data.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = data.loc[data['rank_latest'] == 1]
        valid = data.loc[data['rank_latest'] == 2]
        train = data.loc[data['rank_latest'] > 2]
        assert train['userId'].nunique()==valid['userId'].nunique()==test['userId'].nunique(), 'Not Match Train User, Valid User with Test User'
        return train.iloc[:,:-1], valid.iloc[:,:-1], test.iloc[:,:-1]

    def __negative_sampling(self, data):
        interact_status = (
            data.groupby('userId')['movieId']
            .apply(set)
            .reset_index()
            .rename(columns={'movieId': 'interacted_items'}))
        item_pool = set(data['movieId'].unique())
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, self.num_neg_test))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def __get_train_dataset(self, data, negatives):
        items = np.zeros((len(data) * (self.num_neg + 1), 2), dtype=np.int32)
        targets = np.zeros((len(data) * (self.num_neg + 1), ), dtype=np.float32)
        data = pd.merge(data, negatives[['userId', 'negative_items']], on='userId')
        data['negatives'] = data['negative_items'].apply(lambda x: random.sample(x, self.num_neg))
        idx = 0
        for row in data.itertuples():
            items[idx][0] = row.userId
            items[idx][1] = row.movieId
            targets[idx] = row.rating
            idx += 1
            for movieId in row.negatives:
                items[idx][0] = row.userId
                items[idx][1] = movieId
                idx += 1
        dataset = MovieLens_Dataset(items=items, targets=targets)
        return dataset

    def __get_val_test_dataset(self, data, negatives):
        items = np.zeros((len(data) * (self.num_neg_test + 1), 2), dtype=np.int32)
        targets = np.zeros((len(data) * (self.num_neg_test + 1), ), dtype=np.float32)
        data = pd.merge(data, negatives[['userId', 'negative_samples']], on='userId')
        idx = 0
        for row in data.itertuples():
            items[idx][0] = row.userId
            items[idx][1] = row.movieId
            targets[idx] = row.rating
            idx += 1
            for movieId in row.negative_samples:
                items[idx][0] = row.userId
                items[idx][1] = movieId
                idx += 1
        dataset = MovieLens_Dataset(items=items, targets=targets)
        return dataset
    
    def __preprocessing(self): 
        if not self.preprocessed:
            data = self._load_data(self.path)
            data = self.__reindex(data)
            data = self.__preprocess_target(data)
            train_data, valid_data, test_data = self.__leave_one_out(data)
            negatives = self.__negative_sampling(data)
            self.train_dataset = self.__get_train_dataset(train_data, negatives)
            self.valid_dataset = self.__get_val_test_dataset(valid_data, negatives)
            self.test_dataset = self.__get_val_test_dataset(test_data, negatives)
            self.__save_preprocessed_data()
        else:
            self.train_dataset, self.valid_dataset, self.test_dataset = self.__load_preprocessed_data()

        self.field_dims = self.train_dataset.items.max(axis=0) + 1 # because ID begins from 0
    
    def get_data_loaders(self, batch_size, num_workers):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=True, 
                                                   num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(self.valid_dataset, 
                                                   batch_size=self.num_neg_test+1, 
                                                   shuffle=False, 
                                                   num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, 
                                                  batch_size=self.num_neg_test+1, 
                                                  shuffle=False, 
                                                  num_workers=num_workers)
        return train_loader, valid_loader, test_loader        
        

class MovieLens1M_Data(MovieLens20M_Data):
    """
    MovieLens 1M Dataset
    Data preparation
        treat samples with a rating less than 3 as negative samples
    :param dataset_path: MovieLens dataset path
    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, preprocessed, config):
        super().__init__(preprocessed, config)
        
    def _load_data(self, path, sep='::', engine='python', header=None):
        names = ['userId', 'movieId', 'rating', 'timestamp']
        return pd.read_csv(path, sep=sep, header=header, names=names ,engine=engine)
        
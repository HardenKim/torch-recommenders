import random
import numpy as np 
import pandas as pd
from collections import defaultdict
import torch
from torch.utils.data import Dataset


class MovieLens1M_Data(object):
    """
    Preprocessing for MovieLens
    """
    
    def __init__(self,
                 path='../data/ml-1m/ratings.dat',
                 min_rating=0,
                 negative_sampling=True,
                 num_neg=4,
                 num_neg_test=100
    ):
          
        self.path = path
        self.min_rating = min_rating
        self.negative_sampling = negative_sampling
        self.num_neg = num_neg
        self.num_neg_test = num_neg_test
        
        data = self._load_data()
        data = self._reindex(data)
        data = self._preprocess_target(data)        
        
        self.item_pool = data['movieId'].unique()
        self.num_users = data['userId'].nunique()
        self.num_movies = data['movieId'].nunique()
        self.field_dims = [self.num_users, self.num_movies]
        
        self.positive_items = self._get_positive_items(data)
        self.train_data, self.valid_data, self.test_data = self._leave_one_out(data)
    
    def _load_data(self, sep='::', engine='python', header=None):
        names = ['userId', 'movieId', 'rating', 'timestamp']
        return pd.read_csv(self.path, sep=sep, engine=engine, header=header, names=names)
        
    def _reindex(self, data):
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
    
    def _preprocess_target(self, data):
        """
        Set rating as binary feedback
        """
        data['rating'] = data['rating'].map(lambda x: 1 if x > self.min_rating else 0)
        return data
    
    def _leave_one_out(self, data):
        """
        leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
        """
        data['rank_latest'] = data.groupby(['userId'])['timestamp'].rank(method='first', ascending=False).astype(int)
        # train = data.loc[data['rank_latest'] > 2].sort_values(['userId', 'rank_latest'])
        # train.drop_duplicates(subset=['userId','movieId'], keep='first', inplace=True, ignore_index=True) # Deduplication
        train = data.loc[data['rank_latest'] > 2]
        valid = data.loc[data['rank_latest'] == 2]
        test = data.loc[data['rank_latest'] == 1]
        assert train['userId'].nunique()==valid['userId'].nunique()==test['userId'].nunique(), 'Not Match Train User, Valid User with Test User'
        return train.iloc[:,:3].to_numpy(), valid.iloc[:,:3].to_numpy(), test.iloc[:,:3].to_numpy()
        
    def _get_positive_items(self, data):
        positive_items = defaultdict(set)
        for row in data.itertuples():
            positive_items[row.userId].add(row.movieId)
        return positive_items
    
    def _get_dataset(self, data, num_neg, negative_sampling=True):
        return MovieLens_Dataset(data=data,
                                 item_pool=self.item_pool,
                                 positive_items=self.positive_items,
                                 negative_sampling=negative_sampling,
                                 num_neg=num_neg)
    
    def get_train_dataset(self):
        return self._get_dataset(self.train_data, self.num_neg, self.negative_sampling)
        
    def get_valid_dataset(self):
        return self._get_dataset(self.valid_data, self.num_neg_test, True)
        
    def get_test_dataset(self):
        return self._get_dataset(self.test_data, self.num_neg_test, True)
        

class MovieLens20M_Data(MovieLens1M_Data):
    
    def __init__(self,
                 path='../data/ml-20m/ratings.csv',
                 min_rating=0,
                 negative_sampling=False,
                 num_neg=4,
                 num_neg_test=100
    ):
        
        super().__init__(path, min_rating, negative_sampling, num_neg, num_neg_test)

    def _load_data(self, sep=',', engine='c', header='infer'):
        return pd.read_csv(self.path, sep=sep, engine=engine, header=header)    
    

class MovieLens_Dataset(Dataset):
    
    def __init__(self,
                 data,
                 item_pool,
                 positive_items,
                 negative_sampling=False,
                 num_neg=100,
                 ):
        
        self.data = data
        self.item_pool = item_pool
        self.positive_items = positive_items
        self.negative_sampling = negative_sampling
        self.num_neg = num_neg
        self.total_len = self._get_total_length()
        
    def _get_total_length(self):
        return len(self.data)
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        """
        Returns:
            items (2-D array): num_users * [userId, movieId]
            ratings (1-D array): num_users * [rating]
        """
        
        if not self.negative_sampling:
            return torch.LongTensor([self.data[index][0], self.data[index][1]]), torch.LongTensor(1)
        
        user = self.data[index][0]
        users = np.array([user] * (self.num_neg + 1), dtype=np.int32)
        movies = np.zeros(self.num_neg + 1, dtype=np.int32)
        ratings = np.zeros(self.num_neg + 1, dtype=np.float32)
        
        movies[0] = self.data[index][1]
        movies[1:] = random.sample(set(self.item_pool) - self.positive_items[user], self.num_neg)
        ratings[0] = 1
        
        items = np.c_[users, movies] # concatenate
    
        return torch.LongTensor(items), torch.LongTensor(ratings)

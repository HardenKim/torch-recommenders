import random
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset


class MovieLens20M_Dataset(Dataset):
    """
    Dataset for MovieLens
    Args:
        items (2-D array): num_users * [userId, movieId]
        ratings (1-D array): num_users * [rating]
    """
    
    def __init__(self,
                 path='../data/ml-20m/ratings.csv',
                 min_rating=0,
                 num_neg=4,
                 num_neg_test=100
    ):
          
        self.path = path
        self.min_rating = min_rating
        self.num_neg = num_neg
        self.num_neg_test = num_neg_test
        self.users = None
        self.movies = None
        self.ratings = None
        self.train_index = None
        self.valid_index = None
        self.test_index = None
        
        self._preprocessing()

        self.items = np.c_[self.users, self.movies] # concatenate
        self.field_dims = self.items.max(axis=0) + 1 # because ID begins from 0
        self.total_len = self._get_total_length()
    
    def _load_data(self, sep=',', engine='c', header='infer'):
        return pd.read_csv(self.path, sep=sep, engine=engine, header=header)
        
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
        train = data.loc[data['rank_latest'] > 2]
        valid = data.loc[data['rank_latest'] == 2]
        test = data.loc[data['rank_latest'] == 1]
        assert train['userId'].nunique()==valid['userId'].nunique()==test['userId'].nunique(), 'Not Match Train User, Valid User with Test User'
        return train.iloc[:,:3], valid.iloc[:,:3], test.iloc[:,:3]
    
    def _get_negative_items(self, data):
        interact_status = (
            data.groupby('userId')['movieId']
            .apply(set)
            .reset_index()
            .rename(columns={'movieId': 'interacted_items'}))
        item_pool = set(data['movieId'].unique())
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: item_pool - x)
        return interact_status[['userId', 'negative_items']]
    
    def _negative_sampling(self, data, negative_items, num_negatives):
        negative_sampling_len = len(data) * (num_negatives + 1)
        users = np.zeros((negative_sampling_len, ), dtype=np.int32)
        movies = np.zeros((negative_sampling_len, ), dtype=np.int32)
        ratings = np.zeros((negative_sampling_len, ), dtype=np.float32)
        data = pd.merge(data, negative_items, on='userId')
        data['negative_samples'] = data['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        idx = 0
        for row in data.itertuples():
            users[idx:idx+num_negatives+1] = row.userId
            movies[idx] = row.movieId
            movies[idx+1:idx+num_negatives+1] = row.negative_samples
            ratings[idx] = row.rating
            idx += num_negatives+1
        return users, movies, ratings
    
    def _preprocessing(self):
        data = self._load_data()
        data = self._reindex(data)
        data = self._preprocess_target(data)
        negative_items = self._get_negative_items(data)
        train_data, valid_data, test_data = self._leave_one_out(data)
        train_users, train_movies, train_ratings = self._negative_sampling(train_data, negative_items ,self.num_neg)
        valid_users, valid_movies, valid_ratings = self._negative_sampling(valid_data, negative_items,self.num_neg_test)
        test_users, test_movies, test_ratings = self._negative_sampling(test_data, negative_items, self.num_neg_test)
        
        self.train_index = list(range(len(train_users)))
        self.valid_index = list(range(self.train_index[-1] + 1, self.train_index[-1] + 1 + len(valid_users)))
        self.test_index = list(range(self.valid_index[-1] + 1, self.valid_index[-1] + 1 + len(test_users)))
        
        self.users = np.concatenate([train_users, valid_users, test_users]) 
        self.movies = np.concatenate([train_movies, valid_movies, test_movies])
        self.ratings = np.concatenate([train_ratings, valid_ratings, test_ratings])
        
    def _get_total_length(self):
        return len(self.users)
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        return self.items[index], self.ratings[index]
        
        
class MovieLens1M_Dataset(MovieLens20M_Dataset):
    """
    MovieLens 1M Dataset
    Data preparation
        treat samples with a rating less than 3 as negative samples
    :param dataset_path: MovieLens dataset path
    Reference:
        https://grouplens.org/datasets/movielens
    """
    
    def __init__(self,
                 path='../data/ml-1m/ratings.dat',
                 min_rating=0,
                 num_neg=4,
                 num_neg_test=100
    ):
        
        super().__init__(path, min_rating, num_neg, num_neg_test)

    def _load_data(self, sep='::', engine='python', header=None):
        names = ['userId', 'movieId', 'rating', 'timestamp']
        return pd.read_csv(self.path, sep=sep, engine=engine, header=header, names=names)
        
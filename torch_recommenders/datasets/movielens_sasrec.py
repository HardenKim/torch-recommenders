import random
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset

class MovieLens20M_Data(object):
    """
    Preprocessing for MovieLens
    """
    
    def __init__(self,
                 path='../data/ml-1m/ratings.dat',
                 min_rating=0,
                 max_len=50,
                 num_neg_test=100
                 ):
        
        self.path = path
        self.min_rating = min_rating
        self.max_len = max_len
        self.num_neg_test = num_neg_test
        self.users = None
        self.num_users = None
        self.num_movies = None
        self.negative_items = None
        self.train_seq = None
        self.valid_seq = None
        self.test_seq = None
        
        self._preprocessing()
    
    def _load_data(self, sep=',', engine='c', header='infer'):
        return pd.read_csv(self.path, sep=sep, engine=engine, header=header)

    def _reindex(selef, data):
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
        train = data.loc[data['rank_latest'] > 2].sort_values(['userId', 'rank_latest'])
        # train.drop_duplicates(subset=['userId','movieId'], keep='first', inplace=True, ignore_index=True) # Deduplication
        valid = data.loc[data['rank_latest'] == 2]
        test = data.loc[data['rank_latest'] == 1]
        assert train['userId'].nunique()==valid['userId'].nunique()==test['userId'].nunique(), 'Not Match Train User, Valid User with Test User'
        return train, valid, test

    def _get_negative_items(self, data):
        interact_status = (
            data.groupby('userId')['movieId']
            .apply(set)
            .reset_index()
            .rename(columns={'movieId': 'interacted_items'}))
        item_pool = set(data['movieId'].unique())
        return interact_status['interacted_items'].apply(lambda x: item_pool - x).to_dict()

    def _preprocessing(self): 
        data = self._load_data()
        data = self._reindex(data)
        data = self._preprocess_target(data)
        train_data, valid_data, test_data = self._leave_one_out(data)
        
        self.users = data['userId'].unique()
        self.num_users = data['userId'].nunique()
        self.num_movies = data['movieId'].nunique()
        self.negative_items = self._get_negative_items(data)
        self.train_seq = train_data.groupby('userId')['movieId'].apply(np.array).to_dict()
        self.valid_seq = valid_data.groupby('userId')['movieId'].apply(np.array).to_dict()
        self.test_seq = test_data.groupby('userId')['movieId'].apply(np.array).to_dict()
        
    def get_train_dataset(self):
        return MovieLens_Train_Dataset(users=self.users,
                                       num_users=self.num_users,
                                       num_movies=self.num_movies,
                                       negative_items=self.negative_items,
                                       train_sequences=self.train_seq,
                                       max_len=self.max_len
        )
        
    def get_valid_dataset(self):
        return MovieLens_Valid_Dataset(users=self.users,
                                       num_users=self.num_users,
                                       num_movies=self.num_movies,
                                       negative_items=self.negative_items,
                                       train_sequences=self.train_seq,
                                       valid_sequences=self.valid_seq,
                                       max_len=self.max_len,
                                       num_neg_test=self.num_neg_test
        )
    
    def get_test_dataset(self):
        return MovieLens_Test_Dataset(users=self.users,
                                      num_users=self.num_users,
                                      num_movies=self.num_movies,
                                      negative_items=self.negative_items,
                                      train_sequences=self.train_seq,
                                      valid_sequences=self.valid_seq,
                                      test_sequences=self.test_seq,
                                      max_len=self.max_len,
                                      num_neg_test=self.num_neg_test
        )
         
             
class MovieLens1M_Data(MovieLens20M_Data):
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
                 max_len=50,
                 num_neg_test=100
                 ):
        super().__init__(path, min_rating, max_len, num_neg_test)

    def _load_data(self, sep='::', engine='python', header=None):
        names = ['userId', 'movieId', 'rating', 'timestamp']
        return pd.read_csv(self.path, sep=sep, engine=engine, header=header, names=names)        


class MovieLens_Train_Dataset(Dataset):
    
    def __init__(self,
                 users,
                 num_users,
                 num_movies,
                 negative_items,
                 train_sequences,
                 valid_sequences=None,
                 test_sequences=None,
                 max_len=50,
                 num_neg_test=100
                 ):
        
        self.users = users
        self.num_users = num_users
        self.num_movies = num_movies
        self.negative_items = negative_items
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences        
        self.test_sequences = test_sequences
        self.max_len = max_len
        self.num_neg_test = num_neg_test
        self.total_len = self._get_total_length()
        
    def _get_total_length(self):
        return len(self.users)
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        """
        Returns:
            user: user
            seq: sequence excluding the last item
            pos: next items of seq
            neg: negative items
        """
        
        user = self.users[index]
        sequence = self.train_sequences[user][::-1]
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        
        if len(sequence) >= self.max_len + 1:
            seq[:] = sequence[-self.max_len-1:-1]
            pos[:] = sequence[-self.max_len:]
        else:
            seq[-len(sequence)+1:] = sequence[:-1]
            pos[-len(sequence)+1:] = sequence[1:]
            
        neg[:] = random.sample(self.negative_items[user], self.max_len)
        
        return user, seq, pos, neg
        
        
class MovieLens_Valid_Dataset(MovieLens_Train_Dataset):
    
    def __init__(self,
                 users,
                 num_users,
                 num_movies,
                 negative_items,
                 train_sequences,
                 valid_sequences=None,
                 test_sequences=None,
                 max_len=50,
                 num_neg_test=100
                 ):
        super().__init__(users, num_users, num_movies ,negative_items, train_sequences, valid_sequences, test_sequences, max_len, num_neg_test)

    def __getitem__(self, index):
        user = self.users[index]
        sequence = self.train_sequences[user][::-1]
        seq = np.zeros([self.max_len], dtype=np.int32)
        item_idx = np.zeros([self.num_neg_test + 1], dtype=np.int32)
        
        if len(sequence) >= self.max_len:
                seq[:] = sequence[-self.max_len:]
        else:
            seq[-len(sequence):] = sequence[:]
            
        item_idx[0] = self.valid_sequences[user][0]
        item_idx[1:] = random.sample(self.negative_items[user], self.num_neg_test)
        
        return user, seq, item_idx

        
class MovieLens_Test_Dataset(MovieLens_Train_Dataset):
    
    def __init__(self,
                 users,
                 num_users,
                 num_movies,
                 negative_items,
                 train_sequences,
                 valid_sequences=None,
                 test_sequences=None,
                 max_len=50,
                 num_neg_test=100
                 ):
        super().__init__(users, num_users, num_movies, negative_items, train_sequences, valid_sequences, test_sequences, max_len, num_neg_test)

    def __getitem__(self, index):
        user = self.users[index]
        sequence = self.train_sequences[user][::-1]
        seq = np.zeros([self.max_len], dtype=np.int32)
        item_idx = np.zeros([self.num_neg_test + 1], dtype=np.int32)
        
        seq[-1] = self.valid_sequences[user][0]
        if len(sequence) >= self.max_len:
                seq[:-1] = sequence[-self.max_len+1:]
        else:
            seq[-len(sequence)-1:-1] = sequence[:]
    
        item_idx[0] = self.test_sequences[user][0]
        item_idx[1:] = random.sample(self.negative_items[user], self.num_neg_test)
        
        return user, seq, item_idx
        
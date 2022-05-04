import random
import numpy as np 
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset


class MovieLens1M_Data(object):
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
        
        data = self._load_data()
        data = self._reindex(data)
        data = self._preprocess_target(data)
        
        self.users = data['userId'].unique()
        self.movies = data['movieId'].unique()
        self.num_users = data['userId'].nunique()
        self.num_movies = data['movieId'].nunique()
        self.positive_items = self._get_positive_items(data) # include valid, test data 
        self.train_seq, self.valid_seq, self.test_seq = self._get_sequences(data)

    def _load_data(self, sep='::', engine='python', header=None):
        names = ['userId', 'movieId', 'rating', 'timestamp']
        return pd.read_csv(self.path, sep=sep, engine=engine, header=header, names=names)  
    
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
        return train.iloc[:,:2], valid.iloc[:,:2], test.iloc[:,:2]
    
    def _get_sequences(self, data):
        train_seq = defaultdict(list)
        valid_seq = defaultdict(list)
        test_seq = defaultdict(list)
        
        train_data, valid_data, test_data = self._leave_one_out(data)
        
        for row in train_data.itertuples():
            train_seq[row.userId].append(row.movieId)
        for row in valid_data.itertuples():
            valid_seq[row.userId].append(row.movieId)
        for row in test_data.itertuples():
            test_seq[row.userId].append(row.movieId)
        
        return train_seq, valid_seq, test_seq
    
    def _get_positive_items(self, data):
        positive_items = defaultdict(set)
        for row in data.itertuples():
            positive_items[row.userId].add(row.movieId)
        return positive_items
    
    def get_train_dataset(self):
        return Train_Dataset(users=self.users,
                             movies=self.movies,
                             positive_items=self.positive_items,
                             train_sequences=self.train_seq,
                             max_len=self.max_len
        )
        
    def get_valid_dataset(self):
        return Valid_Dataset(users=self.users,
                             movies=self.movies,
                             positive_items=self.positive_items,
                             train_sequences=self.train_seq,
                             valid_sequences=self.valid_seq,
                             max_len=self.max_len,
                             num_neg_test=self.num_neg_test
        )
    
    def get_test_dataset(self):
        return Test_Dataset(users=self.users,
                            movies=self.movies,
                            positive_items=self.positive_items,
                            train_sequences=self.train_seq,
                            valid_sequences=self.valid_seq,
                            test_sequences=self.test_seq,
                            max_len=self.max_len,
                            num_neg_test=self.num_neg_test
        )
         
             
class MovieLens20M_Data(MovieLens1M_Data):
    
    def __init__(self,
                 path='../data/ml-20m/ratings.csv',
                 min_rating=0,
                 max_len=50,
                 num_neg_test=100
                 ):
        super().__init__(path, min_rating, max_len, num_neg_test)
    
    def _load_data(self, sep=',', engine='c', header='infer'):
        return pd.read_csv(self.path, sep=sep, engine=engine, header=header)      


class Train_Dataset(Dataset):
    
    def __init__(self,
                 users,
                 movies,
                 positive_items,
                 train_sequences,
                 valid_sequences=None,
                 test_sequences=None,
                 max_len=50,
                 num_neg_test=100
                 ):
        
        self.users = users
        self.movies = movies
        self.positive_items = positive_items
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
            
        neg[:] = random.sample(set(self.movies) - self.positive_items[user], self.max_len)
        
        return user, seq, pos, neg
        
        
class Valid_Dataset(Train_Dataset):
    
    def __init__(self,
                 users,
                 movies,
                 positive_items,
                 train_sequences,
                 valid_sequences=None,
                 test_sequences=None,
                 max_len=50,
                 num_neg_test=100
                 ):
        super().__init__(users, movies ,positive_items, train_sequences, valid_sequences, test_sequences, max_len, num_neg_test)

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
        item_idx[1:] = random.sample(set(self.movies) - self.positive_items[user], self.num_neg_test)
        
        
        return user, seq, item_idx

        
class Test_Dataset(Train_Dataset):
    
    def __init__(self,
                 users,
                 movies,
                 positive_items,
                 train_sequences,
                 valid_sequences=None,
                 test_sequences=None,
                 max_len=50,
                 num_neg_test=100
                 ):
        super().__init__(users, movies, positive_items, train_sequences, valid_sequences, test_sequences, max_len, num_neg_test)

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
        item_idx[1:] = random.sample(set(self.movies) - self.positive_items[user], self.num_neg_test)
        
        return user, seq, item_idx
        
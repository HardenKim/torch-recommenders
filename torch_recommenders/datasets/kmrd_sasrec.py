import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
from datasets.movielens_sasrec import MovieLens1M_Data


class KMRD2M_Data(MovieLens1M_Data):
    
    def __init__(self,
                 path='../data/kmrd/rates-2m.csv',
                 min_rating=0,
                 max_len=50,
                 num_neg_test=100
                 ):
        super().__init__(path, min_rating, max_len, num_neg_test)

    def _load_data(self, sep=',', engine='c', header='infer'):
        data = pd.read_csv(self.path, sep=sep, engine=engine, header=header)
        data.columns =['userId', 'movieId', 'rating', 'timestamp']
        return data
    
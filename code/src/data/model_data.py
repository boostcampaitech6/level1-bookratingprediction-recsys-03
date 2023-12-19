import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import re
from tqdm import tqdm
from src.data import process_context_data
from src.data import process_text_data


def hybrid_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    
    #NCF 
    field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)

    #process text에서 문제
    text_train = process_text_data(train, books, user2idx, isbn2idx, args.device, train=True, user_summary_merge_vector=args.vector_create, item_summary_vector=args.vector_create)
    text_test = process_text_data(test, books, user2idx, isbn2idx, args.device, train=False, user_summary_merge_vector=args.vector_create, item_summary_vector=args.vector_create)
    

    data = {
            'train':train,
            'test':test,  #drop rating?
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'text_train':text_train,
            'text_test':text_test,
            }


    return data

def model_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """
    #각 모델에 들어갈 데이터 분리
    # 모델에 맞게 필요없는 feature 삭제
    upper_30_train = int(len(data['train']['user_id'].unique())*0.3)  #user_id의 30%
    upper_30_test = int(len(data['test']['user_id'].unique())*0.3)
    cold_start_train = data['train'].groupby('user_id')['rating'].count().sort_values(ascending=True).head(upper_30_train).index
    cold_start_test = data['test'].groupby('user_id')['rating'].count().sort_values(ascending=True).head(upper_30_test).index
    
    data_cold = data.copy()
    data_not_cold = data.copy()

    data_cold['train'] = data_cold['train'][(data_cold['train']['user_id'].isin(cold_start_train))]
    data_cold['test'] = data_cold['test'][(data_cold['test']['user_id'].isin(cold_start_test))]
    
    data_not_cold['train'] = data_not_cold['train'][~data_not_cold['train']['user_id'].isin(cold_start_train)]
    data_not_cold['test'] = data_not_cold['test'][~data_not_cold['test']['user_id'].isin(cold_start_train)]

    #불필요한 항목 삭제
    #data_cold: LSTM / data_not_cold: NCF #delete item
    #idx는 train의 영향을 받음 <- 나중에 오류생길수도...일단 킵고잉
    del data_cold['field_dims']
    del data_not_cold['text_train']
    del data_not_cold['text_test']
    
    data_not_cold['test'] = data_not_cold['test'].drop(['rating'], axis=1)

    return data_cold, data_not_cold
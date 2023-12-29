import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.cluster import KMeans

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def my_data_cluster(args, index_data) -> pd.DataFrame:
    """K-Means 클러스터 컬럼 추가
    Args:
        args (_type_): arguments
        index_data (_type_): Indexed pd.DataFrame 

    Returns:
        pd.DataFrame
    """    
    data = index_data

    # KMeans 모델 생성 및 학습
    kmeans = KMeans(n_clusters=args.cluster, random_state=0) # n_clusters는 클러스터의 개수
    clusters = kmeans.fit_predict(data)

    # 클러스터 결과를 데이터프레임에 추가
    data['cluster'] = clusters

    # 결과 출력
    return data

def my_data_rating(index_data) -> pd.DataFrame:
    def group(x):
        if x <= 5:
            return 0
        elif x > 5 & x <= 7:
            return 1
        elif x > 7:
            return 2
        
    data = index_data
    data['rating_group'] = data['rating']
    data['rating_group'] = data['rating_group'].apply(group)

    # 결과 출력
    return data

def process_context_data(users, books, ratings1, ratings2):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    """

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[-1])
    users = users.drop(['location'], axis=1)
    
    # location_country가 없는 데이터는 location_city가 있는 location_country으로 넣어주기
    # 결측치 3% -> 0.4%
    city_country_list = list(users[users['location_city'].notnull() & users['location_country'].notnull()][['location_city', 'location_country']].value_counts().index)
    city_country_dic = {city:value for city, value in city_country_list}
    users['location_country'] = users['location_city'].apply(lambda x: city_country_dic.get(x))

    # category 총 4천개 중 동일 category 상위 101개(34개 중복) category 는 88% 비율을 차지함
    import re
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    category_count = books['category'].value_counts().reset_index()
    category_list = list(category_count[category_count['count']>=34]['category'])
    books['category'] = books['category'].apply(lambda x: x if x in category_list else 'other')

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)
    
    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'book_title', 'year_of_publication']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'book_title', 'year_of_publication']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'book_title', 'year_of_publication']], on='isbn', how='left')

    # location 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}
    
    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    booktitle2idx = {v:k for k,v in enumerate(context_df['book_title'].unique())}
    yearofpublication2idx = {v:k for k,v in enumerate(context_df['year_of_publication'].unique())}
    
    # 나이 결측치 처리
    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)
    
    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    train_df['book_title'] = train_df['book_title'].map(booktitle2idx)
    train_df['year_of_publication'] = train_df['year_of_publication'].map(yearofpublication2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)
    test_df['book_title'] = test_df['book_title'].map(booktitle2idx)
    test_df['year_of_publication'] = test_df['year_of_publication'].map(yearofpublication2idx)
    
    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
        "booktitle2idx":booktitle2idx,
        "yearofpublication2idx":yearofpublication2idx
    }

    return idx, train_df, test_df

def process_context_data_2(args, users, books, ratings1, ratings2):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    """

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[-1])
    users = users.drop(['location'], axis=1)
    
    # location_country가 없는 데이터는 location_city가 있는 location_country으로 넣어주기
    # 결측치 3% -> 0.4%
    city_country_list = list(users[users['location_city'].notnull() & users['location_country'].notnull()][['location_city', 'location_country']].value_counts().index)
    city_country_dic = {city:value for city, value in city_country_list}
    users['location_country'] = users['location_city'].apply(lambda x: city_country_dic.get(x))

    # category 총 4천개 중 동일 category 상위 101개(34개 이상 중복) category 는 88% 비율을 차지함
    import re
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    category_count = books['category'].value_counts().reset_index()
    category_list = list(category_count[category_count['count']>=34]['category'])
    books['category'] = books['category'].apply(lambda x: x if x in category_list else 'other')

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'book_title', 'year_of_publication']], on='isbn', how='left')

    # location 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}
    # 나이 결측치 처리
    context_df['age'] = context_df['age'].fillna(int(context_df['age'].mean()))
    context_df['age'] = context_df['age'].apply(age_map)
    context_df['location_city'] = context_df['location_city'].map(loc_city2idx)
    context_df['location_state'] = context_df['location_state'].map(loc_state2idx)
    context_df['location_country'] = context_df['location_country'].map(loc_country2idx)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    booktitle2idx = {v:k for k,v in enumerate(context_df['book_title'].unique())}
    yearofpublication2idx = {v:k for k,v in enumerate(context_df['year_of_publication'].unique())}
    context_df['category'] = context_df['category'].map(category2idx)
    context_df['publisher'] = context_df['publisher'].map(publisher2idx)
    context_df['language'] = context_df['language'].map(language2idx)
    context_df['book_author'] = context_df['book_author'].map(author2idx)
    context_df['book_title'] = context_df['book_title'].map(booktitle2idx)
    context_df['year_of_publication'] = context_df['year_of_publication'].map(yearofpublication2idx)
    
    # context_df = my_data_cluster(args, context_df)
    # 'age', 'location_city', 'location_state', 'location_country', 'category', 'publisher', 'language', 'book_author', 'book_title', 'year_of_publication'
    # 'age', 'location_city', 'location_state', 'publisher', 'book_author'
    # context_df = context_df.drop(['rating'], axis=1)
    context_df = context_df.drop(['rating', 'location_country', 'category', 'language', 'book_title', 'year_of_publication'], axis=1)
    

    train_df = ratings1.merge(context_df, how='left', on=['user_id', 'isbn'])
    # train_df = pd.concat([train_df, train_df[train_df['rating']<=5]], axis=0)
    # train_df = pd.concat([train_df, train_df[train_df['rating']<=5].sample(frac=0.1)], axis=0)
    test_df = ratings2.merge(context_df, how='left', on=['user_id', 'isbn'])
    
    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
        "booktitle2idx":booktitle2idx,
        "yearofpublication2idx":yearofpublication2idx
    }

    return idx, train_df, test_df


def my_data_load(args):
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
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    # idx, context_train, context_test = process_context_data(users, books, train, test)
    idx, context_train, context_test = process_context_data_2(args, users, books, train, test)
    
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, len(idx['loc_city2idx']), len(idx['loc_state2idx']),len(idx['publisher2idx']), len(idx['author2idx'])
                            ], dtype=np.uint32)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data


def my_data_split(args, data):
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
    
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    
    return data

def my_data_kfold(args, data, n):
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
    
    idx, n = 0, n
    pd_split = int(data['train'].shape[0] / n)
    
    valid_data_list = []
    for i in range(n):
        valid_data_list.append(data['train'].iloc[idx:idx + pd_split])
        idx += pd_split
    
    data['X_train'] = [data['train'].drop(valid_data_list[i].index.to_list(), axis=0).drop(['rating'], axis=1) for i in range(n)]
    data['y_train'] = [data['train'].drop(valid_data_list[i].index.to_list(), axis=0)['rating'] for i in range(n)]
    data['X_valid'] = [valid_data_list[i].drop(['rating'], axis=1) for i in range(n)]
    data['y_valid'] = [valid_data_list[i]['rating'] for i in range(n)]
    
    return data

def my_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data

def kfold_loader(args, data, n):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    train_dataset = [TensorDataset(torch.LongTensor(data['X_train'][i].values), torch.LongTensor(data['y_train'][i].values)) for i in range(n)]
    valid_dataset = [TensorDataset(torch.LongTensor(data['X_valid'][i].values), torch.LongTensor(data['y_valid'][i].values)) for i in range(n)]
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = [DataLoader(train_dataset[i], batch_size=args.batch_size, shuffle=args.data_shuffle) for i in range(n)]
    valid_dataloader = [DataLoader(valid_dataset[i], batch_size=args.batch_size, shuffle=args.data_shuffle) for i in range(n)]
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
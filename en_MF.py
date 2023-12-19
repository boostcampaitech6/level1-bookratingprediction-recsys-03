import pandas as pd 
import numpy as np
from surprise import Dataset,Reader
from sklearn.preprocessing import OrdinalEncoder
from surprise import SVD

path = '/data/ephemeral/home/workplace/code/data/'
books = pd.read_csv(path + "books.csv")
users = pd.read_csv(path + "users.csv")
ratings = pd.read_csv(path + "train_ratings.csv")

encoder = OrdinalEncoder()
ratings['isbn'] = encoder.fit_transform(ratings[['isbn']])
# reader = Reader(rating_scale=(1,10))
# data =Dataset.load_from_df(ratings,reader)

# MF를 사용하여 rating 개수로 두 집단으로 나눠 따로 학습을 시킨 뒤 예측하기

cnt = ratings.user_id.value_counts()

ratings['cnt'] = ratings['user_id'].map(cnt)

ratings.sort_values('cnt', ascending=False, inplace=True)

hot_df = ratings.iloc[:234003,:]
cold_df = ratings.iloc[234003:,:]

def pre(df):
    df.drop(['cnt'],axis=1,inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

hot_df = pre(hot_df)
cold_df = pre(cold_df)

reader = Reader(rating_scale=(1,10))

h_data = Dataset.load_from_df(hot_df, reader)
h_data = h_data.build_full_trainset()

c_data = Dataset.load_from_df(cold_df, reader)
c_data = c_data.build_full_trainset()


h_model = SVD(random_state=42)

c_model = SVD(random_state=42)

h_model.fit(h_data)
c_model.fit(c_data)

submit = pd.read_csv(path + 'sample_submission.csv')
test = pd.read_csv(path + 'test_ratings.csv')

test['isbn'] = encoder.fit_transform(test[['isbn']])

for i in range(len(submit)):
    if list(ratings[ratings['user_id'] == submit.iloc[i,0]]['cnt']):
        if list(ratings[ratings['user_id'] == submit.iloc[i,0]]['cnt'])[0] >=5:
            submit['rating'][i] = h_model.predict(test['user_id'][i], test['isbn'][i]).est
        else:
            submit['rating'][i] = c_model.predict(test['user_id'][i], test['isbn'][i]).est    
    else:
        # 데이터가 아예 없는 경우
        submit['rating'][i] = c_model.predict(test['user_id'][i], test['isbn'][i]).est


submit.to_csv('submit.csv',index=False)


# submit 8번 no data
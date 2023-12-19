import pandas as pd 
import numpy as np
from surprise import Dataset,Reader
from sklearn.preprocessing import LabelEncoder
from surprise import SVD

path = '/data/ephemeral/home/workplace/code/data/'
books = pd.read_csv(path + "books.csv")
users = pd.read_csv(path + "users.csv")
ratings = pd.read_csv(path + "train_ratings.csv")

encoder = LabelEncoder()
ratings['isbn'] = encoder.fit_transform(ratings['isbn'])

reader = Reader(rating_scale=(1,10))
data =Dataset.load_from_df(ratings,reader)
data = data.build_full_trainset()


model = SVD(random_state = 42)

model.fit(data)

submit = pd.read_csv(path + 'sample_submission.csv')
test = pd.read_csv(path + 'test_ratings.csv')

submit['rating'] = test.apply(lambda x: model.predict(x['user_id'],x['isbn']).est, axis=1)

submit.to_csv('mf_submit.csv',index=False)
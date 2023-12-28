import numpy as np
import pandas as pd
from tqdm import tqdm

class Ensemble:
    '''
    [description]
    앙상블을 진행하는 클래스입니다.

    [parameter]
    filenames: 앙상블을 진행할 모델의 이름을 리스트 형태로 입력합니다.
    filepath: 앙상블을 진행할 모델의 csv 파일이 저장된 경로를 입력합니다.
    '''
    def __init__(self, filenames:str, filepath:str):
        self.filenames = filenames
        self.output_list = []

        output_path = [filepath+filename+'.csv' for filename in filenames]
        self.output_frame = pd.read_csv(output_path[0]).drop('rating',axis=1)
        self.output_df = self.output_frame.copy()

        for path in output_path:
            self.output_list.append(pd.read_csv(path)['rating'].to_list())
        for filename,output in zip(filenames,self.output_list):
            self.output_df[filename] = output


    def simple_weighted(self,weight:list):
        '''
        [description]
        직접 weight를 지정하여, 앙상블합니다.
        
        [parameter]
        weight: 각 모델의 weight를 리스트 형태로 입력합니다.
        이 때, weight의 합은 1이 되도록 입력해 주세요.
        
        [return]
        result: 앙상블 결과를 리스트 형태로 반환합니다.
        '''
        if not len(self.output_list)==len(weight):
            raise ValueError("model과 weight의 길이가 일치하지 않습니다.")
        if np.sum(weight)!=1:
            raise ValueError("weight의 합이 1이 되도록 입력해 주세요.")

        pred_arr = np.append([self.output_list[0]], [self.output_list[1]], axis=0)
        for i in range(2, len(self.output_list)):
            pred_arr = np.append(pred_arr, [self.output_list[i]], axis=0)
        result = np.dot(pred_arr.T, np.array(weight))
        return result.tolist()


    def average_weighted(self):
        '''
        [description]
        (1/n)의 가중치로 앙상블을 진행합니다.
        
        [return]
        result: 앙상블 결과를 리스트 형태로 반환합니다.
        '''
        weight = [1/len(self.output_list) for _ in range(len(self.output_list))]
        pred_weight_list = [pred*np.array(w) for pred, w in zip(self.output_list,weight)]
        result = np.sum(pred_weight_list, axis=0)
        return [min(10,x) for x in result.tolist()]


    def mixed(self):
        '''
        [description]
        Negative case 발생 시, 다음 순서에서 예측한 rating으로 넘어가서 앙상블합니다.
        
        [return]
        result: 앙상블 결과를 리스트 형태로 반환합니다.
        '''
        result = self.output_df[self.filenames[0]].copy()
        for idx in range(len(self.filenames)-1):
            pre_idx = self.filenames[idx]
            post_idx = self.filenames[idx+1]
            result[self.output_df[pre_idx]<1] = self.output_df.loc[self.output_df[pre_idx]<1,post_idx]
        return result.tolist()

    def custom(self):
        '''
        FM과 CF 앙상블.
        FM의 가중치: 1/log(1 + (# of ratings of user + 1) + (# of ratings of book + 1))
        CF의 가중치: 1 - FM의 가중치
        '''
        
        ratings = pd.read_csv('./data/train_ratings.csv')
        user_counts = ratings.groupby('user_id')['rating'].count().reset_index()
        book_counts = ratings.groupby('isbn')['rating'].count().reset_index()

        test = pd.read_csv('./data/test_ratings.csv')

        result = []
        for user_id, isbn, FM, CF in tqdm(zip(test['user_id'], test['isbn'], self.output_list[0], self.output_list[1])):
            user_count = user_counts[user_counts['user_id']==user_id]['rating'].values[0] if user_id in user_counts['user_id'].values else 0
            book_count = book_counts[book_counts['isbn'] == isbn]['rating'].values[0] if isbn in book_counts['isbn'].values else 0
            
            w = np.log(3) / np.log(1 + (user_count + 1) + (book_count + 1))

            result.append(w*FM + (1-w)*CF)

        return result
    
    def custom2(self):
        '''
        FM과 CF 앙상블.
        FM의 가중치: 1/log(1 + (# of ratings of user + 1) * (# of ratings of book + 1))
        CF의 가중치: 1 - FM의 가중치
        '''
        
        ratings = pd.read_csv('./data/train_ratings.csv')
        user_counts = ratings.groupby('user_id')['rating'].count().reset_index()
        book_counts = ratings.groupby('isbn')['rating'].count().reset_index()

        test = pd.read_csv('./data/test_ratings.csv')

        result = []
        for user_id, isbn, FM, CF in tqdm(zip(test['user_id'], test['isbn'], self.output_list[0], self.output_list[1])):
            user_count = user_counts[user_counts['user_id']==user_id]['rating'].values[0] if user_id in user_counts['user_id'].values else 0
            book_count = book_counts[book_counts['isbn'] == isbn]['rating'].values[0] if isbn in book_counts['isbn'].values else 0
            
            w = np.log(2) / np.log(1 + (user_count + 1) * (book_count + 1))

            result.append(w*FM + (1-w)*CF)

        return result
    
    def custom3(self):
        '''
        FM과 CF 앙상블.
        FM의 가중치: 1/log(1 + (# of ratings of user + 1) * (# of ratings of book * 5 + 1))
        CF의 가중치: 1 - FM의 가중치
        '''
        
        ratings = pd.read_csv('./data/train_ratings.csv')
        user_counts = ratings.groupby('user_id')['rating'].count().reset_index()
        book_counts = ratings.groupby('isbn')['rating'].count().reset_index()

        test = pd.read_csv('./data/test_ratings.csv')

        result = []
        for user_id, isbn, FM, CF in tqdm(zip(test['user_id'], test['isbn'], self.output_list[0], self.output_list[1])):
            user_count = user_counts[user_counts['user_id']==user_id]['rating'].values[0] if user_id in user_counts['user_id'].values else 0
            book_count = book_counts[book_counts['isbn'] == isbn]['rating'].values[0] if isbn in book_counts['isbn'].values else 0
            
            w = np.log(2) / np.log(1 + (user_count + 1) * (book_count * 5 + 1))

            result.append(w*FM + (1-w)*CF)

        return result
    
    def custom4(self):
        '''
        FM과 CF 앙상블.
        FM: (# of ratings of user) <= 5 and (# of ratings of book) <= 5
        CF: 그 외
        '''
        
        ratings = pd.read_csv('./data/train_ratings.csv')
        user_counts = ratings.groupby('user_id')['rating'].count().reset_index()
        book_counts = ratings.groupby('isbn')['rating'].count().reset_index()

        test = pd.read_csv('./data/test_ratings.csv')

        result = []
        for user_id, isbn, FM, CF in tqdm(zip(test['user_id'], test['isbn'], self.output_list[0], self.output_list[1])):
            user_count = user_counts[user_counts['user_id']==user_id]['rating'].values[0] if user_id in user_counts['user_id'].values else 0
            book_count = book_counts[book_counts['isbn'] == isbn]['rating'].values[0] if isbn in book_counts['isbn'].values else 0
            
            if user_count <= 5 or book_count <= 5:
                result.append(FM)
            else:
                result.append(CF)

        return result
    
    def custom5(self):
        '''
        FM과 CF 앙상블.
        FM: (# of ratings of book) > 5
        CF: 그 외
        '''
        
        ratings = pd.read_csv('./data/train_ratings.csv')
        #user_counts = ratings.groupby('user_id')['rating'].count().reset_index()
        book_counts = ratings.groupby('isbn')['rating'].count().reset_index()

        test = pd.read_csv('./data/test_ratings.csv')

        result = []
        for user_id, isbn, FM, CF in tqdm(zip(test['user_id'], test['isbn'], self.output_list[0], self.output_list[1])):
            #user_count = user_counts[user_counts['user_id']==user_id]['rating'].values[0] if user_id in user_counts['user_id'].values else 0
            book_count = book_counts[book_counts['isbn'] == isbn]['rating'].values[0] if isbn in book_counts['isbn'].values else 0
            
            if book_count > 5:
                output = FM
            else:
                output = CF
            
            if output > 10:
                result.append(10)
            elif output < 0:
                result.append(0)
            else:
                result.append(output)

        return result
    
    def custom6(self):
        '''
        FM과 CF 앙상블.
        FM: CF 예측 4 미만 / LSTM
        CF: 그 외 (FM과 평균) / MF
        '''
        
        result = []
        for FM, CF in tqdm(zip(self.output_list[0], self.output_list[1])):
            if CF < 4:
                output = FM
            else:
                output = (FM+CF)/2

            if output > 10:
                result.append(10)
            elif output < 0:
                result.append(0)
            else:
                result.append(output)

        return result
    
    def custom7(self):
        '''
        FM과 CF 앙상블.
        FM: 1 - CF의 가중치
        CF: log(2) / log(1 + (# of book ratings + 1))
        '''
        
        ratings = pd.read_csv('./data/train_ratings.csv')
        #user_counts = ratings.groupby('user_id')['rating'].count().reset_index()
        book_counts = ratings.groupby('isbn')['rating'].count().reset_index()

        test = pd.read_csv('./data/test_ratings.csv')

        result = []
        for user_id, isbn, FM, CF in tqdm(zip(test['user_id'], test['isbn'], self.output_list[0], self.output_list[1])):
            #user_count = user_counts[user_counts['user_id']==user_id]['rating'].values[0] if user_id in user_counts['user_id'].values else 0
            book_count = book_counts[book_counts['isbn'] == isbn]['rating'].values[0] if isbn in book_counts['isbn'].values else 0
            
            w = np.log(2) / np.log(1 + (book_count + 1))

            result.append((1-w)*FM + w*CF)

        return result
    
    def custom8(self):
        ratings = pd.read_csv('./data/train_ratings.csv')
        user_counts = ratings.groupby('user_id')['rating'].count().reset_index()
        book_counts = ratings.groupby('isbn')['rating'].count().reset_index()

        test = pd.read_csv('./data/test_ratings.csv')

        mu = ratings['rating'].mean()

        result = []
        for user_id, isbn, LSTM, MF in tqdm(zip(test['user_id'], test['isbn'], self.output_list[0], self.output_list[1])):
            user_count = user_counts[user_counts['user_id']==user_id]['rating'].values[0] if user_id in user_counts['user_id'].values else 0
            book_count = book_counts[book_counts['isbn'] == isbn]['rating'].values[0] if isbn in book_counts['isbn'].values else 0
            
            if user_count == 0 and book_count == 0:
                result.append(mu)
                continue

            w = np.log(2) / np.log(1 + (book_count + 1))
            w *= (LSTM + MF)/20

            result.append((1-w)*LSTM + w*MF)

        return result
    
    def custom9(self):
        ratings = pd.read_csv('./data/train_ratings.csv')
        user_counts = ratings.groupby('user_id')['rating'].count().reset_index()
        book_counts = ratings.groupby('isbn')['rating'].count().reset_index()

        test = pd.read_csv('./data/test_ratings.csv')

        mu = ratings['rating'].mean()

        result = []
        for user_id, isbn, LSTM, MF in tqdm(zip(test['user_id'], test['isbn'], self.output_list[0], self.output_list[1])):
            user_count = user_counts[user_counts['user_id']==user_id]['rating'].values[0] if user_id in user_counts['user_id'].values else 0
            book_count = book_counts[book_counts['isbn'] == isbn]['rating'].values[0] if isbn in book_counts['isbn'].values else 0
            
            if user_count == 0 and book_count == 0:
                result.append(mu)
                continue

            if MF < 4:
                output = LSTM
            else:
                w = np.log(2) / np.log(1 + (book_count + 1))
                output = (1-w)*LSTM + w*MF

            output = min(10, (max(0, output)))
            result.append(output)

        return result
    
    def custom10(self):
        ratings = pd.read_csv('./data/train_ratings.csv')
        mu = ratings['rating'].mean()

        result = []
        for LSTM, MF in tqdm(zip(self.output_list[0], self.output_list[1])):
            avg = (LSTM+MF)/2
            diff = abs(mu-avg)/10

            output = 0.5*(1+diff)*LSTM + 0.5*(1-diff)*MF

            if output > 10:
                result.append(10)
            elif output < 0:
                result.append(0)
            else:
                result.append(output)

        return result

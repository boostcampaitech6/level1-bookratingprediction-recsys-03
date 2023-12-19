import numpy as np
import torch
import torch.nn as nn


# feature 사이의 상호작용을 효율적으로 계산합니다.
class FactorizationMachine(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.v = nn.Parameter(torch.rand(input_dim, latent_dim), requires_grad = True)
        self.linear = nn.Linear(input_dim, 1, bias=True)


    def forward(self, x):
        linear = self.linear(x)
        square_of_sum = torch.mm(x, self.v) ** 2
        sum_of_square = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        output = linear + (0.5 * pair_interactions)
        return output


# factorization을 통해 얻은 feature를 embedding 합니다.
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout = 0.5)

    def forward(self, x):
        _, (output,_) = self.lstm(x)
        return output
        


# 기존 유저/상품 벡터와 유저/상품 리뷰 벡터를 결합하여 FM으로 학습하는 모델을 구현합니다.
class LSTM_FM(nn.Module):
    def __init__(self, args, data):
        super(LSTM_FM, self).__init__()
        self.field_dims = np.array([len(data['user2idx']), len(data['isbn2idx'])], dtype=np.uint32)
        self.embedding = FeaturesEmbedding(self.field_dims, args.deepconn_embed_dim)
        self.hidden_size = 3   
        self.num_layers = 2

        self.lstm_u = LSTM(
            input_size=1,
            hidden_size=self.hidden_size,  #hyperparam
            num_layers=self.num_layers
        )

        self.lstm_i = LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        self.fm = FactorizationMachine(
                                        input_dim=(self.hidden_size*self.num_layers*2) + (args.deepconn_embed_dim*len(self.field_dims)),
                                        latent_dim=args.deepconn_latent_dim,
                                        )


    def forward(self, x):
        user_isbn_vector, user_text_vector, item_text_vector = x[0], x[1], x[2]
        user_isbn_feature = self.embedding(user_isbn_vector)
        user_text_feature = self.lstm_u(user_text_vector).permute(1,0,2)
        item_text_feature = self.lstm_i(item_text_vector).permute(1,0,2)
  
        feature_vector = torch.cat([
                                    user_isbn_feature.view(-1, user_isbn_feature.size(1) * user_isbn_feature.size(2)),
                                    user_text_feature.reshape(-1, user_text_feature.size(1) * user_text_feature.size(2)),
                                    item_text_feature.reshape(-1, item_text_feature.size(1) * item_text_feature.size(2))
                                    ], dim=1)
        output = self.fm(feature_vector)
        return output.squeeze(1)

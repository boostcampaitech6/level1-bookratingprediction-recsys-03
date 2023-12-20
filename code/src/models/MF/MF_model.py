import numpy as np
import torch
import torch.nn as nn

# factorization을 통해 얻은 feature를 embedding 합니다.
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_normal_(self.embedding.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
    
# user와 item의 latent factor를 활용하여 GMF를 구현합니다.
class MatrixFactorization(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.user_field_idx = np.array((0, ), dtype=np.int32)
        self.item_field_idx = np.array((1, ), dtype=np.int32)
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.mu = data['train']['rating'].mean()
        self.b_u = nn.Parameter(torch.zeros(self.field_dims[0]))
        self.b_i = nn.Parameter(torch.zeros(self.field_dims[1]))
        self.b = nn.Parameter(torch.zeros(1))
        '''
        self.mu_u = data['train'].groupby('user_id')['rating'].mean()
        self.mu_i = data['train'].groupby('isbn')['rating'].mean()
        self.device = args.device'''
        
    def forward(self, x):
        uid = x[:, 0]
        iid = x[:, 1]
        '''mu_u = torch.tensor([self.mu_u[id] if id in self.mu_u else self.mu for id in uid]).to(self.device)
        mu_i = torch.tensor([self.mu_i[id] if id in self.mu_i else self.mu for id in iid]).to(self.device)'''
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        gmf = user_x * item_x
        return  self.mu + gmf.sum(dim=1) + self.b_u[uid] + self.b_i[iid] + self.b
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNLinkPredictor(nn.Module):
    def __init__(self, num_in_features, num_out_features=128, num_users=1):
        super().__init__()
        self.conv1 = GCNConv(num_in_features, num_out_features)
        self.conv2 = GCNConv(num_out_features, num_out_features)
        self.num_users = num_users

    def forward(self, x, edge_index):
        user_embedding = x[:self.num_users]
        movie_features = x[self.num_users:].detach()
        
        x = torch.cat([user_embedding, movie_features], dim=0)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x[self.num_users:] = movie_features
        return x

# 유저 - 영화 간 edge를 입력받아 유저-영화 간 관계 예측
class LinkPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, input_dim * 2)
        self.fc2 = nn.Linear(input_dim * 2, input_dim)
        self.fc3 = nn.Linear(input_dim, input_dim)
        self.fc4 = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, z, edge_index):
        row, col = edge_index
        z_row = z[row]
        z_col = z[col]
        # 임베딩 연결
        z_concat = torch.cat([z_row, z_col], dim=1)
        
        # 비선형 활성화 함수
        x = F.relu(self.fc1(z_concat))
        # 정규화를 위한 드롭아웃
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        return torch.sigmoid(self.fc4(x))
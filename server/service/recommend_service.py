import torch
import numpy as np
import pandas as pd

from dtos.response_dto import ResponseDto
from model.models import GCNLinkPredictor, LinkPredictor
from utils.utils import create_new_user_embedding, recommend_movies_for_new_user

genre_mapping = {
    1 :"Action",
    2 :"Adventure",
    3 :"Animation",
    4 :"Biography",
    5 :"Comedy",
    6 :"Crime",
    7 :"Documentary",
    8 :"Drama",
    9 :"Family",
    10 :"Fantasy",
    11 :"History",
    12 :"Horror",
    13 :"Music",
    14 :"Musical",
    15 :"Mystery",
    16 :"Romance",
    17 :"Sci-Fi",
    18 :"Sport",
    19 :"Thriller",
    20 :"War",
    21 :"Western"
}

class RecommendService:
    def __init__(self):
        self.movie_features = torch.from_numpy(np.load("./data/final_features_without_directors_0518.npy")).float()
        self.interactions = pd.read_csv("./data/interactions.csv")


        num_in_features = self.movie_features.shape[1]
        num_out_features = self.movie_features.shape[1]
        self.num_users = 1

        # load saved models
        gcn_model = GCNLinkPredictor(num_in_features, num_out_features, self.num_users)
        link_predictor = LinkPredictor(num_out_features)
        gcn_model.load_state_dict(torch.load('./model/gcn_model.pth'))
        link_predictor.load_state_dict(torch.load('./model/link_predictor.pth'))

        # 모델을 evaluation 모드로 변경
        gcn_model.eval()
        link_predictor.eval()

        self.gcn_model = gcn_model
        self.link_predictor = link_predictor

    async def recommendation(self, user_id: int, genre_id_list: list) -> ResponseDto:
        new_user_interacted_movies = self.interactions.loc[self.interactions['user_id'] == user_id]["movie_id"].values
        print(new_user_interacted_movies)
        new_user_embedding = create_new_user_embedding(self.movie_features, list(new_user_interacted_movies))

        new_x = torch.cat([new_user_embedding.view(1, -1), self.movie_features], dim=0)
        num_users = 1

        # 유저 - 영화 간 상호작용 edge index로 변환
        user_indices = [i for i in range(num_users)]
        movie_indices = [i + num_users for i in new_user_interacted_movies]

        edge_index = torch.tensor([user_indices * len(movie_indices), movie_indices], dtype=torch.long)

        node_embeddings = self.gcn_model(new_x, edge_index)

        num_recommendations = 20
        movie_id_list = recommend_movies_for_new_user(self.link_predictor, node_embeddings, num_recommendations=num_recommendations)        

        return ResponseDto(
                status=200,
                message="Recommend Successfully",
                data=movie_id_list.tolist()
            )

    async def genre_based_recommendation(self, genre_id: int):
        pass

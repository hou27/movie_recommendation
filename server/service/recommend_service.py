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
        self.movie_features = torch.from_numpy(np.load("./data/final_features_0528.npy")).float()
        self.interactions = pd.read_csv("./data/interactions.csv")

        num_in_features = self.movie_features.shape[1]
        num_out_features = self.movie_features.shape[1]
        self.num_users = 1

        # load saved models
        gcn_model = GCNLinkPredictor(num_in_features, num_out_features, self.num_users)
        link_predictor = LinkPredictor(num_out_features)
        gcn_model.load_state_dict(torch.load('./model/gcn_model_0602_heavy.pth'))
        link_predictor.load_state_dict(torch.load('./model/link_predictor_0602_heavy.pth'))
        print("Model loaded successfully")

        # 모델을 evaluation 모드로 변경
        gcn_model.eval()
        link_predictor.eval()

        self.gcn_model = gcn_model
        self.link_predictor = link_predictor

    async def recommendation(self, user_id: int, genre_id: int) -> ResponseDto:
        new_user_interacted_movies = self.interactions.loc[self.interactions['user_id'] == user_id]["movie_id"].values
        new_user_interacted_movies = new_user_interacted_movies - 1 # db index와 data index를 맞추기 위해 1을 빼줌
        num_movies = self.movie_features.shape[0]
        genre_indexs = None

        num_recommendations = 20
        num_user_interacted_movies = len(new_user_interacted_movies)
        
        # 유저가 상호작용한 영화가 없을 경우
        if num_user_interacted_movies == 0:
            return ResponseDto(
                status=400,
                message="No Interacted Movies",
                data=[]
            )
        movie_id_list = []

        # 장르 기반 추천일 경우
        if genre_id:
            genre = genre_mapping[genre_id]
            genre_indexs = np.load(f"./genre_index/{genre}_indexs.npy")      

        # 장르 기반 추천이 아닐 경우
        elif not genre_id:
            recommend_movie_count_per_loop = int(num_recommendations/num_user_interacted_movies)
            if recommend_movie_count_per_loop == 0:
                recommend_movie_count_per_loop = 1
            # loop backward
            for i in range(num_user_interacted_movies - 1, -1, -1):
                node_embeddingsx = self.__create_node_embedding([new_user_interacted_movies[i]])
                movie_id_list += recommend_movies_for_new_user(
                        self.link_predictor, 
                        node_embeddings,
                        num_movies=num_movies, 
                        num_recommendations=recommend_movie_count_per_loop, 
                        genre_indexs=genre_indexs,
                        interacted_movie_index=new_user_interacted_movies
                    ).tolist()
                # 위 결과에서 int(num_recommendations/num_user_interacted_movies)개를 랜덤으로 선정
                movie_id_list = np.random.choice(
                        movie_id_list, 
                        int(num_recommendations/num_user_interacted_movies), replace=False
                    ).tolist()
                
            # 중복 제거
            movie_id_list = list(set(movie_id_list))

        node_embeddings = self.__create_node_embedding(new_user_interacted_movies)
        movie_id_list += recommend_movies_for_new_user(
                self.link_predictor, 
                node_embeddings,
                num_movies=num_movies, 
                num_recommendations=num_recommendations,
                genre_indexs=genre_indexs,
                interacted_movie_index=new_user_interacted_movies
            ).tolist()[:20-len(movie_id_list)]
        
        # 중복 제거
        movie_id_list = list(set(movie_id_list))

        return ResponseDto(
                status=200,
                message="Recommend Successfully",
                data=movie_id_list
            )
    
    async def get_related_movies(self, movie_id: int) -> ResponseDto:
        movie_id -= 1
        num_movies = self.movie_features.shape[0]
        node_embeddings = self.__create_node_embedding([movie_id])
        related_movies = recommend_movies_for_new_user(
                self.link_predictor, 
                node_embeddings, 
                num_movies=num_movies, 
                num_recommendations=20,
                interacted_movie_index=[movie_id]
            ).tolist()

        return ResponseDto(
                status=200,
                message="Get Related Movies Successfully",
                data=related_movies
            )
    
    # private method for create node embeddings
    def __create_node_embedding(self, new_user_interacted_movies: list):
        new_user_embedding = create_new_user_embedding(self.movie_features, new_user_interacted_movies)

        new_x = torch.cat([new_user_embedding.view(1, -1), self.movie_features], dim=0)
        num_users = 1

        # 유저 - 영화 간 상호작용 edge index로 변환
        user_indices = [i for i in range(num_users)]
        movie_indices = [i + num_users for i in new_user_interacted_movies]

        edge_index = torch.tensor([user_indices * len(movie_indices), movie_indices], dtype=torch.long)

        node_embeddings = self.gcn_model(new_x, edge_index)

        return node_embeddings

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
import time

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
        self.num_users = 1
        self.num_recommendations = 20

        self.movie_features = self._load_movie_features()
        self.num_movies = self.movie_features.shape[0]
        self.interactions = self._load_interactions()
        self.dislikes = self._load_dislikes()
        self.gcn_model, self.link_predictor = self._load_models()

        self.last_interaction_update = 0
        self.last_dislike_update = 0
        self.update_interval = 10

    async def recommendation(self, user_id: int, genre_id: int) -> ResponseDto:
        self._update_data_if_needed()

        new_user_interacted_movies = self._get_user_interactions(user_id)
        new_user_disliked_movies = self._get_user_dislikes(user_id)
        
        if len(new_user_interacted_movies) == 0:
            return ResponseDto(status=400, message="No Interacted Movies", data=[])

        genre_indexs = self._get_genre_indexes(genre_id) if genre_id else None

        if genre_id:
            movie_id_list = await self._recommend_by_genre(new_user_interacted_movies, new_user_disliked_movies, genre_indexs)
        else:
            movie_id_list = await self._recommend_without_genre(new_user_interacted_movies, new_user_disliked_movies)

        return ResponseDto(status=200, message="Recommend Successfully", data=movie_id_list)

    async def get_related_movies(self, movie_id: int) -> ResponseDto:
        related_movies = await self._get_related_movies(movie_id - 1)
        return ResponseDto(status=200, message="Get Related Movies Successfully", data=related_movies)
    
    def _update_data_if_needed(self):
        current_time = time.time()
        if current_time - self.last_interaction_update > self.update_interval:
            self.interactions = self._load_interactions()
            self.last_interaction_update = current_time
        
        if current_time - self.last_dislike_update > self.update_interval:
            self.dislikes = self._load_dislikes()
            self.last_dislike_update = current_time

    def _load_movie_features(self) -> torch.Tensor:
        return torch.from_numpy(np.load("./data/final_features_0528.npy")).float()

    def _load_interactions(self) -> pd.DataFrame:
        return pd.read_csv("./data/interactions.csv")

    def _load_dislikes(self) -> pd.DataFrame:
        try:
            return pd.read_csv("./data/dislikes.csv")
        except FileNotFoundError:
            return pd.DataFrame(columns=["user_id", "movie_id"])

    def _load_models(self) -> Tuple[GCNLinkPredictor, LinkPredictor]:
        num_in_features = num_out_features = self.movie_features.shape[1]
        
        gcn_model = GCNLinkPredictor(num_in_features, num_out_features, self.num_users)
        link_predictor = LinkPredictor(num_out_features)
        
        gcn_model.load_state_dict(torch.load('./model/gcn_model_0602_heavy.pth'))
        link_predictor.load_state_dict(torch.load('./model/link_predictor_0602_heavy.pth'))
        
        gcn_model.eval()
        link_predictor.eval()
        
        return gcn_model, link_predictor

    def _get_user_interactions(self, user_id: int) -> np.ndarray:
        return self.interactions.loc[self.interactions['user_id'] == user_id]["movie_id"].values - 1

    def _get_user_dislikes(self, user_id: int) -> np.ndarray:
        return self.dislikes.loc[self.dislikes['user_id'] == user_id]["movie_id"].values - 1

    def _get_genre_indexes(self, genre_id: int) -> np.ndarray:
        genre = genre_mapping[genre_id]
        return np.load(f"./genre_index/{genre}_indexs.npy")

    async def _recommend_by_genre(self, interacted_movies: np.ndarray, disliked_movies: np.ndarray, genre_indexs: np.ndarray) -> List[int]:
        node_embeddings = self.__create_node_embedding(interacted_movies)
        return recommend_movies_for_new_user(
            self.link_predictor, 
            node_embeddings,
            num_movies=self.num_movies, 
            num_recommendations=self.num_recommendations,
            genre_indexs=genre_indexs,
            interacted_movie_index=interacted_movies,
            disliked_movie_index=disliked_movies
        ).tolist()

    async def _recommend_without_genre(self, interacted_movies: np.ndarray, disliked_movies: np.ndarray) -> List[int]:
        movie_id_list = []
        recommend_movie_count_per_loop = max(1, self.num_recommendations // len(interacted_movies))

        for movie in reversed(interacted_movies):
            node_embeddings = self.__create_node_embedding([movie])
            tmp_movie_id_list = recommend_movies_for_new_user(
                self.link_predictor, 
                node_embeddings,
                num_movies=self.num_movies, 
                num_recommendations=self.num_recommendations, 
                interacted_movie_index=interacted_movies,
                disliked_movie_index=disliked_movies
            ).tolist()
            movie_id_list.extend(np.random.choice(tmp_movie_id_list, recommend_movie_count_per_loop, replace=False).tolist())

        movie_id_list = list(set(movie_id_list))

        if len(movie_id_list) < self.num_recommendations:
            node_embeddings = self.__create_node_embedding(interacted_movies)
            tmp_final_movie_id_list = recommend_movies_for_new_user(
                self.link_predictor, 
                node_embeddings,
                num_movies=self.num_movies, 
                num_recommendations=self.num_recommendations,
                interacted_movie_index=interacted_movies,
                disliked_movie_index=disliked_movies
            ).tolist()
            movie_id_list.extend(np.random.choice(tmp_final_movie_id_list, self.num_recommendations - len(movie_id_list), replace=False).tolist())

        return list(set(movie_id_list))

    async def _get_related_movies(self, movie_id: int) -> List[int]:
        node_embeddings = self.__create_node_embedding([movie_id])
        return recommend_movies_for_new_user(
            self.link_predictor, 
            node_embeddings, 
            num_movies=self.num_movies, 
            num_recommendations=self.num_recommendations,
            interacted_movie_index=[movie_id]
        ).tolist()

    def __create_node_embedding(self, interacted_movies: List[int]) -> torch.Tensor:
        new_user_embedding = create_new_user_embedding(self.movie_features, interacted_movies)
        new_x = torch.cat([new_user_embedding.view(1, -1), self.movie_features], dim=0)
        
        user_indices = [0] * len(interacted_movies)
        movie_indices = [i + self.num_users for i in interacted_movies]
        edge_index = torch.tensor([user_indices, movie_indices], dtype=torch.long)

        return self.gcn_model(new_x, edge_index)
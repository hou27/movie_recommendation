from typing import List
import numpy as np

from dtos.recommend_dto import RecommendResponseDto

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
    movie_embedding = np.load("./data/embeddings_without_directors_gat_0512.npy")

    async def default_recommendation(self, user_movie_id_list: list) -> RecommendResponseDto:
        movie_id_list = []
        
        for movie_id in user_movie_id_list:
            euclidean_dist = [
                {
                    "index": i,
                    "distance": np.linalg.norm(self.movie_embedding[movie_id - 1] - vector)
                } for i, vector in enumerate(self.movie_embedding)
            ]
            euclidean_dist.sort(key=lambda x: x["distance"])

            # Get the top 10 most similar movies
            top_10 = euclidean_dist[1:11]
            top_10 = [movie["index"] + 1 for movie in top_10]
            movie_id_list.extend(top_10)            

        return RecommendResponseDto(
                status=200,
                message="Recommend Successfully",
                data=movie_id_list
            )

    async def genre_based_recommendation(self, genre_id: int):
        pass

import torch
import numpy as np

# 유저 임베딩 생성 함수
def create_new_user_embedding(movie_features, interacted_movie_indices):
    new_user_embedding = movie_features[interacted_movie_indices].mean(dim=0)

    return new_user_embedding

# 새로운 유저에 대한 영화 추천
# def recommend_movies_for_new_user(link_predictor, node_embeddings, num_users = 1, num_movies = 9525, num_recommendations=5, genre_indexs=None, interacted_movie_index=None):
#     movie_indices = torch.arange(num_users, num_users + num_movies) # 영화 인덱스 생성 (유저 수만큼 offset)
    
#     # user-movie pairs 생성
#     user_movie_pairs = torch.stack([torch.zeros(num_movies, dtype=torch.long), movie_indices], dim=0)

#     # 기존 유저가 본 영화 제외
#     already_seen_movies = interacted_movie_index
#     mask = torch.ones(num_movies, dtype=torch.bool)
#     if genre_indexs is not None:
#         mask = torch.zeros(num_movies, dtype=torch.bool) # 모든 영화에 대해 False로 초기화
#         mask[genre_indexs] = True # 해당 장르의 영화만 True로 변경    
#     mask[already_seen_movies] = False
#     user_movie_pairs = user_movie_pairs[:, mask]
    
#     # user-movie pairs의 score 계산
#     scores = link_predictor(node_embeddings, user_movie_pairs)
    
#     # top N 추천
#     _, top_indices = torch.topk(scores.squeeze(), num_recommendations)
#     top_movie_indices = user_movie_pairs[1][top_indices] - num_users + 1 # 유저 수만큼 offset 재조정 후 db index와 맞추기 위해 1을 더해줌
    
#     return top_movie_indices

# 제외할 영화 인덱스
excluded_movies = np.load('./data/excluded_movie_indices_0810.npy')

def recommend_movies_for_new_user(link_predictor, node_embeddings, num_users=1, num_movies=9525, num_recommendations=5, genre_indexs=None, interacted_movie_index=None):
    movie_indices = torch.arange(num_users, num_users + num_movies)  # 영화 인덱스 생성 (유저 수만큼 offset)
    
    # user-movie pairs 생성
    user_movie_pairs = torch.stack([torch.zeros(num_movies, dtype=torch.long), movie_indices], dim=0)

    # 마스크 생성
    mask = torch.ones(num_movies, dtype=torch.bool)
    
    # 장르 필터링
    if genre_indexs is not None:
        mask = torch.zeros(num_movies, dtype=torch.bool)  # 모든 영화에 대해 False로 초기화
        mask[genre_indexs] = True  # 해당 장르의 영화만 True로 변경
    
    # 이미 본 영화 제외
    if interacted_movie_index is not None:
        mask[interacted_movie_index] = False
    
    # 제외할 영화 인덱스 적용
    excluded_movies_tensor = torch.tensor(excluded_movies, dtype=torch.long)
    mask[excluded_movies_tensor] = False
    
    # 마스크 적용
    user_movie_pairs = user_movie_pairs[:, mask]
    
    # user-movie pairs의 score 계산
    scores = link_predictor(node_embeddings, user_movie_pairs)
    
    # top N 추천
    _, top_indices = torch.topk(scores.squeeze(), num_recommendations)
    top_movie_indices = user_movie_pairs[1][top_indices] - num_users + 1  # 유저 수만큼 offset 재조정 후 db index와 맞추기 위해 1을 더해줌
    
    return top_movie_indices
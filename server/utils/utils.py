import torch

# 유저 임베딩 생성 함수
def create_new_user_embedding(movie_features, interacted_movie_indices):
    new_user_embedding = movie_features[interacted_movie_indices].mean(dim=0)

    return new_user_embedding

# 새로운 유저에 대한 영화 추천
def recommend_movies_for_new_user(link_predictor, node_embeddings, num_users = 1, num_movies = 9525, num_recommendations=5):
    movie_indices = torch.arange(num_users, num_users + num_movies) # 영화 인덱스 생성 (유저 수만큼 offset)
    
    # user-movie pairs 생성
    user_movie_pairs = torch.stack([torch.zeros(num_movies, dtype=torch.long), movie_indices], dim=0)
    print(user_movie_pairs)
    
    # user-movie pairs의 score 계산
    scores = link_predictor(node_embeddings, user_movie_pairs)
    
    # top N 추천
    _, top_indices = torch.topk(scores.squeeze(), num_recommendations)
    top_movie_indices = user_movie_pairs[1][top_indices] - num_users # 유저 수만큼 offset 재조정
    
    return top_movie_indices
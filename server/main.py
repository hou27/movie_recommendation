from typing import List
from fastapi import Depends, FastAPI, Query

from service.interaction_service import InteractionService
from service.recommend_service import RecommendService
from dtos.response_dto import ResponseDto
from dtos.interaction_request_dto import ControlLikeRequest, SurveyResultRequest

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "hou27"}

@app.get("/recommendation", status_code=200, response_model=ResponseDto)
async def recommendation(
        recommend_service: RecommendService = Depends(RecommendService),
        user_id: int = Query(..., alias="userId"), # required
        genre_id: int = Query(None, alias="genreId"), # optional
    ) ->  ResponseDto:
    # genre_id_list = []
    # if genre_ids:
    #     genre_id_list = [int(id) for id in genre_ids.split(',')]

    return await recommend_service.recommendation(
            user_id=user_id, 
            genre_id=genre_id
        )

@app.get("/recommendation/movie/{movie_id}", status_code=200, response_model=ResponseDto)
async def get_related_movies(
        movie_id: int,
        recommend_service: RecommendService = Depends(RecommendService),
    ) -> ResponseDto:
    return await recommend_service.get_related_movies(movie_id)

@app.post("/surveys/result", status_code=201, response_model=ResponseDto)
async def survey_result(
        request: SurveyResultRequest,
        interaction_service: InteractionService = Depends(InteractionService),
    ) ->  ResponseDto:
    user_id = request.userId
    movie_id_list = request.movieIds

    return await interaction_service.survey_result(user_id, movie_id_list)

@app.post("/likes", status_code=201, response_model=ResponseDto)
async def control_like(
        request: ControlLikeRequest,
        interaction_service: InteractionService = Depends(InteractionService),
    ) ->  ResponseDto:
    user_id = request.userId
    movie_id = request.movieId

    return await interaction_service.control_interaction(user_id, movie_id, "like")

@app.post("/dislikes", status_code=201, response_model=ResponseDto)
async def control_dislike(
        request: ControlLikeRequest,
        interaction_service: InteractionService = Depends(InteractionService),
    ) ->  ResponseDto:
    user_id = request.userId
    movie_id = request.movieId

    return await interaction_service.control_interaction(user_id, movie_id, "dislike")

# @app.get("/likes", status_code=200, response_model=ResponseDto)
# async def get_interactions(
#         interaction_service: InteractionService = Depends(InteractionService),
#         user_id: int = Query(..., alias="userId"),
#     ) ->  ResponseDto:

#     return await interaction_service.get_interactions(user_id)
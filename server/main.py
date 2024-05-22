from typing import List
from fastapi import Depends, FastAPI, Query

from service.interaction_service import InteractionService
from service.recommend_service import RecommendService
from dtos.response_dto import ResponseDto

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
            user_id=int(user_id), 
            genre_id=genre_id
        )

@app.post("/surveys/result", status_code=201, response_model=ResponseDto)
async def survey_result(
        interaction_service: InteractionService = Depends(InteractionService),
        user_id: int = Query(..., alias="userId"),
        movie_ids: str = Query(..., alias="movieIds")
    ) ->  ResponseDto:
    movie_id_list = [int(id) for id in movie_ids.split(',')]

    return await interaction_service.survey_result(user_id, movie_id_list)

@app.post("/likes", status_code=201, response_model=ResponseDto)
async def control_like(
        interaction_service: InteractionService = Depends(InteractionService),
        user_id: int = Query(..., alias="userId"),
        movie_id: int = Query(..., alias="movieId")
    ) ->  ResponseDto:
    return await interaction_service.control_like(user_id, movie_id)

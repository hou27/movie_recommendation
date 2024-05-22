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
        user_id: int = Query(...), # required
        genre_id: int = Query(None), # optional
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
        userId: int = Query(..., alias="userId"),
        movieIds: str = Query(...)
    ) ->  ResponseDto:
    movie_id_list = [int(id) for id in movieIds.split(',')]

    return await interaction_service.survey_result(userId, movie_id_list)

@app.post("/likes", status_code=201, response_model=ResponseDto)
async def control_like(
        interaction_service: InteractionService = Depends(InteractionService),
        userId: int = Query(..., alias="userId"),
        movieId: int = Query(...)
    ) ->  ResponseDto:
    return await interaction_service.control_like(userId, movieId)

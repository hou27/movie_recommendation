from typing import List
from fastapi import Depends, FastAPI, Query

from service.recommend_service import RecommendService
from dtos.recommend_dto import RecommendResponseDto

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "hou27"}


@app.get("/default_recommendation", status_code=200, response_model=RecommendResponseDto)
async def default_recommendation(
        recommend_service: RecommendService = Depends(RecommendService),
        movieIds: str = Query(...)
    ) ->  RecommendResponseDto:
    movieIds_list = [int(id) for id in movieIds.split(',')]

    return await recommend_service.default_recommendation(user_movie_id_list=movieIds_list)

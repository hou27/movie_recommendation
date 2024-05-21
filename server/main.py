from typing import List
from fastapi import Depends, FastAPI, Query

from service.recommend_service import RecommendService
from dtos.recommend_dto import RecommendResponseDto

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "hou27"}


@app.get("/recommendation", status_code=200, response_model=RecommendResponseDto)
async def recommendation(
        recommend_service: RecommendService = Depends(RecommendService),
        user_id: int = Query(...), # required
        genre_ids: str = Query(None), # optional
    ) ->  RecommendResponseDto:
    genre_id_list = []
    if genre_ids:
        genre_id_list = [int(id) for id in genre_ids.split(',')]

    return await recommend_service.recommendation(
            user_id=int(user_id), 
            genre_id_list=genre_id_list
        )

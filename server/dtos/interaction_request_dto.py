from pydantic import BaseModel
from typing import List

class SurveyResultRequest(BaseModel):
    userId: int
    movieIds: List[int]

class ControlLikeRequest(BaseModel):
    userId: int
    movieId: int
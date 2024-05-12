from pydantic import BaseModel
from typing import List


class RecommendResponseDto(BaseModel):
    status: int
    message: str
    data: list = []
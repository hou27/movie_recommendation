from pydantic import BaseModel
from typing import List


class ResponseDto(BaseModel):
    status: int
    message: str
    data: list = []
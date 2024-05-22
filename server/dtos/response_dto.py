from pydantic import BaseModel

class ResponseDto(BaseModel):
    status: int
    message: str
    # data is list or None
    data: list = None

from pydantic import BaseModel



class image_predRequestModel(BaseModel):
    image: str
    class_name : str


class textRequestModel(BaseModel):
    prompt : str = "ML Summer school"


class textResponseModel(BaseModel):
    execution_time: int = 0
    result : str = ""
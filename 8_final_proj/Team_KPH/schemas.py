from pydantic import BaseModel

class ImagePredRequest(BaseModel):
    image: str

from pydantic import BaseModel

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    image_base64: str  

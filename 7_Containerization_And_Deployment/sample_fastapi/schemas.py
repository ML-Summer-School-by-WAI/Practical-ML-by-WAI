from pydantic import BaseModel,Field
from typing import Literal


class studentRequestModel(BaseModel):

    class_name : Literal["ML_in_Prod_1","ML_in_Prod_2", "Big_Data"]

    stu_name: str = "Mg ba"
    stu_id : int = Field(..., ge=100,le=150, description="Student IDs must between 100 and 150")

    stu_age : int = Field(..., ge=16,le=35, description="Student IDs must between 100 and 150")


class textRequestModel(BaseModel):
    prompt : str = "What is deep learning"

class textResponseModel(BaseModel):
    execution_time : int = 0 
    result : str = ""




class image_predRequestModel(BaseModel):
    image: str
    class_name : str


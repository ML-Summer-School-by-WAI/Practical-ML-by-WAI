from fastapi import FastAPI,Request, Body
import uvicorn
import time
from models.schemas import image_predRequestModel, textRequestModel, textResponseModel
from contextlib import asynccontextmanager
from model_work import CatAndDogModel,TextGenerationModel


ml_models = {}
@asynccontextmanager
async def lifespan(app: FastAPI):

    catAndDogModel = CatAndDogModel()
    catAndDogModel.load_model()


    textGenModel = TextGenerationModel()

    ml_models["catAndDogModel"] = catAndDogModel
    ml_models["textGenModel"] = textGenModel

    
    yield
    ml_models.clear()




app = FastAPI(lifespan=lifespan)



@app.get("/")
def home():
    return {"message": "Hello, World!"}


@app.post("/predict")
def predict(request: Request, body: image_predRequestModel = Body(...)) -> image_predRequestModel:
    

    request_data = body
    catAndDogModel = ml_models["catAndDogModel"]

    img_array = catAndDogModel.preprocess_image(request_data.image)
    class_name, prediction = catAndDogModel.predict(img_array)
    str_output = f"Prediction: {class_name}, Confidence: {prediction:.2f}"
    request_data.class_name = str_output
    
    
    return request_data




@app.post("/text_generation")
async def generate_text(request: Request, body: textRequestModel = Body(...)) -> textResponseModel:

    start_time = time.time()
    response_text =  ml_models["textGenModel"].generate_text(body.prompt)
    execution_time = start_time - time.time()
    print(f"Response: {response_text}")
    print(f"Execution time: {execution_time:.2f} seconds")



    response = textResponseModel(execution_time=int(execution_time * 1000), result=response_text)
    return response




if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8888)
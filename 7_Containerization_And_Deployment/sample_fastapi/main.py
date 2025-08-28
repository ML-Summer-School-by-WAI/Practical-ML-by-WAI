from fastapi import FastAPI, Request,Body
from schemas import studentRequestModel,textRequestModel,textResponseModel,image_predRequestModel
import time
import uvicorn
import asyncio


from fastapi import status,Response,Query
from fastapi.responses import StreamingResponse


from contextlib import asynccontextmanager
from model_work import textModel,audioModel
from model_work import CatAndDogModel,TextGenerationModel




ml_models ={}

@asynccontextmanager
async def startup_lifespan(app : FastAPI):

    text_m_obj = textModel()
    text_m_obj.load_pipeline()
    ml_models["text_m_obj"] = text_m_obj


    audio_m_obj = audioModel()
    audio_m_obj.load_audio_model()
    ml_models["audio_m_obj"] = audio_m_obj


    catAndDogModel = CatAndDogModel()
    catAndDogModel.load_model()
    ml_models["catAndDogModel"] = catAndDogModel




    textGenModel = TextGenerationModel()    
    ml_models["textGenModel"] = textGenModel



    yield
    ml_models.clear()





app = FastAPI(lifespan=startup_lifespan)


@app.get("/")
def home():
    return "hello world"


@app.post("/get_student")
def get_student(request : Request,
                body : studentRequestModel = Body(...)) -> textResponseModel:
    
    start_time = time.time()
    result = "OK"
    return textResponseModel(
        execution_time=int(time.time()-start_time),
        result=result
    )


@app.post("/sync")
def sync_prediction(prompt: str) -> textResponseModel:
    start_time = time.time()
    time.sleep(5)

    
    result = "OK"
    return textResponseModel(
        execution_time=int(time.time()-start_time),
        result=result
    )



@app.post("/async")
async def async_prediction() ->textResponseModel:
    start_time = time.time()
    await asyncio.sleep(5)

    result = "OK"
    return textResponseModel(
        execution_time=int(time.time()-start_time),
        result=result
    )


@app.post("/text_gen")
def serve_text_gen(request : Request,
                body : textRequestModel = Body(...)) -> textResponseModel:
    start_time = time.time()
    generated_text = ml_models["text_m_obj"].predict(user_message = body.prompt)

    return textResponseModel(
            execution_time=int(time.time()-start_time),
            result=generated_text
        )

@app.get("/audio_gen",
          responses={status.HTTP_200_OK:{"content" : {"audio/wav":{}}}},
          response_class=StreamingResponse,)
async def serve_audio_gen(prompt = Query(...),prest : audioModel.VoicePresets = Query(default="v2/en_speaker_9")) -> StreamingResponse:
    
    output_audio_array = ml_models["audio_m_obj"].generate_audio(prompt)
    
    return StreamingResponse(output_audio_array, media_type="audio/wav",headers={"Content-Disposition": "inline; filename=generated_audio.wav"})




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
    uvicorn.run("main:app", host='0.0.0.0', port=8888, reload=True)
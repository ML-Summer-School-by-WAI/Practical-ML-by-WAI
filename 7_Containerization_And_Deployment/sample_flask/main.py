from flask import Flask
import tensorflow  as tf
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

@app.get("/")
def home():
    return "Hello there"

@app.get("/check_gpu")
def check_gpu():
    gpu_status = tf.test.is_gpu_available()
    openai_key = os.getenv('OPENAI_KEY')
    return {"gpu_status " : gpu_status,
            "openAI_key" : openai_key}




if __name__== "__main__":
    app.run(port=8888)
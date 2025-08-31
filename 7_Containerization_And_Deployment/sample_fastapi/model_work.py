
from typing import Literal
import numpy as np
from io import BytesIO
from transformers import AutoProcessor, AutoModel, Pipeline, pipeline
from typing import Literal
import torch
import utils
import os
import json
from openai import OpenAI
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


class textModel():
    def __init__(self):
        self.pipeline = None
        self.device_name = None
        if torch.backends.mps.is_available():
            self.device_name = "mps" #cuda
        else:
            self.device_name = "cpu"


    def load_pipeline(self):
        print("loading textModel")
        self.pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device=self.device_name)
        print("textModel is loaded ")


    def predict(self,user_message):
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {"role": "user", "content": user_message},
        ]
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

        outputs_data = outputs[0]["generated_text"].split("<|assistant|>")
        if len(outputs_data) != 2:
            return "unexpected output len"

        result = outputs_data[1].strip()
        return result



class audioModel():
    
    VoicePresets = Literal["v2/en_speaker_1", "v2/en_speaker_9"]

    def __init__(self):
        self.preset = "v2/en_speaker_9"
        self.processor = None
        self.model = None

    def load_audio_model(self) -> tuple[AutoProcessor, AutoModel]:

        print("loading audioModel")

        #Download the small bark processor which prepares input text prompt for the core model
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")


        #Download the bark model which will be used to generate the output audio.
        self.model = AutoModel.from_pretrained("suno/bark-small")

        print("audioModel is loaded ")
    

    def generate_audio(
        self,
        prompt: str ) -> tuple[np.array, int]:

        if self.processor is None or self.model is None:
            return
        

        # Preprocess text prompt with a speaker voice preset embedding and return a Pytorch tensor array of tokenized inputs using return_tensors="pt"
        inputs = self.processor(text=[prompt], return_tensors="pt", voice_preset=self.preset)


        # Generate an audio array that contains amplitude values of the synthesized audio signal over time.
        output = self.model.generate(**inputs, do_sample=True).cpu().numpy().squeeze()

        # Obtain the sampling rate from model generating configurations which can be used to produce the audio.
        sample_rate = self.model.generation_config.sample_rate
        audio_buffer = utils.audio_array_to_buffer(output,sample_rate)


        return audio_buffer



class CatAndDogModel:

    def __init__(self):
        self.model_path = os.getcwd() + "/ml_models/cat_and_dog_model.h5"
        self.class_path = os.getcwd() + "/ml_models/class_names.json"
        self.input_img_size = (128,128)  # Assuming the model expects 224x224 images
        self.model = None
        self.class_indices = None


    def load_model(self):
        print("Loading model and class indices...")
        try:
            self.model = load_model(self.model_path)
    
            with open(self.class_path, 'r') as f:
                self.class_indices = json.load(f)

            print("Model and class indices loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model or class indices: {e}")
            return False
 
        return True
    

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=self.input_img_size)
        img_array = image.img_to_array(img)  # shape: (_, _, 3)
        img_array = img_array / 255.0        # normalize

        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, _, _, 3)

 
        return img_array


    def predict(self, img_array):
        prediction = self.model.predict(img_array)[0][0]
        class_name = "dogs" if prediction > 0.5 else "cats"
        return class_name, prediction




class TextGenerationModel:

    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    
    def generate_text(self, prompt):
        completion = self.client.chat.completions.create(
        model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        messages=[
            {"role": "system", "content": "Always answer in rhymes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        )
        reponse = completion.choices[0].message.content


        return reponse

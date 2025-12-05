import pandas as pd
import numpy as np
# from params import *

# from dl_logic.registry import load_model # A VENIR
import io
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from keras import models
from carte_territoire.dl_logic.model import predict_model

CHUNK_SIZE=256
app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

## 💡 Preload the model to accelerate the predictions
## and then store the model in an `app.state.model` global variable, accessible across all routes!
## This will prove very useful for the Demo Day
app.state.model = models.load_model("../trained_models/my_model_0512.keras", compile=False)

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END

@app.get("/predict")
def predict(X_test:str):      # IMAGE OR ARRAY as an input ?
    output = X_test + " bien reçu"
    return output

@app.post("/upload-and-process/")
async def upload_and_process_image(file: UploadFile = File(...)):
    """
    Receives an uploaded image, processes it, and returns the result.
    """
    # 1. Read the file content into memory
    # You must await file.read() for UploadFile
    file_content = await file.read()

    # 2. Open the image from the bytes content and convert to array
    image = Image.open(io.BytesIO(file_content))
    array = np.array(image)
    array_resize = array[:256,:256,:]
    array_pred = predict_model(app.state.model, array_resize, (CHUNK_SIZE, CHUNK_SIZE, 3))
    label_pred = Image.fromarray(array_pred)

    output_buffer = io.BytesIO()
    label_pred.save(output_buffer, format="JPEG") # Use JPEG, PNG, etc.
    processed_image_bytes = output_buffer.getvalue()

    return Response(content=processed_image_bytes, media_type="image/jpeg")

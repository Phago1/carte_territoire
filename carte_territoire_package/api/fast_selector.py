import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from carte_territoire_package.params import *

# from dl_logic.registry import load_model # A VENIR
import io
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from keras import models
from carte_territoire_package.dl_logic.model import predict_model
from carte_territoire_package.interface.utils import labels_to_rgb
from carte_territoire_package.dl_logic.labels import FLAIR_CLASS_DATA, REDUCED_7

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
current_dir = Path(__file__).parent
path_to_model_dir = current_dir.parent / 'trained_models'
model_path = path_to_model_dir / 'my_model_0512.keras'  #'models_20251210-004639.keras'
app.state.model = models.load_model(model_path, compile=False)
DICO_LABEL = FLAIR_CLASS_DATA
CHUNK_SIZE_API = 256

@app.get("/")
def root():
    return dict(greeting="Hi there, the API is working !")

@app.get("/predict")
def predict(test:str):
    output = test + " bien reçu"
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
    X_test = array[:,:,:3]
    shape_orig = X_test.shape

    # 3. Process: Apply the model
    y_pred = predict_model(app.state.model, X_test)

    # 5. Recombine: Place the result into the correct slice of the output array
    # The output shape is (CHUNK_SIZE, CHUNK_SIZE)

    y_pred_back = tf.image.resize(np.expand_dims(y_pred ,-1),
                        size=shape_orig[:2],
                        preserve_aspect_ratio=True,
                        antialias=True)
    y_test = y_pred_back[:, :, 0].numpy().astype(np.int32)
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    y_test = labels_to_rgb(y_test, DICO_LABEL)
    label_pred = Image.fromarray(y_test)

    output_buffer = io.BytesIO()

    label_pred.save(output_buffer, format="PNG")
    processed_image_bytes = output_buffer.getvalue()

    return Response(content=processed_image_bytes, media_type="image/PNG")

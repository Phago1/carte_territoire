import pandas as pd
import numpy as np
from pathlib import Path
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

    # --------------------------------------------------------------------------
    # Splitting the input array (H, W, 3) into smaller chunks, processesing them
    # reassembling the results into a single 2D array (H_fit, W_fit),
    # where H_fit and W_fit are the largest dimensions divisible by CHUNK_SIZE.
    # --------------------------------------------------------------------------

    # 1. Determine the dimensions that fit the chunk size
    size_a, size_b, num_channels = X_test.shape

    # Calculate the largest dimensions that are perfectly divisible by chunk_size
    h_fit = (size_a // CHUNK_SIZE) * CHUNK_SIZE
    w_fit = (size_b // CHUNK_SIZE) * CHUNK_SIZE

    # Crop the array to the top-left section that fits the chunk grid
    # This addresses the requirement: "keep only the top - left of the input image"
    cropped_array = X_test[:h_fit, :w_fit, :]

    num_chunks_rows = h_fit // CHUNK_SIZE
    num_chunks_cols = w_fit // CHUNK_SIZE

    # Initialize the output array with the final desired shape (H_fit, W_fit)
    reassembled_array = np.zeros((h_fit, w_fit), dtype=np.int64)

    # 2. Iterate through rows and columns of chunks
    for i in range(num_chunks_rows):
        for j in range(num_chunks_cols):
            # Calculate pixel indices for the current chunk
            row_start = i * CHUNK_SIZE
            row_end = row_start + CHUNK_SIZE
            col_start = j * CHUNK_SIZE
            col_end = col_start + CHUNK_SIZE

            # 3. Split: Extract the current chunk (shape: CHUNK_SIZE, CHUNK_SIZE, 3)
            current_chunk = cropped_array[row_start:row_end, col_start:col_end, :] / 255.

            # 4. Process: Apply the external function
            processed_chunk = predict_model(app.state.model, current_chunk)

            # 5. Recombine: Place the result into the correct slice of the output array
            # The output shape is (CHUNK_SIZE, CHUNK_SIZE)
            reassembled_array[row_start:row_end, col_start:col_end] = processed_chunk

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    y_test = labels_to_rgb(reassembled_array, DICO_LABEL)
    label_pred = Image.fromarray(y_test)

    output_buffer = io.BytesIO()

    label_pred.save(output_buffer, format="PNG")
    processed_image_bytes = output_buffer.getvalue()

    return Response(content=processed_image_bytes, media_type="image/PNG")

# WORK IN PROGRESS

from dl_logic.preprocessor import pairs_crea, images_to_chunks, slice_to_chunks
from dl_logic.model import initialize_model, compile_model, train_model, predict_model, plot_predict
from dl_logic.labels import reduce_mask, flair_class_data, REDUCED_7
from PIL import Image
import numpy as np
from params import *


## Retrieve List of tuples Tile / Label
ortho_GCP_filenames, label_GCP_filenames = pairs_crea()

## Load tensorflows TODO
# ortho_arrays, label_arrays = function to load 42 Tensors (5000, 5000, 3) and 42 Tensors (5000, 5000)

## Create X_train, y_train containing the chunks of size (256, 256)
X_train, y_train = images_to_chunks(ortho_arrays, label_arrays)
# iF reduction True, number of classes set to 7
reduce_mask(y_train) # TODO


## Create X_val, y_val containing the chunks of size (256, 256)
label_image_val = Image.open('gs://carte-ter-bucket/val/90-2023-0985-6735-LA93-0M20-IRC-E080_res1.00m_labels.tif')
tile_image_val = Image.open('gs://carte-ter-bucket/val/90-2023-0985-6735-LA93-0M20-IRC-E080_res1.00m.tif')

label_array_val = np.array(label_image_val)
tile_array_val = np.array(tile_image_val)

X_val, y_val = slice_to_chunks(tile_array_val, label_array_val)

## Model
if LBL_REDUCTION:
    target_class_ID = REDUCED_7.keys
else:
    target_class_ID = flair_class_data.keys
model = initialize_model((CHUNK_SIZE, CHUNK_SIZE), 7 is LBL_REDUCTION else 15)
model = compile_model(model, target_class_ID)
# model, history = train_model(model, X_train, y_train TODO

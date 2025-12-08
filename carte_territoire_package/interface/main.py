# WORK IN PROGRESS

from dl_logic.preprocessor import pairs_crea, images_to_chunks, slice_to_chunks, get_tf_dataset
from dl_logic.model import initialize_model, compile_model, train_model, predict_model, plot_predict
from dl_logic.model import initialize_unet_model, initialize_CNN_model
from dl_logic.labels import reduce_mask, reduce_mask, REDUCED_7, flair_class_data
from dl_logic.registry import save_results, save_model
from PIL import Image
import numpy as np
from params import *


# ## Retrieve List of tuples Tile / Label
# ortho_GCP_filenames, label_GCP_filenames = pairs_crea()

# ## Load tensorflows TODO
# # ortho_arrays, label_arrays = function to load 42 Tensors (5000, 5000, 3) and 42 Tensors (5000, 5000)

# ## Create X_train, y_train containing the chunks of size (256, 256)
# X_train, y_train = images_to_chunks(ortho_arrays, label_arrays)
# # iF reduction True, number of classes set to 7
# reduce_mask(y_train) # TODO


# ## Create X_val, y_val containing the chunks of size (256, 256)
# label_image_val = Image.open('gs://carte-ter-bucket/val/90-2023-0985-6735-LA93-0M20-IRC-E080_res1.00m_labels.tif')
# tile_image_val = Image.open('gs://carte-ter-bucket/val/90-2023-0985-6735-LA93-0M20-IRC-E080_res1.00m.tif')

# label_array_val = np.array(label_image_val)
# tile_array_val = np.array(tile_image_val)

# X_val, y_val = slice_to_chunks(tile_array_val, label_array_val)

# ## Model
# if LBL_REDUCTION:
#     target_class_ID = REDUCED_7.keys
# else:
#     target_class_ID = flair_class_data.keys
# model = initialize_model((CHUNK_SIZE, CHUNK_SIZE), 7 is LBL_REDUCTION else 15)
# model = compile_model(model, target_class_ID)
# # model, history = train_model(model, X_train, y_train TODO

# # metrics
# IoU = np.min(history.history['iou'])

# params = dict(
#         context="train",
#         chunk_size=CHUNK_SIZE,
#         row_count=len(X_train),
#     )

#  ===========================================================================
#  ===========================================================================

#       ==========================================
#       --------------- Preprocess ---------------
#       BATCH_SIZE = ????
def preprocess():
    ds_train = get_tf_dataset('train/')
    ds_val = get_tf_dataset('val/')

    print("✅ preprocess() done \n")
    return ds_train, ds_val

#       ==========================================
#       ----------------- Train ------------------
def train(model_category:str='unet', ds_train, ds_val):   # unet architecture or cnn
    if LBL_REDUCTION:
        target_class_ID = REDUCED_7.keys
    else:
        target_class_ID = flair_class_data.keys
    num_class = len(target_class_ID)

    if model_category == 'unet':
        model = initialize_unet_model((CHUNK_SIZE,CHUNK_SIZE,3), num_class)

    if model_category == 'cnn':
        model = initialize_CNN_model((CHUNK_SIZE, CHUNK_SIZE), num_class)



    model = compile_model(model, target_class_ID)
    history, model = train_model(model,
                                    ds_train,
                                    ds_val,
                                    epochs=30,
                                    batch_size = BATCH_SIZE, # BATCH_SIZE est une var d'environnement
                                    patience=3)


    IoU = np.max(history.history['mean_io_u'])
    accuracy = np.max(history.history['val_accuracy'])
    metrics = dict(IoU=IoU,accuracy=accuracy)

    model_name = model.model_name
    params = dict(context="train",
                  chunk_size=CHUNK_SIZE,
                  model=model_name,
                  #row_count=len(X_train)      # = nombre de chunk (ou de tuile), a verifier la syntaxe
                  )
    # results saved on hard drive from registry.py
    save_results(params=params, metrics=metrics)
    # model weight saved on gcs, can be saved locally too
    save_model(model=model)

    return IoU

# WORK IN PROGRESS

from carte_territoire_package.dl_logic.preprocessor import get_tf_dataset
from carte_territoire_package.dl_logic.model import initialize_cnn_model, initialize_unet_model, initialize_unet_plus_model
from carte_territoire_package.dl_logic.model import compile_model, train_model, predict_model, build_model_metrics, plot_predict
from carte_territoire_package.dl_logic.labels import REDUCED_7_NO_COLORS, FLAIR_CLASS_DATA_NO_COLORS
from carte_territoire_package.dl_logic.registry import save_results, save_model
from PIL import Image
import numpy as np
from carte_territoire_package.params import *


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
def preprocess():
    ds_train = get_tf_dataset('train/')
    ds_val = get_tf_dataset('val/')
    ds_test = get_tf_dataset('test/')

    print("✅ preprocess() done \n")
    return ds_train, ds_val, ds_test

#       ==========================================
#       ----------------- Train ------------------
def train(ds_train, ds_val, ds_test, epochs: int=100, patience: int=5):
    if LBL_REDUCTION == True:
        target_class_ID = REDUCED_7_NO_COLORS.keys()
        target_class_values = list(REDUCED_7_NO_COLORS.values())
    else:
        target_class_ID = FLAIR_CLASS_DATA_NO_COLORS.keys()
        target_class_values = list(FLAIR_CLASS_DATA_NO_COLORS.values())

    num_class = len(target_class_ID)

    if MODEL_ARCH == 'cnn':
        model = initialize_cnn_model(number_of_classes=num_class)
        print('Model architecture: CNN')

    elif MODEL_ARCH == 'unet':
        model = initialize_unet_model(number_of_classes=num_class)
        print('Model architecture: UNET')

    elif MODEL_ARCH == 'unet_plus':
        model = initialize_unet_plus_model(number_of_classes=num_class)
        print('Model architecture: UNET_PLUS')

    else:
        print('❌ No model defined')

    model = compile_model(model=model, number_of_classes=num_class)

    history = train_model(model,
                          ds_train,
                          ds_val,
                          epochs=epochs,
                          patience=patience)

    IoU_train = np.max(history.history['mean_io_u'])
    IoU_val = np.max(history.history['val_mean_io_u'])

    IoU_per_class_test, IoU_test = build_model_metrics(model=model,
                                                dataset=ds_test,
                                                num_classes=num_class,
                                                class_names=target_class_values,
                                                verbose=True)

    metrics = dict(IoU_train=IoU_train,
                   IoU_val=IoU_val,
                   IoU_test=IoU_test,
                   IoU_per_class_test=IoU_per_class_test
                )

    model_name = model.model_name

    params = dict(context="train",
                  model=model_name,
                  lbl_reduction=LBL_REDUCTION,
                  chunk_size=CHUNK_SIZE,
                  batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE,
                  )

    save_results(params=params, metrics=metrics)
    print('results saved')

    save_model(model=model)
    print('model weight')

    return history, model, metrics, params

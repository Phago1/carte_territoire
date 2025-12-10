# Here we initialize, compile, train and evaluate the model

from keras import Sequential, layers, metrics, losses, Input
from keras.models import Model
from keras.optimizers import Adam, schedules
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from carte_territoire_package.params import *
from sklearn.metrics import confusion_matrix
from carte_territoire_package.interface.utils import labels_to_rgb
from carte_territoire_package.dl_logic.labels import FLAIR_CLASS_DATA, REDUCED_7

def initialize_cnn_model(input_shape: int = (CHUNK_SIZE, CHUNK_SIZE, 3),
                         number_of_classes: int = 7 if LBL_REDUCTION == True else 16):
    """
    input_shape usualy like X_train.shape[1:]
    repoduced from state-of-the-art recommandation:
    https://deeplearningwithpython.io/chapters/chapter11_image-segmentation/
    """

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2D(128, 3, padding='same', activation='relu'))
    model.add(layers.Conv2D(256, 3, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2D(256, 3, padding='same', activation='relu'))

    model.add(layers.Conv2DTranspose(256, 3, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(128, 3, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(64, 3, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'))

    model.add(layers.Conv2D(number_of_classes, 1, padding='same', activation='softmax'))

    model.model_name = "cnn"

    return model


def compile_model(model,
                  learning_rate = LEARNING_RATE, # can be a float or 'exponential'
                  number_of_classes: int = 7 if LBL_REDUCTION==True else 16):
    """
    """
    # --- LOSS ---
    dice_loss = losses.Dice(reduction='sum_over_batch_size', name='dice')
    focal_loss = losses.CategoricalFocalCrossentropy(reduction='sum_over_batch_size', name='focal')
    def combined_loss(y_true, y_pred):
        """
        one-hot-encode y_true so dice and focal losses can be computed
        tf.one_hot need integer inputs while they are originaly float32
        --> tf.cast to encode in int32
        then create a custom function merging dice and focal losses

        """
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=number_of_classes)

        total_loss = 0.5*dice_loss(y_true_oh, y_pred) + 0.5*focal_loss(y_true_oh, y_pred)

        return total_loss

    # --- LEARNING_RATE ---
    if learning_rate == 'exponential':
        learning_rate = schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=30*BATCH_SIZE,
            decay_rate=0.8,
            staircase=True,
            )
        print('✅ LEARNING_RATE is Exponential Decay')

    elif type(float(learning_rate)) is float:
        learning_rate = float(learning_rate)
        print(f'✅ LEARNING_RATE is {learning_rate}')

    IoU = metrics.MeanIoU(num_classes=number_of_classes, sparse_y_true=True, sparse_y_pred=False)

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=combined_loss,
                  metrics=[IoU]
                  )

    return model


def train_model(model, ds_train, ds_val, epochs=100, patience=5):
    """
    train model from train and val tensorfflow datasets
    """

    es = EarlyStopping(patience=patience, restore_best_weights=True)

    history = model.fit(ds_train,
              validation_data=ds_val,
              epochs=epochs,
              callbacks=[es]
              )

    return history


def predict_model(model, X_pred: tuple, input_shape: tuple = (CHUNK_SIZE, CHUNK_SIZE, 3)):

    """
    .predict expect a shape==(batch_size, height_trained_model, width_trained_model, 3)
    Better to provide an X_pred correctly shaped (height_trained_model, width_trained_model, 3)
    Otherwise an automatic resizing is done.

    Expand_dims is used to have a shape==(1, height, width, 3) for a single image.

    X_pred.shape==(height, width, 3)
    y_label.shape==(height,width)

    .predict returns probabilities for each class for each pixel. .argmax() recovers the
    most probable class for each pixel
    """

    if X_pred.shape != input_shape:
        shape = X_pred.shape
        X_pred = tf.image.resize(X_pred,
                        size=input_shape[0:2],
                        preserve_aspect_ratio=True,
                        antialias=True)
        print(f"X_pred resized from {shape} to {X_pred.shape}")

    X_pred = X_pred/255.
    print(f'X_pred normalized, X_pred[0,0,:]={X_pred[0,0,:]}')

    X_pred = np.expand_dims(X_pred, axis=0)
    print(f'dim added to X_pred, X_pred.shape={X_pred.shape}')

    y_pred = model.predict(X_pred)
    y_pred = y_pred.reshape(y_pred[0].shape)
    y_pred = np.argmax(y_pred, axis=-1)

    return y_pred


def plot_predict(X_pred, y_pred, y_label):

    # --- Palette selection ---
    y_pred = labels_to_rgb(y_pred, REDUCED_7 if LBL_REDUCTION == True else FLAIR_CLASS_DATA)
    y_label = labels_to_rgb(y_label, REDUCED_7 if LBL_REDUCTION == True else FLAIR_CLASS_DATA)

    fig, ((ax0, ax1, ax2)) = plt.subplots(1, 3, figsize=(18,18))

    # Compare Ortho / Groundtruth / Predicted
    ax0.imshow(X_pred)
    ax0.set_title("Image")
    ax0.axis("off")

    ax1.imshow(y_label)
    ax1.set_title("Ground truth")
    ax1.axis("off")

    ax2.imshow(y_pred)
    ax2.set_title("Predicted label")
    ax2.axis("off")

    plt.show()


def build_model_metrics(model, dataset, number_of_classes, class_names=None, verbose=True):
    """
    Computes evaluation metrics for a semantic segmentation model.

    This function processes an entire dataset to:
      - generate predictions for all images,
      - build a global confusion matrix (num_classes x num_classes),
      - compute the Intersection over Union (IoU) for each class,
      - compute the overall mean IoU (mIoU).

    Parameters
    ----------
    model : tf.keras.Model
        The trained segmentation model used to generate predictions.

    dataset : built with get_tf_dataset for test

    num_classes : int
        Number of segmentation classes.

    class_names : list of str, optional
        Human-readable class names used when printing results.

    verbose : bool
        If True, prints the confusion matrix and per-class IoUs.

    Returns
    -------
    cm : ndarray (num_classes x num_classes)
        The confusion matrix built over all pixels of the dataset.

    iou_per_class : ndarray (num_classes,)
        IoU score for each class.

    miou : float
        The mean IoU computed over all classes (ignoring NaNs).
    """
    y_true_all = []
    y_pred_all = []

    for images, labels in dataset:
        # 1. Predictions from test_set
        preds = model.predict(images, verbose=0)        # (B, H, W, C)
        y_pred = np.argmax(preds, axis=-1)              # (B, H, W)

        # 2. Convert labels into np array
        y_true = labels.numpy()                         # (B, H, W)

        # 3. Flatten arrays to produce one big array
        y_true_all.append(y_true.ravel())
        y_pred_all.append(y_pred.ravel())

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    # 4️⃣ Confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(number_of_classes))

    # 5️⃣ IoU per class
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1) - tp    # For a class i: all true pixels of class i that were misclassified elsewhere
                                # → sum of row i, minus the true positives.
    fp = cm.sum(axis=0) - tp    # For a class i: all pixels predicted as class i that actually belong to another class
                                # → sum of column i, minus the true positives.
    union = tp + fp + fn
    iou_per_class = np.where(union > 0, tp / union, np.nan)
    miou = np.nanmean(iou_per_class)

    if verbose:
        # print("=== Confusion matrix (counts) ===")
        # print(cm)
        print("\n=== IoU per class ===")
        for c in range(number_of_classes):
            name = class_names[c] if class_names is not None else f"class {c}"
            print(f"{name:20s}: IoU = {iou_per_class[c]:.3f}")
        print(f"\n➡ mIoU globale : {miou:.3f}")

    return iou_per_class, miou


def initialize_unet_model(input_shape: tuple = (CHUNK_SIZE, CHUNK_SIZE, 3),
                          number_of_classes: int = 7 if LBL_REDUCTION==True else 16):
    """
    taken from: https://github.com/ChinmayParanjape/Satellite-imagery-segmentation-using-U-NET/blob/main/SEGMENTATION_MODEL_AND%C2%A0_PREPROCESSING.ipynb

    """

    inputs = Input(shape=input_shape)

    s = inputs

    #Contraction path
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = layers.Dropout(0.2)(c1)  # Original 0.1
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.2)(c2)  # Original 0.1
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.2)(c8)  # Original 0.1
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Dropout(0.2)(c9)  # Original 0.1
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = layers.Conv2D(number_of_classes, 1, activation='softmax')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])

    model.model_name = "unet"

    return model



# ======== Convolution Block UNet++ ========
def conv_block(x, filters):
    """
    Standard conv block for UNet/UNet++:
    Conv2D -> BN -> ReLU -> Conv2D -> BN -> ReLU
    """
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


# ======== UNet++ Architecture ========
def initialize_unet_plus_model(
        input_shape: tuple = (CHUNK_SIZE, CHUNK_SIZE, 3),
        number_of_classes: int = 7 if LBL_REDUCTION==True else 16,
        deep_supervision=False,
        base_filters=32
    ):
    """
    U-Net++ model.

    U-Net++ is an enhanced version of the original U-Net architecture.
    It replaces each skip-connection with a series of intermediate convolutional blocks,
    allowing feature maps to be progressively refined before being passed to the decoder.
    This leads to better multi-scale feature fusion, sharper boundaries, and improved
    segmentation of small or complex structures compared to vanilla U-Net.
    """

    inputs = Input(shape=input_shape)

    # ========= ENCODER =========
    x_00 = conv_block(inputs, base_filters)            # 32
    x_10 = conv_block(layers.MaxPooling2D((2, 2))(x_00), base_filters * 2)   # 64
    x_20 = conv_block(layers.MaxPooling2D((2, 2))(x_10), base_filters * 4)   # 128
    x_30 = conv_block(layers.MaxPooling2D((2, 2))(x_20), base_filters * 8)   # 256
    x_40 = conv_block(layers.MaxPooling2D((2, 2))(x_30), base_filters * 16)  # 512

    # ========= DECODER (nested) =========
    # Level j = 1
    x_01 = conv_block(layers.concatenate([
        x_00, layers.UpSampling2D((2, 2))(x_10)
    ]), base_filters)
    x_11 = conv_block(layers.concatenate([
        x_10, layers.UpSampling2D((2, 2))(x_20)
    ]), base_filters * 2)
    x_21 = conv_block(layers.concatenate([
        x_20, layers.UpSampling2D((2, 2))(x_30)
    ]), base_filters * 4)
    x_31 = conv_block(layers.concatenate([
        x_30, layers.UpSampling2D((2, 2))(x_40)
    ]), base_filters * 8)

    # Level j = 2
    x_02 = conv_block(layers.concatenate([
        x_00, x_01, layers.UpSampling2D((2, 2))(x_11)
    ]), base_filters)
    x_12 = conv_block(layers.concatenate([
        x_10, x_11, layers.UpSampling2D((2, 2))(x_21)
    ]), base_filters * 2)
    x_22 = conv_block(layers.concatenate([
        x_20, x_21, layers.UpSampling2D((2, 2))(x_31)
    ]), base_filters * 4)

    # Level j = 3
    x_03 = conv_block(layers.concatenate([
        x_00, x_01, x_02, layers.UpSampling2D((2, 2))(x_12)
    ]), base_filters)
    x_13 = conv_block(layers.concatenate([
        x_10, x_11, x_12, layers.UpSampling2D((2, 2))(x_22)
    ]), base_filters * 2)

    # Level j = 4 (final)
    x_04 = conv_block(layers.concatenate([
        x_00, x_01, x_02, x_03,
        layers.UpSampling2D((2, 2))(x_13)
    ]), base_filters)

    # ========= OUTPUT(S) =========
    if deep_supervision:
        o1 = layers.Conv2D(number_of_classes, 1, activation="softmax")(x_01)
        o2 = layers.Conv2D(number_of_classes, 1, activation="softmax")(x_02)
        o3 = layers.Conv2D(number_of_classes, 1, activation="softmax")(x_03)
        o4 = layers.Conv2D(number_of_classes, 1, activation="softmax")(x_04)
        outputs = layers.Average()([o1, o2, o3, o4])
    else:
        outputs = layers.Conv2D(number_of_classes, 1, activation="softmax")(x_04)

    model = Model(inputs, outputs, name="UNetPlusPlus")
    return model



# def plot_predict(X_pred, y_pred, y_label):

#     y_pred = labels_to_rgb(y_pred, FLAIR_CLASS_DATA)
#     y_label = labels_to_rgb(y_label, FLAIR_CLASS_DATA)

#     fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)) = plt.subplots(3, 3, figsize=(18,18))

#     # Compare Ortho to Label
#     ax0.imshow(X_pred)
#     ax0.set_title("Ortho")
#     ax0.axis("off")

#     ax1.imshow(y_label, cmap="tab20")
#     ax1.set_title("Label")
#     ax1.axis("off")

#     ax2.imshow(X_pred, alpha=0.55)
#     ax2.imshow(y_label, cmap="tab20", alpha=0.45)
#     ax2.set_title("Overlay Ortho/Label")
#     ax2.axis("off")

#     # Compare Ortho to Predict
#     ax3.imshow(X_pred)
#     ax3.set_title("Ortho")
#     ax3.axis("off")

#     ax4.imshow(y_pred, cmap="tab20")
#     ax4.set_title("Predicted")
#     ax4.axis("off")

#     ax5.imshow(X_pred, alpha=0.55)
#     ax5.imshow(y_pred, cmap="tab20", alpha=0.45)
#     ax5.set_title("Overlay Ortho/Predicted")
#     ax5.axis("off")

#     # Compare Label to Predict
#     ax6.imshow(y_label, cmap="tab20")
#     ax6.set_title("Label")
#     ax6.axis("off")

#     ax7.imshow(y_pred, cmap="tab20")
#     ax7.set_title("Predicted")
#     ax7.axis("off")

#     ax8.imshow(y_label, alpha=0.55)
#     ax8.imshow(y_pred, cmap="tab20", alpha=0.45)
#     ax8.set_title("Overlay Label/Predicted")
#     ax8.axis("off")

#     plt.show()

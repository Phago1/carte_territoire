# Here we initialize, compile, train and evaluate the model

from keras import Sequential, layers, metrics, losses, Input
from keras.models import Model
from keras.optimizers import Adam, schedules
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from params import *

def initialize_CNN_model(input_shape: tuple, number_of_classes: int):
    """
    input_shape usualy like X_train.shape[1:]
    repoduced from state-of-the-art recommandation:
    https://deeplearningwithpython.io/chapters/chapter11_image-segmentation/
    """

    model = Sequential()

    model.add(Input(shape=input_shape))

    # model.add(layers.Rescaling(1.0/255)) #to remove when rescale included in pipe

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


def compile_model(model, number_of_classes: int):
    """
    target_class_ids must contain the ids of the classes to be classified.
    It is used by keras.metrics to compute the IoU for each class in this list.
    """
    def combined_loss(y_true, y_pred):
        """
        one-hot-encode y_true so dice and focal losses can be computed
        tf.one_hot need integer inputs while they are originaly float32
        --> tf.cast to encode in int32
        then create a custom function merging dice and focal losses

        """
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=number_of_classes)

        dice_loss = losses.Dice(reduction='sum_over_batch_size', name='dice')
        focal_loss = losses.CategoricalFocalCrossentropy(reduction='sum_over_batch_size', name='focal')

        total_loss = 0.5*dice_loss(y_true_oh, y_pred) + 0.5*focal_loss(y_true_oh, y_pred)

        return total_loss

    lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
        )

    optimizer = Adam(learning_rate=lr_schedule)

    IoU = metrics.MeanIoU(num_classes=number_of_classes, sparse_y_true=True, sparse_y_pred=False)

    model.compile(optimizer=optimizer,
                  loss=combined_loss,
                  metrics=['accuracy',
                           IoU
                           ]
                  )

    return model


def train_model(model, ds_train, ds_val, epochs=30, batch_size = BATCH_SIZE, patience=3):
    """
    train model from train and val tensorfflow datasets
    """

    es = EarlyStopping(patience=patience, restore_best_weights=True)

    history = model.fit(ds_train,
              validation_data=ds_val,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[es]
              )

    return history, model


def predict_model(model, X_pred, input_shape):

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

    X_pred = np.expand_dims(X_pred, axis=0)

    y_pred = model.predict(X_pred)
    y_pred = y_pred.reshape(y_pred[0].shape)
    y_pred = np.argmax(y_pred, axis=-1)

    return y_pred


def plot_predict(X_pred, y_pred, y_label):

    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)) = plt.subplots(3, 3, figsize=(18,18))

    # Compare Ortho to Label
    ax0.imshow(X_pred)
    ax0.set_title("Ortho")
    ax0.axis("off")

    ax1.imshow(y_label, cmap="tab20")
    ax1.set_title("Label")
    ax1.axis("off")

    ax2.imshow(X_pred, alpha=0.55)
    ax2.imshow(y_label, cmap="tab20", alpha=0.45)
    ax2.set_title("Overlay Ortho/Label")
    ax2.axis("off")

    # Compare Ortho to Predict
    ax3.imshow(X_pred)
    ax3.set_title("Ortho")
    ax3.axis("off")

    ax4.imshow(y_pred, cmap="tab20")
    ax4.set_title("Predicted")
    ax4.axis("off")

    ax5.imshow(X_pred, alpha=0.55)
    ax5.imshow(y_pred, cmap="tab20", alpha=0.45)
    ax5.set_title("Overlay Ortho/Predicted")
    ax5.axis("off")

    # Compare Label to Predict
    ax6.imshow(y_label, cmap="tab20")
    ax6.set_title("Label")
    ax6.axis("off")

    ax7.imshow(y_pred, cmap="tab20")
    ax7.set_title("Predicted")
    ax7.axis("off")

    ax8.imshow(y_label, alpha=0.55)
    ax8.imshow(y_pred, cmap="tab20", alpha=0.45)
    ax8.set_title("Overlay Label/Predicted")
    ax8.axis("off")

    plt.show()


def initialize_unet_model(input_shape: tuple, number_of_classes: int):
    """
    taken from: https://github.com/ChinmayParanjape/Satellite-imagery-segmentation-using-U-NET/blob/main/SEGMENTATION_MODEL_AND%C2%A0_PREPROCESSING.ipynb
    """

    inputs = Input(shape=input_shape)
    # inputs = layers.Rescaling(1.0/255)(inputs) #to remove when rescale included in pipe

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

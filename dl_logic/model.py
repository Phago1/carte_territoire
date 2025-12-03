# Here we initialize, compile, train and evaluate the model
# https://keras.io/guides/functional_api/

from keras import Sequential, layers, Input
from keras.optimizers import Adam
from keras import metrics
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

def initialize_model(input_shape: tuple, number_of_classes: int):
    """
    input_shape usualy like X_train[1:]
    no padding implemented at first to simplify output shape management
    """

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    # model.add(layers.MaxPool2D(pool_size=(3, 3)))

    model.add(layers.Conv2D(32, (2,2), padding='same', activation='relu'))
    # model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(16, (2,2), padding='same', activation='relu'))
    # model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(4, (2,2), padding='same', activation='relu'))
    # model.add(layers.MaxPool2D(pool_size=(2, 2)))

    # model.add(layers.Conv2D(256, (2,2), padding='same', strides=(1,1), activation='relu'))
    # model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(number_of_classes, kernel_size=1, activation='softmax'))

    return model


def compile_model(model, target_class_ids: list, learning_rate=0.01):
    """
    target_class_ids must contains the ids of the classes to be classified.
    It is used by keras.metrics to compute the IoU for each class in this list.
    """

    adam = Adam(learning_rate=learning_rate)

    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=[
                      'accuracy',
                      metrics.IoU(num_classes=len(target_class_ids),
                                  target_class_ids=target_class_ids)
                           ]
                  )

    return model


def train_model(model, X, y, validation_size=0.2, shuffle=True, epochs=30, batch_size=16, patience=3):
    """
    Use train_test_split instead od validation_split to shuffle set before splitting.
    Because validation_split take the las X% of the set and then shuffle --> risk of working on a whole blue area.

    y has to be to_categorical
    """

    es = EarlyStopping(patience=patience, restore_best_weights=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, shuffle=shuffle)

    history = model.fit(X_train,
              y_train,
              validation_data=(X_val, y_val),
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[es])

    return history


def predict_model(model, X_pred):

    """
    .predict expect a shape==(batch_size, height, width, 3) so expand_dims is used to
    have a shape==(1, height, width, 3) for a single image.

    X_pred.shape==(height, width, 3)
    y_label.shape==(height,width)

    .predict returns probabilities for each class for each pixel. .argmax() recovers the
    most probable class for each pixel
    """

    X_pred = np.expand_dims(X_pred, axis=0)

    y_pred = model.predict(X_pred)
    y_pred = np.argmax(y_pred[0], axis=-1)

    return y_pred


def plot_predict(X_pred, y_pred, y_label):

    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)) = plt.subplots(3, 3, figsize=(18,18))

    # Compare Ortho to Label
    ax0.imshow(X_pred[0])
    ax0.set_title("Ortho")
    ax0.axis("off")

    ax1.imshow(y_label)
    ax1.set_title("Label")
    ax1.axis("off")

    ax2.imshow(X_pred[0], alpha=0.55)
    ax2.imshow(y_label, cmap="tab20", alpha=0.45)
    ax2.set_title("Overlay Ortho/Label")
    ax2.axis("off")

    # Compare Ortho to Predict
    ax3.imshow(X_pred[0])
    ax3.set_title("Ortho")
    ax3.axis("off")

    ax4.imshow(y_pred, cmap="tab20")
    ax4.set_title("Predicted")
    ax4.axis("off")

    ax5.imshow(X_pred[0], alpha=0.55)
    ax5.imshow(y_pred, cmap="tab20", alpha=0.45)
    ax5.set_title("Overlay Ortho/Predicted")
    ax5.axis("off")

    # Compare Label to Predict
    ax6.imshow(y_label)
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

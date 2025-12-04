# Here we save the results and the model
# We also add a load function

# libraries
from params import *

import pickle
import os
import time

from tensorflow import keras
from google.cloud import storage

import mlflow
from mlflow.tracking import MlflowClient

# functions
# metric Iou

"""
dans le main il faut créer un dictionnaire pour params et metrics aves les valeurs qu'on veut sauvegarder
pour metrics: Iou et ??
dans le main:
Iou = dict(Iou = np.min(history.history['iou']))
params = dict(
        context="train",
        class_number= ...,
        number_of_chunks=len(...),
        reduction_mask = True
)
"""
def save_results(params: dict, metrics: dict):

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")

    # Google Cloud Storage saving
    if MODEL_TARGET == "gcs":
#        try:               # recommended cause uploading can fail
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

            # Upload params
        if params is not None:
            params_blob = bucket.blob(f"training_output/params/{timestamp}.pickle")
            params_blob.upload_from_filename(params_path)

            # Upload metrics
        if metrics is not None:
            metrics_blob = bucket.blob(f"training_output/metrics/{timestamp}.pickle")
            metrics_blob.upload_from_filename(metrics_path)

        print(f"✅ Results uploaded to GCS bucket '{BUCKET_NAME}'")

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    # Save model locally
    if LOCAL_SAVE == 'yes':
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
        model.save(model_path)

        print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None

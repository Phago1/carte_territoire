# function to save the results and parameters of the model
# function to save the model
# function to load the model

# libraries
from carte_territoire_package.params import *

import glob
import pickle
import os
import time
import keras
from colorama import Fore, Style

from keras import models
from google.cloud import storage



"""
dans le main il faut créer un dictionnaire pour params et metrics aves les valeurs qu'on veut sauvegarder
pour metrics: Iou et ??
dans le main:
IoU = dict(Iou = np.min(history.history['iou']))
params = dict(
        context="train",
        class_number= ...,
        number_of_chunks=len(...),
        reduction_mask = True
)
"""

# params and metrics
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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def load_model() -> models:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = models.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="training-outputs/models/"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

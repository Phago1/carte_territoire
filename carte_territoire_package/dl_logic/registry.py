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
import pandas as pd
from colorama import Fore, Style

from keras import models
from google.cloud import storage



"""
Params and Metrics should be defined in main.py and it's recommended to save
them as a dictionary to ease results manupulation.
-par exemple, dans le main:
 IoU = dict(Iou = np.max(history.history['mean_io_u']))
 params = dict(
              context="train",
              class_number= ...,
              number_of_chunks=len(...),
              reduction_mask = True
)
"""

# params and metrics
def save_results(params: dict, metrics: dict):
    """
    Results are saved locally every time and in the bucket if wanted.
    Date and time as the model's name to track training.
    When saved in the bucket, it is done/uploaded from the local file.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        # Ensure folder exists
        os.makedirs(os.path.dirname(params_path), exist_ok=True)
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        # Ensure folder exists
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print(f"✅ Results saved locally. \nPath: {LOCAL_REGISTRY_PATH}")

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

    if MODEL_TARGET == "local":
        print('model only saved locally')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def load_results_to_dataframe():
    """
    Search params and metrics in Bucket which are affiliated to saved models
    Import them in a single DataFrame with 1 line per run
    """
    # import blobs of all models, paramas, and metrics stored in Bucket
    client = storage.Client()

    blobs_models = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models/"))
    blobs_params = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="training_output/params/"))
    blobs_metrics = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="training_output/metrics/"))

    # select params and metrics blobs only related to saved models
    params_list = []
    metrics_list = []
    for blob in blobs_models:
        patern = str(blob)[-17:-10]
        for blob_p in blobs_params:
            if patern in str(blob_p):
                params_list.append(blob_p)
        for blob_m in blobs_metrics:
            if patern in str(blob_m):
                metrics_list.append(blob_m)

    assert len(params_list) == len(metrics_list)

    # select models only related to saved metrics and params
    models_list = []
    for blob in params_list:
        patern = str(blob)[-17:-10]
        for blob_m in blobs_models:
            if patern in str(blob_m):
                models_list.append(blob_m)

    assert len(params_list) == len(models_list)

    # regroup params, metrics and models linked together
    params_metrics_models = list(zip(params_list, metrics_list, models_list))

    # initiate an empty pandas dataframe with columns names
    params_columns = ['context', 'model', 'lbl_reduction', 'chunk_size', 'batch_size', 'learning_rate']
    metrics_columns = ['IoU_train', 'IoU_val', 'IoU_test', 'IoU_per_class_test']
    columns = params_columns + metrics_columns + ['model_blob']

    df = pd.DataFrame(columns=columns)

    # feed the df with values extracted from params and metrics pickles
    # model is stored as a blob and not directly imported to keep a light df
    for blob in params_metrics_models:
        blob_bytes_params = blob[0].download_as_bytes()
        obj_params = pickle.loads(blob_bytes_params)
        df_params = pd.DataFrame([obj_params])

        blob_bytes_metrics = blob[1].download_as_bytes()
        obj_metrics = pickle.loads(blob_bytes_metrics)
        df_metrics = pd.DataFrame([obj_metrics])

        model_blob = blob[1]
        df_models = pd.DataFrame([model_blob], columns=['model_blob'])

        df_blob = pd.concat((df_params, df_metrics, df_models), axis='columns')

        df = pd.concat((df, df_blob), ignore_index=True)

    return df

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def save_model(model: keras.Model = None) -> None:
    """
    Model saved locally and/or in the bucket, set the .env to choose.
    Date and time as the model's name to track training.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.keras")

    if MODEL_TARGET not in {"local", "gcs", "both"}:
        raise ValueError(f"❌ Invalid MODEL_TARGET='{MODEL_TARGET}'. "
                         "Expected one of: 'local', 'gcs', 'both'.")

    # The model must be saved locally to be uploaded in the bucket
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.keras")
    # Ensure folder exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    if MODEL_TARGET == 'local':
        print("✅ Model saved locally")

    elif MODEL_TARGET == "gcs":
        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.keras" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        # for gcs we need a local file to upload, then we can remove it
        try:
            os.remove(model_path)
        except OSError:
            pass
        print("✅ Model saved to GCS")
        return None

    elif MODEL_TARGET == "both":
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

    Return None (but do not Raise) if no model is found

    """
    print(f'Loading from {MODEL_TARGET}')

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        # safety check if local_model_paths is not empty
        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        print(most_recent_model_path_on_disk)
        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = models.load_model(most_recent_model_path_on_disk, compile=False)

        print("✅ Model loaded from local disk")
        return latest_model

    elif MODEL_TARGET in {"gcs", "both"}:
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models/"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = models.load_model(latest_model_path_to_save, compile=False)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

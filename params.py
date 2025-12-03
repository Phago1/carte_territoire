import os

#Size of the chunk to train the model (e.g. 256 x 256)
CHUNK_SIZE = 256

#Threshold of class null > if 0.1 -> chunk with less than 10% of class null is cleared for model training
THRESHOLD_0 = 0.1

# DATA_LOCATION dans .env est - gcs pour google cloud storage
#                             - local
DATA_LOCATION = os.environ.get('DATA_LOCATION')

# MODEL_TARGET dans .env est - mlflow
#                            - gcs
#                            - local
MODEL_TARGET = os.environ.get('MODEL_TARGET')
BUCKET_NAME  = os.environ.get('BUCKET_NAME')

# Where we
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "Code", "Phago1","training_outputs")

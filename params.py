import os

# Size of the chunk to train the model (e.g. 256 x 256)
CHUNK_SIZE=256

# Threshold of class null > if 0.1 -> chunk with less than 10% of class null is cleared for model training
THRESHOLD_0=0.1

# Size of the batch to train the model
BATCH_SIZE=32

# Reduction of number of classes from 15 to 7 : True or False, if True we work in 7 classes
LBL_REDUCTION=False

# DATA_LOCATION dans .env est - gcs pour google cloud storage
#                             - local
DATA_LOCATION = os.environ.get('DATA_LOCATION')

# MODEL_TARGET dans .env est - mlflow
#                            - gcs
#                            - local
MODEL_TARGET = os.environ.get('MODEL_TARGET')
BUCKET_NAME  = os.environ.get('BUCKET_NAME')

# local save
LOCAL_SAVE = os.environ.get('LOCAL_SAVE')
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "Code", "Phago1","training_outputs")

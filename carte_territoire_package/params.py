import os

# Architecture of the model : cnn, unet, or unet_plus
MODEL_ARCH = os.environ.get('MODEL_ARCH')
# Size of the chunk to train the model (e.g. 256 x 256)
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE'))
# Threshold of class null > if 0.1 -> chunk with less than 10% of class null is cleared for model training
THRESHOLD_0 = float(os.environ.get('THRESHOLD_0'))
# Size of the batch to train the model
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
# Reduction of number of classes from 15 to 7 : True or False, if True we work in 7 classes
LBL_REDUCTION = os.environ.get('LBL_REDUCTION')
# Stores patches of (CHUNK_SIZE*CHUNK_SIZE) in cache : True only if 32Go<RAM
CACHE = os.environ.get('CACHE')

LEARNING_RATE = os.environ.get('LEARNING_RATE')

# data in bucket or locally
DATA_LOCATION = os.environ.get('DATA_LOCATION')

# model save locally or in the bucket
MODEL_TARGET = os.environ.get('MODEL_TARGET')
BUCKET_NAME  = os.environ.get('BUCKET_NAME')
# origin model load
MODEL_ORIGIN = os.environ.get('MODEL_ORIGIN')

LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "Code", "Phago1","training_outputs")

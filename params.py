import os

#Size of the chunk to train the model (e.g. 256 x 256)
CHUNK_SIZE = 256

#Threshold of class null > if 0.1 -> chunk with less than 10% of class null is cleared for model training
THRESHOLD_0 = 0.1

# DATA_LOCATION dans .env est soit gcs pour google cloud storage
#                             soit local
DATA_LOCATION = os.environ.get('DATA_LOCATION')

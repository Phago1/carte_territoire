# This module is to be launched to create chunks of all the 42 tiles of the train
# in a specific folder in the bucket train-chunks/

import os
from google.cloud import storage
import rasterio
from rasterio.windows import Window
import numpy as np
import re
from params import *
import matplotlib.pyplot as plt
from dl_logic.preprocessor import slice_to_chunks

# --------------------------------------------------
# GCP setup
# --------------------------------------------------
BUCKET_NAME = os.environ["BUCKET_NAME"]
TRAIN_PREFIX = "train/"
CHUNK_PREFIX = "train-chunks/"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# --------------------------------------------------
# Filename patterns
# --------------------------------------------------
img_regex = re.compile(r"(.+)_res1\.00m\.tif$")
lbl_regex = re.compile(r"(.+)_res1\.00m_labels\.tif$")

# --------------------------------------------------
# List & match image/label pairs in train/
# --------------------------------------------------
all_blobs = list(bucket.list_blobs(prefix=TRAIN_PREFIX))

images = {}
labels = {}

for blob in all_blobs:
    fname = blob.name.split("/")[-1]

    m_img = img_regex.match(fname)
    m_lbl = lbl_regex.match(fname)

    if m_img:
        key = m_img.group(1)
        images[key] = blob
    elif m_lbl:
        key = m_lbl.group(1)
        labels[key] = blob

pairs = [(k, images[k], labels[k]) for k in sorted(images.keys()) if k in labels]
print(f"Found {len(pairs)} image/label pairs.")

# --------------------------------------------------
# Helper to read a TIFF from GCS using rasterio
# --------------------------------------------------
def read_tiff_from_gcs(blob):
    bytes_data = blob.download_as_bytes()
    memfile = rasterio.io.MemoryFile(bytes_data)
    dataset = memfile.open()
    arr = dataset.read()  # shape = (band, H, W)
    profile = dataset.profile
    dataset.close()
    memfile.close()
    return arr, profile

# --------------------------------------------------
# Helper to write array as TIFF to GCS using rasterio
# --------------------------------------------------
def write_tiff_to_gcs(arr, base_profile, blob_path):
    profile = base_profile.copy()
    profile.update(
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0]
    )

    memfile = rasterio.io.MemoryFile()
    with memfile.open(**profile) as dst:
        dst.write(arr)

    bucket.blob(blob_path).upload_from_string(
        memfile.read(),
        content_type="image/tiff"
    )
    memfile.close()

# --------------------------------------------------
# Process all pairs
# --------------------------------------------------
cidx = 1
for idx, (base_name, img_blob, lbl_blob) in enumerate(pairs):

    print(f"\nProcessing: {idx} - {base_name}")

    # Load data
    img_arr, img_profile = read_tiff_from_gcs(img_blob)
    lbl_arr, lbl_profile = read_tiff_from_gcs(lbl_blob)

    # Slice into chunks
    img_arr = np.transpose(img_arr, (1, 2, 0))
    lbl_arr = np.transpose(lbl_arr, (1, 2, 0))[:,:,0]
    img_chunks, lbl_chunks = slice_to_chunks(img_arr, lbl_arr)

    for img_chunk, lbl_chunk in zip(img_chunks, lbl_chunks):

        suffix = f"_chunk_{cidx:05d}.tif"

        img_chunk_name = f"{CHUNK_PREFIX}{base_name}_res1.00m{suffix}"
        lbl_chunk_name = f"{CHUNK_PREFIX}{base_name}_res1.00m_labels{suffix}"

        # Write image chunk
        img_chunki = np.transpose(img_chunk, (2, 0, 1))
        write_tiff_to_gcs(img_chunki, img_profile, img_chunk_name)
        print(f"  ➤ Uploaded {img_chunk_name}")

        # Write label chunk
        lbl_chunki = np.expand_dims(lbl_chunk, 0)
        write_tiff_to_gcs(lbl_chunki, lbl_profile, lbl_chunk_name)
        print(f"  ➤ Uploaded {lbl_chunk_name}")
        cidx += 1

print("\n✔ All chunks uploaded successfully to train-chunks/ (rasterio version)")

# --------------------------------------------------
# Function to retrieve chunks from the bucket all pairs
# --------------------------------------------------
def retrieve_chunks_from_bucket(num_chunk:int):
    '''
    Get chunks from GCP bucket
    Arg : number of chunks returned in a list of tuples (arr_img, arr_lbl)
    Arrays are in rasterio shape
    '''
    CHUNK_PREFIX = "train-chunks/"

    # List all blobs in train-chunks/
    blobs = list(bucket.list_blobs(prefix=CHUNK_PREFIX))

    # Separate image chunks vs label chunks
    img_regex = re.compile(r"(.+)_res1\.00m_chunk_(\d{5})\.tif$")
    lbl_regex = re.compile(r"(.+)_res1\.00m_labels_chunk_(\d{5})\.tif$")

    images = {}
    labels = {}

    for blob in blobs:
        fname = blob.name.split("/")[-1]  # get filename only
        m_img = img_regex.match(fname)
        m_lbl = lbl_regex.match(fname)

        if m_img:
            key = f"{m_img.group(1)}_{m_img.group(2)}"  # base_name + chunk index
            images[key] = blob
        elif m_lbl:
            key = f"{m_lbl.group(1)}_{m_lbl.group(2)}"
            labels[key] = blob

    # Pair them
    pairs = [(images[k], labels[k]) for k in sorted(images.keys()) if k in labels]

    print(f"Found {len(pairs)} image/label chunk pairs.")

    # Example: retrieve arrays
    # from your existing read_tiff_from_gcs function
    output = []
    for img_blob, lbl_blob in pairs[:num_chunk]:  # first 5 chunks
        img_arr, _ = read_tiff_from_gcs(img_blob)
        lbl_arr, _ = read_tiff_from_gcs(lbl_blob)
        print("Image chunk shape:", img_arr.shape, "Label chunk shape:", lbl_arr.shape)
        output.append((img_arr, lbl_arr))

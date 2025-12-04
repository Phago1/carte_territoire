# Here we preprocess the data

import numpy as np
from params import *
from google.cloud import storage
import os
import rasterio
import tensorflow as tf

bucket_name = os.environ["BUCKET_NAME"]

def slice_to_chunks(image: np.ndarray, label: np.ndarray):
    """
    Slices a large image (H, W, C) and its corresponding label (H, W) into
    non-overlapping square chunks of size (chunk_size, chunk_size).
    If 5000 is not divisible by chunk_size, the function will only take
    # the fully divisible top-left area.

    Args:
        image (np.ndarray): The input image array (e.g., 5000x5000x3).
        label (np.ndarray): The input label array (e.g., 5000x5000).
        chunk_size (int): The side length for the square chunks (e.g., 1024).

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: A tuple containing two lists:
            1. A list of image chunks (chunk_size, chunk_size, C).
            2. A list of label chunks (chunk_size, chunk_size).

    All chunks need to have 'class 0' pixel below the threshold_0
    """
    chunk_size=CHUNK_SIZE
    threshold=THRESHOLD_0

    # 1. Input Validation and Dimension Check
    if image.shape[:2] != label.shape:
        raise ValueError(
            f"Image shape {image.shape[:2]} must match label shape {label.shape}."
        )

    H, W, C = image.shape

    # 2. Determine the number of chunks
    # This uses integer division (//) to get the number of full chunks.
    num_h_chunks = H // chunk_size
    num_w_chunks = W // chunk_size

    # 3. Initialize lists for the chunks
    image_chunks = []
    label_chunks = []

    # 4. Loop through the grid and slice the arrays
    for i in range(num_h_chunks):
        for j in range(num_w_chunks):
            # Calculate the starting and ending indices for the current chunk
            h_start = i * chunk_size
            h_end = (i + 1) * chunk_size
            w_start = j * chunk_size
            w_end = (j + 1) * chunk_size

            # Slice the label (H, W)
            lbl_chunk = label[h_start:h_end, w_start:w_end]

            # Test if the chunk is good for training
            if clear_for_chunk(lbl_chunk, threshold):
                label_chunks.append(lbl_chunk)

                # Slice the image (H, W, C)
                img_chunk = image[h_start:h_end, w_start:w_end, :]
                image_chunks.append(img_chunk)
            else:
                continue

    return image_chunks, label_chunks


def images_to_chunks(images: list, labels: list):
    '''
    Creates a set of X and y (ndarrays) from a list of ndarrays images and labels

    Args:
        images (list): list of image arrays (e.g., 5000x5000x3)
        labell (list): list of label arrays (e.g., 5000x5000)
        chunk_size (int): The side length for the square chunks (e.g., 256)

    Returns:
        Tuple containing two ndarrays:
            1. array of image chunks (nb of chunks, chunk_size, chunk_size, C).
            2. array of label chunks (nb of chunks, chunk_size, chunk_size).
    '''
    list_chunks_images = []
    list_chunks_labels = []
    for image, label in zip(images, labels):
        image_chunks, label_chunks = slice_to_chunks(image, label)
        list_chunks_images.extend(image_chunks) #extend the list of chunks to the existing list of chunks
        list_chunks_labels.extend(label_chunks)

    final_images_array = np.stack(list_chunks_images, axis=0)

    # For labels, we need to add a batch dimension before concatenating
    final_labels_array = np.stack(list_chunks_labels, axis=0)

    print(f"You've created a X of shape {final_images_array.shape} and a y of shape {final_labels_array.shape}")

    return final_images_array, final_labels_array


def clear_for_chunk(label_array:np.ndarray, threshold:int):
    '''
    Return True if pixel of class 0 is inferior to the threshold parameter
    False means too many class 0 (outside of boundaries), so no go to use this chunk to train the model
    '''
    num_pixel = label_array.shape[0] * label_array.shape[1]
    num_class0 = np.count_nonzero(label_array==0)
    ratio = num_class0 / num_pixel
    return ratio < threshold


def pairs_crea(prefix:str ="train/"):
    """Parcourt le bucket GCP pour identifier toutes les tuiles orthophotos
    (ici celles dont le nom se termine par "res1.00m.tif").
    Pour chaque tuile trouvée, on construit automatiquement avec glob le chemin
    du fichier de labels associé en ajoutant le suffixe "_labels.tif".
    La fonction check si le label existe puis renvoie une liste de paires (orthophoto, labels)
    qu'on peut ensuite appeler comme dans une liste classique"""

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Liste les clés présentes dans le bucket
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    all_names = {blob.name for blob in blobs}

    ortho_files = [name for name in all_names if name.endswith("res1.00m.tif")]

    pairs_ortho = []
    pairs_labels = []

    for ortho in ortho_files:
        stem = ortho[:-4]   # remove ".tif"
        label = stem + "_labels.tif"

        if label in all_names:
            pairs_ortho.append(
                f"gs://{bucket_name}/{ortho}"
                )
            pairs_labels.append(
                f"gs://{bucket_name}/{label}"
            )
        else:
            print(f"Label manquant dans train/ pour : {ortho}")

    return (pairs_ortho, pairs_labels)


def chunk_generator(prefix:str):
    """
    Function to avoid overloading RAM

    Generator that:
      - iterates through tile pairs from the GCP bucket,
      - loads each tile with rasterio,
      - calls slice_to_chunks(),
      - yields each chunk one by one (image_chunk, label_chunk).
    """
    ortho_paths, label_paths = pairs_crea(prefix=prefix)
    if prefix == "train/":   # TO REMOVE WHEN GOING FULL SCALE
        ortho_paths = ortho_paths[5:10]
        label_paths = label_paths[5:10]

    for ortho_path, label_path in zip(ortho_paths, label_paths):
        with rasterio.open(ortho_path) as src_o:
            image = src_o.read()
            image = np.transpose(image, (1, 2, 0)).astype("float32") / 255.0

        with rasterio.open(label_path) as src_l:
            label = src_l.read(1).astype("int32")

        img_chunks, lab_chunks = slice_to_chunks(image, label)

        for img_chunk, lab_chunk in zip(img_chunks, lab_chunks):
            yield img_chunk, lab_chunk


def get_tf_dataset(
    prefix: str,
    batch_size: int = BATCH_SIZE,
    shuffle_buffer: int = 1024) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from tiles under the given prefix ('train/' or 'val/').

    The dataset yields (X, y) batches directly consumable by model.fit().
    """

    ds = tf.data.Dataset.from_generator(
        lambda: chunk_generator(prefix),
        output_signature=(
            tf.TensorSpec(
                shape=(CHUNK_SIZE, CHUNK_SIZE, 3),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(CHUNK_SIZE, CHUNK_SIZE),
                dtype=tf.int32
            ),
        )
    )

    ds = ds.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

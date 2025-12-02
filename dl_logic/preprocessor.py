# Here we preprocess the data
#TODO: includes stride parameters + remove chunks where 'no class' > 10% or other ?
import numpy as np


def slice_to_chunks(image: np.ndarray, label: np.ndarray, chunk_size: int):
    """
    Slices a large image (H, W, C) and its corresponding label (H, W) into
    non-overlapping square chunks of size (chunk_size, chunk_size).

    Args:
        image (np.ndarray): The input image array (e.g., 5000x5000x3).
        label (np.ndarray): The input label array (e.g., 5000x5000).
        chunk_size (int): The side length for the square chunks (e.g., 1024).

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: A tuple containing two lists:
            1. A list of image chunks (chunk_size, chunk_size, C).
            2. A list of label chunks (chunk_size, chunk_size).
    """

    # 1. Input Validation and Dimension Check
    if image.shape[:2] != label.shape:
        raise ValueError(
            f"Image shape {image.shape[:2]} must match label shape {label.shape}."
        )

    H, W, C = image.shape

    # Ensure the image and label dimensions are divisible by the chunk_size.
    # If 5000 is not divisible by chunk_size, the function will only take
    # the fully divisible top-left area.
    if H % chunk_size != 0 or W % chunk_size != 0:
        print(f"Warning: Dimensions ({H}x{W}) are not perfectly divisible by "
              f"chunk_size ({chunk_size}). Only the largest fully covered "
              f"area will be sliced. (e.g., 4x4 chunks if 5000/1024 = 4.88)")

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

            # Slice the image (H, W, C)
            img_chunk = image[h_start:h_end, w_start:w_end, :]
            image_chunks.append(img_chunk)

            # Slice the label (H, W)
            lbl_chunk = label[h_start:h_end, w_start:w_end]
            label_chunks.append(lbl_chunk)

    return image_chunks, label_chunks


def images_to_chunks(images: list, labels: list, chunk_size:int):
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
        image_chunks, label_chunks = slice_to_chunks(image, label, chunk_size=chunk_size)
        list_chunks_images.extend(image_chunks) #extend the list of chunks to the existing list of chunks
        list_chunks_labels.extend(label_chunks)

    final_images_array = np.stack(list_chunks_images, axis=0)

    # For labels, we need to add a batch dimension before concatenating
    final_labels_array = np.stack(list_chunks_labels, axis=0)

    print(f"You've created a X of shape {final_images_array.shape} and a y of shape {final_labels_array.shape}")

    return final_images_array, final_labels_array

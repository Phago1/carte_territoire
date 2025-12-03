# Here, we load the data
# The data is clean normally but just in case

from pathlib import Path
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.transform import Affine

import matplotlib.pyplot as plt
import numpy as np
import random

# Fonction pour visualiser une tuile ortho et la tuile label correspondante.
# Donner en argument une tuile ou une liste de tuiles SANS le .tif ou le _labels.tif
# Ex."90-2023-0990-6740-LA93-0M20-IRC-E080_res1.00m"

# Modifier le file_path avec le path de votre dossier en local
file_path = Path("/home/pboui/code/Phago1/carte_territoire/raw_data/")

def tiles_viz(tile_name: str, file_path: Path):
    ortho_path = file_path / f"{tile_name}.tif"
    label_path = file_path / f"{tile_name}_labels.tif"

    print("\n=== Vérification :", tile_name, "===\n")

    # Charger ortho (fichiers 3 classes)
    with rasterio.open(ortho_path) as src_o:
        ortho = src_o.read()
        profile_o = src_o.profile
        bounds_o = src_o.bounds

    # Charger labels (fichiers CoSIA 1 classe)
    with rasterio.open(label_path) as src_l:
        labels = src_l.read(1)
        profile_l = src_l.profile
        bounds_l = src_l.bounds

    # --- Vérifications : check des params des fichiers ---
    print("Shapes (ortho vs labels) :", ortho.shape, labels.shape)
    print("Resolution ortho :", profile_o['transform'][0])
    print("Resolution labels:", profile_l['transform'][0])
    print("Bounds ortho :", bounds_o)
    print("Bounds labels:", bounds_l)

    # --- Zoom identique : on extrait une window au milieu de la carte
    # représentant 1/3 de sa taille ---
    h, w = labels.shape
    y0, y1 = h//4, h//4 + h//3
    x0, x1 = w//4, w//4 + w//3

    # Transposition de l'ortho pour passer de (3,5000,5000) à (5000,5000,3) pour
    # faciliter l'affichage ensuite avec plt. On laisse le label tel quel
    ortho_zoom = np.transpose(ortho[:, y0:y1, x0:x1], (1,2,0))
    labels_zoom = labels[y0:y1, x0:x1]

    # --- Affichage avec plt ---
    fig, ax = plt.subplots(1, 3, figsize=(18,6))

    ax[0].imshow(ortho_zoom / 255)
    ax[0].set_title("Zoom ortho")
    ax[0].axis("off")

    ax[1].imshow(labels_zoom, cmap="tab20")
    ax[1].set_title("Zoom labels")
    ax[1].axis("off")

    ax[2].imshow(ortho_zoom / 255, alpha=0.55)
    ax[2].imshow(labels_zoom, cmap="tab20", alpha=0.45)
    ax[2].set_title("Overlay")
    ax[2].axis("off")

    plt.show()

    # --- Analyse classes ---
    u, c = np.unique(labels, return_counts=True)
    print("Classes uniques labels:", u[:20], "..." if len(u)>20 else "")
    print("Pixels non-nuls :", (labels>0).sum())
    print("OK si > 10 000 pixels non-nuls\n")


# Illustration

# tiles_to_test = [
#     "90-2023-0990-6740-LA93-0M20-IRC-E080_res1.00m",
#     "90-2023-0995-6740-LA93-0M20-IRC-E080_res1.00m"
# ]

# for t in tiles_to_test:
#     tiles_viz(t, file_path)



def label_tiles_info(prefix:str,suffix:str):
    """
    Inspect label tiles stored in a Google Cloud Storage bucket.

    This function:
    - lists all objects in the bucket whose path starts with the given `prefix`
      (e.g., "train/"),
    - filters them to keep only those whose filename ends with the given `suffix`
      (e.g., "_labels.tif"),
    - opens each matching GeoTIFF remotely via rasterio (using a gs:// URI),
    - computes basic statistics on the raster mask:
        - unique class values,
        - pixel counts per class,
        - percentage of non-zero pixels,
    - prints a short summary for each label tile.

    Parameters
    ----------
    prefix : str
        Path prefix inside the bucket (simulates a directory). Example: "train/".
    suffix : str
        Filename suffix identifying label files. Example: "_labels.tif".
    """

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # List keys and files in bucket, with prexix train
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    all_names = {blob.name for blob in blobs}

    label_files = [
        f"gs://{bucket_name}/{name}"
        for name in all_names
        if name.endswith(suffix)]

    print("Nb tuiles trouvées :", len(label_files))

    for lf in label_files:
        with rasterio.open(lf) as src:
            lab = src.read(1)
            u, c = np.unique(lab, return_counts=True)
            maxv = int(lab.max())
            minv = int(lab.min())
            nonzero = int((lab > 0).sum())
            total = lab.size
            pct_nonzero = nonzero / total * 100
            weights = c / total * 100

        print(f"\n{lf.split('/')[-1]}")
        print("  nb classes uniques :", len(u))
        print("  % de pixels non-nuls :", f"{round(pct_nonzero, 1)}%")
        print("  classes présentes (valeurs) :", u[:20], "..." if len(u)>20 else "")
        print("  poids classes (%) :", [round(w, 1) for w in weights])

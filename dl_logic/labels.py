# dictionary from the FLAIR challenge
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt

class Label:
    """Represents a single land-cover class."""
    id: int          # numeric ID used in masks
    name: str        # color name
    color: str

# original class definition by CoSIA
flair_data = {
1   : ['building','#db0e9a'] ,
2   : ['pervious surface','#938e7b'],
3   : ['impervious surface','#f80c00'],
4   : ['bare soil','#a97101'],
5   : ['water','#1553ae'],
6   : ['coniferous','#194a26'],
7   : ['deciduous','#46e483'],
8   : ['brushwood','#f3a60d'],
9   : ['vineyard','#660082'],
10  : ['herbaceous vegetation','#55ff00'],
11  : ['agricultural land','#fff30d'],
12  : ['plowed land','#e4df7c'],
13  : ['swimming_pool','#3de6eb'],
14  : ['snow','#ffffff'],
15  : ['clear cut','#8ab3a0'],
16  : ['mixed','#6b714f'],
17  : ['ligneous','#c5dc42'],
18  : ['greenhouse','#9999ff'],
19  : ['other','#000000'],
}

# Reduced schema
"""
for this reduce classification:
- impervious surface counts railroads, roads, sport fields...
at first, let's merge this with pervious surface
- we merge swimming pools and water surfaces
- vineyards, agricultural land and plowed land have the same pattern, lines
we merge them in one class, agriculture
- at first we also merge all green surfaces, man made like herbaceous vegetation
and wild like forest (coniferous and deciduous)
- greenhouse can be added either to agriculture or buildings. for now, we put it in buildings
"""
REDUCED_7 = {
    1: [1, "building", "#db0e9a"],
    2: [2, "built surface", "#938e7b"],
    3: [3, "bare soil", "#a97101"],
    4: [4, "water", "#1553ae"],
    5: [5, "vegetation", "#00a651"],
    6: [6, "agriculture", '#660082'],
    7: [7, "other", "#000000"],
}

# mapping from original to reduce

COSIA19_TO_REDUCED7 = {
    # unchanged
    3: 3,

    # buildings
    1: 1,
    18: 1,

    # surface
    2: 2,
    3: 2,

    # water-like
    5: 4,
    13: 4,

    # vegetation (coniferous, deciduous, brushwood, herbaceous vegetation)
    6: 5,
    7: 5,
    8: 5,
    10: 5,

    # agriculture
    9 : 6,
    11 : 6,
    12 : 6,

    # everything else to 'other'
    14: 7,
    15: 7,
    16: 7,
    17: 7,
    19: 7,
}


# -----------------------------------------------------
#  Target statistics
# -----------------------------------------------------
def count_class(mask):
    """
    count_class takes one image, gives the distribution of class
    and returns a dictionary
    """
    values, counts = np.unique(mask, return_counts=True)
    result = {}
    for v, c in zip(values, counts):
        result[int(v)] = int(c)
    return result

def merge_counts(total_counts, new_counts):
    """
    Docstring pour merge_counts

    :param total_counts: Description
    :param new_counts: Description
    merge_counts merge two dictionaries
    """
    for cls, cnt in new_counts.items():
        if cls in total_counts:
            total_counts[cls] += cnt
        else:
            total_counts[cls] = cnt

def compute_dataset_class_stats(labels_dir, suffix="_labels.tif"):
    """
    Docstring pour compute_dataset_class_stats

    :param labels_dir: where the data is
    :param suffix: by default it is '_labels.tif' to select only the targets

    this function gives us the total value count for each class
    and their percentage
    """
    # to check the directory
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        print(f"Directory not found: {labels_dir}")

    label_paths = sorted(
        p for p in labels_dir.iterdir()
        if p.is_file() and p.name.endswith(suffix)
    )

    if not label_paths:
        print(
            f"No label files ending with '{suffix}' found in {labels_dir}"
        )

    total_counts = {}
    total_pixels = 0

    print(f"Found {len(label_paths)} label files in {labels_dir}\n")

    for path in label_paths:
        print(f"Processing {path.name} ...")
        with rasterio.open(path) as src:
            # assuming one-band label TIFF
            mask = src.read(1)

        file_counts = count_class(mask)
        merge_counts(total_counts, file_counts)
        total_pixels += sum(file_counts.values())

    print("\n=== Global class distribution ===")
    print(f"Total pixels: {total_pixels}")
    print("class_id,count,percentage")

    for cls in sorted(total_counts.keys()):
        count = total_counts[cls]
        pct = 100.0 * count / total_pixels if total_pixels > 0 else 0.0
        print(f"{cls},{count},{pct:.4f}")

    return total_counts


# -----------------------------------------------------
#  Reduce Mask Visualization
# -----------------------------------------------------
# 1. Table de réduction 19 → 7


def build_table(mapping, max_src=19, default=0):
    lut = np.full(max_src + 1, default, dtype=np.uint8)
    for old, new in mapping.items():
        lut[old] = new
    return lut

LUT_19_TO_7 = build_table(COSIA19_TO_REDUCED7)

def reduce_mask(mask):
    return LUT_19_TO_7[mask]


def visualize_label_reduction(image_path, label_path, title=None):
    """
    Affiche côte à côte :
      - l'image aérienne (3 canaux)
      - le masque original (19 classes)
      - le masque réduit (7 classes)
    """

    # Charger l'image (on suppose 3 canaux dans le TIF)
    with rasterio.open(image_path) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)  # (H, W, 3)

    # Charger le masque 19 classes
    with rasterio.open(label_path) as src:
        mask19 = src.read(1)

    # Réduction 19 → 7
    mask7 = reduce_mask(mask19)

    # Affichage
    plt.figure(figsize=(16, 6))

    if title is not None:
        plt.suptitle(title, fontsize=14)

    plt.subplot(1, 3, 1)
    plt.title("Image aérienne (RGB)")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Masque original (19 classes)")
    plt.imshow(mask19, cmap="tab20")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Masque réduit (7 classes)")
    plt.imshow(mask7, cmap="tab10")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

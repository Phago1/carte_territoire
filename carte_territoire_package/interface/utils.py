import numpy as np

def labels_to_rgb(label_array, color_map):
    """
    Converts a 2D label array (class IDs) into a 3D RGB image array
    based on a provided color map.

    Args:
        label_array (np.ndarray): A 2D NumPy array of shape (H, W)
                                  containing integer class IDs (0-15).
        color_map (dict): A dictionary where keys are class IDs and
                          values are lists/tuples like [name, '#hex_color'].

    Returns:
        np.ndarray: A 3D NumPy array of shape (H, W, 3) representing
                    the RGB image, with dtype=np.uint8.
    """
    # 1. Get the dimensions (H=Height, W=Width)
    H, W = label_array.shape

    # 2. Extract the RGB values from the color map and convert them
    #    from hex strings to a list of (R, G, B) tuples.
    #    We assume the color map is ordered by the class ID keys (0-15).

    # Function to convert '#RRGGBB' to (R, G, B) integer tuple
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)

    # Create a lookup array for the colors. This is the core for fast mapping.
    # We use a 16x3 array, where row 'i' stores the RGB for class 'i'.
    # Ensure the keys are sorted to correctly build the lookup table.
    sorted_keys = sorted(color_map.keys())

    # Pre-allocate the lookup table for 16 classes (0-15)
    color_lookup = np.zeros((len(sorted_keys), 3), dtype=np.uint8)

    for i, class_id in enumerate(sorted_keys):
        # The second element of the map's value is the hex color string
        hex_color = color_map[class_id][1]
        color_lookup[class_id] = hex_to_rgb(hex_color)

    # 3. Use NumPy's advanced indexing to map the labels to the colors.
    #    - label_array (H, W) contains indices (0-15).
    #    - color_lookup[label_array] selects the RGB color (3 channels)
    #      for each pixel using the class ID as the row index.
    rgb_image = color_lookup[label_array]

    # 4. Return the resulting RGB image array
    return rgb_image



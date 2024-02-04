import numpy as np
from scipy.ndimage import zoom
from constants import height, width


def wrap(arr):
    # Define a 3x3 patch of 0.5 with middle 1
    patch = 0.5 * np.ones((3, 3))
    patch[1, 1] = 1

    h, w = arr.shape

    # Find the indices of ones in the input array
    ones_indices = np.argwhere(arr == 1)

    new_arr = np.copy(arr)
    for idx in ones_indices:
        i, j = idx
        if i == 0:
            patch = np.ones((2, 3))
            new_arr[i:i + 2, j - 1:j + 2] = patch
        elif i == h-1:
            patch = np.ones((2, 3))
            new_arr[i - 1:i + 1, j - 1:j + 2] = patch
        elif j == 0:
            patch = np.ones((3, 2))
            new_arr[i - 1:i + 2, j:j + 2] = patch
        elif j == w-1:
            patch = np.ones((3, 2))
            new_arr[i - 1:i + 2, j - 1:j + 1] = patch
        else:
            patch = np.ones((3, 3))
            new_arr[i - 1:i + 2, j - 1:j + 2] = patch

    return new_arr

def resize(frame_data, new_height=height, new_width=width):
    current_height, current_width, num_features = frame_data.shape
    
    # Calculate the scaling factors for height and width
    height_scale = new_height / current_height
    width_scale = new_width / current_width

    # Resize the frame data using bilinear interpolation
    resized_data = zoom(frame_data, (height_scale, width_scale, 1), order=1)

    return resized_data
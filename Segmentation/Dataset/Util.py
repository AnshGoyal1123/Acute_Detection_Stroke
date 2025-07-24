import numpy as np
import random

def random_crop_around_lesion(ct_array, mri_array, label_array, lesion_index, crop_size=(56, 56, 56), random_crop_prob=0.0):
    """
    Return a crop around a lesion voxel, with optional random offset.

    Inputs:
    - ct_array, mri_array, label_array: 4D numpy arrays with shape [1, D, H, W]
    - lesion_index: (1, x, y, z) tuple (note: first dim is channel)
    - crop_size: desired output size (D, H, W)
    - random_crop_prob: float [0, 1] chance to add random offset

    Returns:
    - Cropped versions of ct_array, mri_array, label_array
    """
    _, x, y, z = lesion_index
    cd, ch, cw = crop_size
    D, H, W = ct_array.shape[1:]

    if random.random() < random_crop_prob:
        x += np.random.randint(-cd//4, cd//4 + 1)
        y += np.random.randint(-ch//4, ch//4 + 1)
        z += np.random.randint(-cw//4, cw//4 + 1)

    x = np.clip(x - cd // 2, 0, D - cd)
    y = np.clip(y - ch // 2, 0, H - ch)
    z = np.clip(z - cw // 2, 0, W - cw)

    ct_crop = ct_array[:, x:x+cd, y:y+ch, z:z+cw]
    mri_crop = mri_array[:, x:x+cd, y:y+ch, z:z+cw]
    label_crop = label_array[:, x:x+cd, y:y+ch, z:z+cw]

    return ct_crop.astype(np.float32), mri_crop.astype(np.float32), label_crop.astype(np.float32)
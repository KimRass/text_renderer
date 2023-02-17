from pathlib import Path
import numpy as np
import cv2
from numba import jit
from scipy import ndimage as ndi


def _insert_seams(img, dx, pmask=None, rmask=None, visualize=False, rotate=False):
    copied_img = img.copy()
    copied_pmask = pmask.copy() if pmask is not None else None
    copied_rmask = rmask.copy() if rmask is not None else None

    seam_idx_record = list()
    for _ in tqdm(range(dx)):
        seam_idx, seam_mask = _get_seam_with_lowest_energy(img=copied_img, pmask=copied_pmask, rmask=copied_rmask)
        # if visualize:
        #     visualize_process(img=copied_img, seam_mask=seam_mask, rotate=rotate)
        seam_idx_record.append(seam_idx)

        copied_img = _get_seam_carved_image(img=copied_img, seam_mask=seam_mask)
        copied_pmask = _get_seam_carved_mask(mask=copied_pmask, seam_mask=seam_mask)
        copied_rmask = _get_seam_carved_mask(mask=copied_rmask, seam_mask=seam_mask)

    copied_img = img.copy()
    copied_pmask = pmask.copy() if pmask is not None else None
    copied_rmask = rmask.copy() if rmask is not None else None

    seam_idx_record.reverse()
    for _ in tqdm(range(dx)):
        seam_idx = seam_idx_record.pop()
        if visualize:
            # print(copied_img.shape, seam_mask.shape)
            visualize_process(img=copied_img, seam_mask=seam_mask, rotate=rotate)
        copied_img = _get_seam_inserted_image(img=copied_img, seam_idx=seam_idx)
        if copied_pmask is not None:
            copied_pmask = _get_seam_inserted_mask(mask=copied_pmask, seam_idx=seam_idx)
        if copied_rmask is not None:
            copied_rmask = _get_seam_inserted_mask(mask=copied_rmask, seam_idx=seam_idx)

        # Update the remaining seam indices
        for remaining_seam in seam_idx_record:
            remaining_seam[remaining_seam >= seam_idx] += 2
    return copied_img, copied_pmask, copied_rmask

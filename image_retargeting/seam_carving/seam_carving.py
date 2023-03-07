from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import cv2
import numba as nb
from numba import jit, njit
from scipy import ndimage as ndi
from wand.image import Image as wandImage
from wand.drawing import Drawing
from wand.color import Color
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

from process_images import (
    load_image,
    save_image,
    show_image,
    _get_width_and_height,
    _convert_to_2d,
    _convert_to_3d,
    _get_canvas_same_size_as_image
)
from utilities import (
    parse_transcription_df
)
from render_texts import (
    _get_textbox_width_and_height
)


def _convert_to_grayscale(img):
    """Convert an RGB image to a grayscale image"""
    coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
    return (img @ coeffs).astype(img.dtype)


def _get_energy(img: np.ndarray) -> np.ndarray:
    gray_img = _convert_to_grayscale(img)
    
    """Get backward energy map from the source image"""
    assert gray_img.ndim == 2

    # gray_img = gray_img.astype(np.float32)
    grad_x = sobel(gray_img, axis=1)
    grad_y = sobel(gray_img, axis=0)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy


def _get_backward_energy_map(img, gray=False):
    if not gray:
        copied_img = img.copy()
    else:
        copied_img = _convert_to_grayscale(img)
    xgrad = ndi.convolve1d(copied_img, weights=np.array([1, 0, -1]), axis=1, mode="wrap")
    ygrad = ndi.convolve1d(copied_img, weights=np.array([1, 0, -1]), axis=0, mode="wrap")
    if not gray:
        grad_mag = np.sqrt(
            np.sum(xgrad ** 2, axis=2) + np.sum(ygrad ** 2, axis=2)
        )
    else:
        grad_mag = np.sqrt(xgrad ** 2 + ygrad ** 2)
    return grad_mag


def show_energy_map(energy_map) -> None:
    copied = energy_map.copy()
    copied -= copied.min()
    copied /= copied.max()
    copied *= 255
    copied = copied.astype("uint8")
    show_image(copied)


def _rotate_image(img, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(img, k)


def visualize_process(img, seam_mask, rotate=False) -> None:
    if img.ndim == 3:
        vis = img.copy()[:, :, :: -1].astype("uint8")
    elif img.ndim == 2:
        vis = img.copy().astype("uint8")

    if seam_mask is not None:
        if img.ndim == 3:
            vis[seam_mask == False] = np.array([0, 0, 255])
        elif img.ndim == 2:
            vis[seam_mask == False] = 0

    if rotate:
        vis = _rotate_image(img=vis, clockwise=False)

    width, height = _get_width_and_height(img)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.moveWindow("output", x=0, y=0)
    cv2.resizeWindow("output", width=int(1_200 * width / height), height=1_200)
    cv2.imshow("output", vis)
    cv2.waitKey(1)


# @njit(
#     nb.types.Tuple((nb.int16[:], nb.boolean[:, :]))(nb.int8[:, :, :], nb.int8[:, :], nb.int8[:, :])
# )
@jit
def _get_seam_with_lowest_energy(img, pmask, rmask, text_alignment, xmin, ymin, xmax, ymax):
    energy_map = _get_backward_energy_map(img, gray=False)
    energy_map[rmask == 255] = -100
    energy_map[pmask == 255] = 100

    if text_alignment == "left":
        energy_map[ymin: ymin + textbox_height, xmax - 1: xmax] = -10_000

    height, width, _ = img.shape
    backtrack = np.zeros_like(energy_map, dtype="int")
    ### Populate DP matrix
    for row in range(1, height):
        for col in range(0, width):
            if col == 0:
                idx = np.argmin(energy_map[row - 1, col: col + 2])
                backtrack[row, col] = idx + col
                least_energy = energy_map[row - 1, idx + col]
            else:
                idx = np.argmin(energy_map[row - 1, col - 1: col + 2])
                backtrack[row, col] = idx + (col - 1)
                least_energy = energy_map[row - 1, idx + col - 1]
            energy_map[row, col] += least_energy

    ### Backtrack to find path
    seam = list()
    seam_mask = np.ones((height, width), dtype="bool")
    col = np.argmin(energy_map[-1])
    for row in range(height - 1, -1, -1):
        col = backtrack[row, col]

        seam_mask[row, col] = False
        seam.append(col)
    seam.reverse()
    return np.array(seam, dtype="uint16"), seam_mask


@jit
def _get_seam_carved_image(img, seam_mask):
    height, width, _ = img.shape
    seam_mask_3d = _convert_to_3d(seam_mask)
    # Delete all the pixels marked False in the mask and resize it to the new image dimensions
    result = img[seam_mask_3d].reshape((height, width - 1, 3))
    return result


@jit
def _get_seam_carved_mask(mask, seam_mask):
    height, width = mask.shape
    return mask[seam_mask].reshape((height, width - 1))


# @njit(nb.int8[:, :, :](nb.int8[:, :, :], nb.int16[:]))
@jit
def _get_seam_inserted_image(img, seam):
    """
    Add a vertical seam to a 3-channel color image
    at the indices provided by averaging the pixels values to the left and right of the seam.
    """
    height, width, _ = img.shape
    output = np.zeros((height, width + 1, 3), dtype="uint8")
    for row in range(height):
        col = seam[row]
        for ch in range(3):
            if col == 0:
                p = np.average(img[row, col: col + 2, ch])
                output[row, col, ch] = p
                output[row, col + 1:, ch] = img[row, col:, ch]
            else:
                p = np.average(img[row, col - 1: col + 1, ch])
                output[row, : col, ch] = img[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = img[row, col:, ch]
    return output


# @njit(nb.int8[:, :](nb.int8[:, :], nb.int16[:]))
@jit
def _get_seam_inserted_mask(mask, seam):
    """
    Add a vertical seam to a mask
    at the indices provided by averaging the pixel values to the left and right of the seam.
    """
    height, width = mask.shape
    output = np.zeros((height, width + 1), dtype="uint8")
    for row in range(height):
        col = seam[row]
        if col == 0:
            p = np.average(mask[row, col: col + 2])
            output[row, col] = p
            output[row, col + 1:] = mask[row, col:]
        else:
            p = np.average(mask[row, col - 1: col + 1])
            output[row, : col] = mask[row, : col]
            output[row, col] = p
            output[row, col + 1:] = mask[row, col:]
    return output


def _insert_seams(img, dx, pmask, rmask, visualize=False, rotate=False):
    copied_img = img.copy()
    copied_pmask = pmask.copy() if pmask is not None else None
    copied_rmask = rmask.copy() if rmask is not None else None

    seam_record = list()
    for _ in tqdm(range(dx)):
        seam, seam_mask = _get_seam_with_lowest_energy(img=copied_img, pmask=copied_pmask, rmask=copied_rmask)
        if visualize:
            visualize_process(img=copied_img, seam_mask=seam_mask, rotate=rotate)

        copied_img = _get_seam_inserted_image(img=copied_img, seam=seam)
        copied_pmask = _get_seam_inserted_mask(mask=copied_pmask, seam=seam)
        copied_rmask = _get_seam_inserted_mask(mask=copied_rmask, seam=seam)

        seam_record.append(seam)
    return copied_img, copied_pmask, copied_rmask, seam_record



def get_dx_and_dy(xmin, ymin, xmax, ymax, font_size, text_alignment, text):
    ctx = Drawing()
    ctx.font_size = font_size
    if text_alignment == "left":
        textbox_width, textbox_height = _get_textbox_width_and_height(ctx=ctx, text=text)
        dx = abs(textbox_width - (xmax - xmin))
        dy = 0
    elif text_alignment == "top":
        textbox_width, textbox_height = _get_textbox_width_and_height(ctx=ctx, text=text.replace("", "\n"))
        dx = 0
        dy = abs(textbox_height - (ymax - ymin))
    return dx, dy, textbox_width, textbox_height



if __name__ == "__main__":
    _, img, img_url = parse_transcription_df(
        "/Users/jongbeomkim/Downloads/20239214_for_jongbeom.csv", index=0
    )
    bboxes = pd.read_excel("/Users/jongbeomkim/Desktop/workspace/text_renderer/seam_carving/1119_3752/1119_3752.xlsx")
    text_removed_img = load_image("/Users/jongbeomkim/Desktop/workspace/text_renderer/seam_carving/1119_3752/1119_3752_text_removed2.jpg")
    temp_img = text_removed_img.copy()

    init_pmask = load_image("/Users/jongbeomkim/Desktop/workspace/text_renderer/seam_carving/1119_3752/1119_3752_mask2.png", gray=True)
    temp_pmask = init_pmask.copy()

    bboxes[["dx", "dy", "textbox_width", "textbox_height"]] = bboxes.apply(
        lambda x: pd.Series(
            get_dx_and_dy(
                xmin=x["xmin"],
                ymin=x["ymin"],
                xmax=x["xmax"],
                ymax=x["ymax"],
                font_size=x["font_size"],
                text_alignment=x["text_alignment"],
                text=x["tr_content"]
            )
        ),
        axis=1
    )

    init_rmask = _get_canvas_same_size_as_image(_convert_to_2d(temp_img), black=True)
    for idx, (
        xmin, ymin, xmax, ymax, ori_content, tr_content, font_size, text_alignment, dx, dy, textbox_width, textbox_height
    ) in enumerate(bboxes.values):
        if text_alignment == "left":
            init_rmask[ymin: ymin + textbox_height, xmax - 1: xmax] = 255
        elif text_alignment == "top":
            init_rmask[ymax - 1: ymax, xmin: xmin + textbox_width] = 255
    temp_rmask = init_rmask.copy()

    for idx, (
        xmin, ymin, xmax, ymax, ori_content, tr_content, font_size, text_alignment, dx, dy, textbox_width, textbox_height
    ) in enumerate(bboxes.values):
        print(idx, ori_content, tr_content)
        for _ in tqdm(range(dx)):
            seam, seam_mask = _get_seam_with_lowest_energy(
                img=temp_img,
                pmask=temp_pmask,
                rmask=temp_rmask,
                text_alignment=text_alignment,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax
            )
            visualize_process(img=temp_img, seam_mask=seam_mask, rotate=rotate)

            temp_img = _get_seam_inserted_image(img=temp_img, seam=seam)
            temp_pmask = _get_seam_inserted_mask(mask=temp_pmask, seam=seam)
            temp_rmask = _get_seam_inserted_mask(mask=temp_rmask, seam=seam)

            # bboxes["xmin"] = bboxes.apply(
            #     lambda x: x["xmin"] + int(
            #         sum(seam[x["ymin"]: x["ymin"] + x["textbox_height"]] <= x["xmin"]) >= x["textbox_height"] / 2
            #     ),
            #     axis=1
            # )
            # bboxes["xmax"] = bboxes.apply(
            #     lambda x: x["xmax"] + int(
            #         sum(seam[x["ymin"]: x["ymin"] + x["textbox_height"]] <= x["xmax"]) >= x["textbox_height"] / 2
            #     ),
            #     axis=1
            # )

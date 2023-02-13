from pathlib import Path
import numpy as np
import cv2
import argparse
from numba import jit
from scipy import ndimage as ndi

SEAM_COLOR = np.array([255, 200, 200])    # seam visualization color (BGR)
PMASK_VALUE = 120000.0              # large energy value for protective masking
MASK_THRESHOLD = 10                       # minimum pixel intensity for binary mask


def visualize(img, boolmask=None, rotate=False):
    vis = img.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == False)] = SEAM_COLOR
    if rotate:
        vis = rotate_image(vis, False)
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis


def resize(img, width):
    dim = None
    h, w = img.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(img, dim)


def rotate_image(img, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(img, k)    

########################################
# ENERGY FUNCTIONS
########################################

def backward_energy(img):
    """
    Simple gradient magnitude energy map.
    """
    xgrad = ndi.convolve1d(img, np.array([1, 0, -1]), axis=1, mode="wrap")
    ygrad = ndi.convolve1d(img, np.array([1, 0, -1]), axis=0, mode="wrap")
    
    grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))

    # vis = visualize(grad_mag)
    # cv2.imwrite("backward_energy_demo.jpg", vis)

    return grad_mag

# @jit
# def forward_energy(img):
#     """
#     Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
#     by Rubinstein, Shamir, Avidan.

#     Vectorized code adapted from
#     https://github.com/axu2/improved-seam-carving.
#     """
#     h, w = img.shape[:2]
#     img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

#     energy = np.zeros((h, w))
#     m = np.zeros((h, w))
    
#     U = np.roll(img, 1, axis=0)
#     L = np.roll(img, 1, axis=1)
#     R = np.roll(img, -1, axis=1)
    
#     cU = np.abs(R - L)
#     cL = np.abs(U - L) + cU
#     cR = np.abs(U - R) + cU
    
#     for i in range(1, h):
#         mU = m[i - 1]
#         mL = np.roll(mU, 1)
#         mR = np.roll(mU, -1)
        
#         mULR = np.array([mU, mL, mR])
#         cULR = np.array([cU[i], cL[i], cR[i]])
#         mULR += cULR

#         argmins = np.argmin(mULR, axis=0)
#         m[i] = np.choose(argmins, mULR)
#         energy[i] = np.choose(argmins, cULR)
    
#     # vis = visualize(energy)
#     # cv2.imwrite("forward_energy_demo.jpg", vis)
#     return energy

########################################
# SEAM HELPER FUNCTIONS
######################################## 

@jit
def add_seam(img, seam_idx):
    """
    Add a vertical seam to a 3-channel color image at the indices provided 
    by averaging the pixels values to the left and right of the seam.

    Code adapted from https://github.com/vivianhylee/seam-carving.
    """
    h, w = img.shape[:2]
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(img[row, col: col + 2, ch])
                output[row, col, ch] = img[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = img[row, col:, ch]
            else:
                p = np.average(img[row, col - 1: col + 1, ch])
                output[row, : col, ch] = img[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = img[row, col:, ch]

    return output

@jit
def add_seam_grayscale(img, seam_idx):
    """
    Add a vertical seam to a grayscale image at the indices provided 
    by averaging the pixels values to the left and right of the seam.
    """    
    h, w = img.shape[:2]
    output = np.zeros((h, w + 1))
    for row in range(h):
        col = seam_idx[row]
        if col == 0:
            p = np.average(img[row, col: col + 2])
            output[row, col] = img[row, col]
            output[row, col + 1] = p
            output[row, col + 1:] = img[row, col:]
        else:
            p = np.average(img[row, col - 1: col + 1])
            output[row, : col] = img[row, : col]
            output[row, col] = p
            output[row, col + 1:] = img[row, col:]

    return output

@jit
def remove_seam(img, boolmask):
    h, w = img.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return img[boolmask3c].reshape((h, w - 1, 3))

@jit
def remove_seam_grayscale(img, boolmask):
    h, w = img.shape[:2]
    return img[boolmask].reshape((h, w - 1))

@jit
def get_minimum_seam(img, pmask=None, rmask=None):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from 
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    h, w = img.shape[:2]
    # energy_fn = forward_energy if USE_FORWARD_ENERGY else backward_energy
    energy_fn = backward_energy
    M = energy_fn(img)

    if pmask is not None:
        M[np.where(pmask > MASK_THRESHOLD)] = PMASK_VALUE

    # give removal mask priority over protective mask by using larger negative value
    if rmask is not None:
        M[np.where(rmask > MASK_THRESHOLD)] = -PMASK_VALUE * 100

    backtrack = np.zeros_like(M, dtype="int")

    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    # backtrack to find path
    seam_idx = list()
    boolmask = np.ones((h, w), dtype="bool")
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), boolmask


def seams_removal(img, num_remove, mask=None, vis=False, rot=False):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(img=img, pmask=pmask)
        if vis:
            visualize(img, boolmask, rotate=rot)
        img = remove_seam(img=img, boolmask=boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)
    return img, mask


def seams_insertion(img, num_add, pmask=None, vis=False, rot=False):
    seams_record = list()
    copied_img = img.copy()
    copied_pmask = pmask.copy() if pmask is not None else None

    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(img=copied_img, pmask=copied_pmask)
        if vis:
            visualize(img=copied_img, boolmask=boolmask, rotate=rot)
        seams_record.append(seam_idx)

        copied_img = remove_seam(img=copied_img, boolmask=boolmask)
        if copied_pmask is not None:
            copied_pmask = remove_seam_grayscale(img=copied_pmask, boolmask=boolmask)
    seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        img = add_seam(img, seam)
        if vis:
            visualize(img, rotate=rot)
        if pmask is not None:
            pmask = add_seam_grayscale(pmask, seam)

        # update the remaining seam indices
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2
    return img, pmask


def seam_carve(img, dy, dx, pmask=None, vis=False):
    img = img.astype(np.float64)
    h, w = img.shape[: 2]
    assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

    if pmask is not None:
        pmask = pmask.astype(np.float64)

    output = img

    if dx < 0:
        output, pmask = seams_removal(img=output, num_remove=-dx, mask=pmask, vis=vis)
    elif dx > 0:
        output, pmask = seams_insertion(img=output, num_add=dx, pmask=pmask, vis=vis, rot=False)

    if dy < 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_removal(output, -dy, mask, vis, rot=True)
        output = rotate_image(output, False)
    elif dy > 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_insertion(output, dy, mask, vis, rot=True)
        output = rotate_image(output, False)
    return output


if __name__ == "__main__":
    dir = Path("/Users/jongbeomkim/Documents/image_inpainting/req2/lbtsm/")
    for path in tqdm(list(dir.glob("*"))):
        pmask = load_image(path)
        pmask = _convert_to_2d(pmask)
        pmask = (pmask >= 250).astype("uint8") * 255

        img = load_image(str(path).replace("lbtsm", "texts_removed"))

        # DOWNSIZE_WIDTH = 500                      # resized image width if SHOULD_DOWNSIZE is True
        # SHOULD_DOWNSIZE = False
        # # downsize image for faster processing
        # h, w = img.shape[: 2]
        # if SHOULD_DOWNSIZE and w > DOWNSIZE_WIDTH:
        #     img = resize(img, width=DOWNSIZE_WIDTH)
        #     if pmask is not None:
        #         pmask = resize(pmask, width=DOWNSIZE_WIDTH)
        #     if rmask is not None:
        #         rmask = resize(rmask, width=DOWNSIZE_WIDTH)

        width, height = _get_width_and_height(img)
        dy = 0
        dx = int(width * 0.1)
        output = seam_carve(img=img, dy=dy, dx=dx, pmask=pmask, vis=False)
        output = output.astype("uint8")
        save_image(
            img1=output, path=f"/Users/jongbeomkim/Documents/image_inpainting/req2/seam_carving/{path.name}"
        )

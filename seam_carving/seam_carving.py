from pathlib import Path
import numpy as np
import cv2
from numba import jit
from scipy import ndimage as ndi


def _get_backward_energy_map(img):
    xgrad = ndi.convolve1d(img, weights=np.array([1, 0, -1]), axis=1, mode="wrap")
    ygrad = ndi.convolve1d(img, weights=np.array([1, 0, -1]), axis=0, mode="wrap")
    grad_mag = np.sqrt(
        np.sum(xgrad ** 2, axis=2) + np.sum(ygrad ** 2, axis=2)
    )
    grad_mag /= grad_mag.max()
    grad_mag *= 255
    grad_mag = grad_mag.astype("uint8")
    # show_image(grad_mag)
    # np.max(grad_mag)
    # np.min(grad_mag)
    # grad_mag.shape

    # vis = visualize(grad_mag)
    # show_image(vis)
    # cv2.imwrite("_get_backward_energy_map_demo.jpg", vis)
    return grad_mag


def _get_removal_mask(energy_map):
    rmask = (energy_map <= 10).astype("uint8") * 255
    return rmask


def _rotate_image(img, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(img, k)


def resize(img, width):
    dim = None
    h, w = img.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(img, dim)


# SEAM_COLOR = np.array([255, 0, 0])    # seam visualization color (BGR)
# weight = 120000.0              # large energy value for protective masking
# mask_thresh = 10                       # minimum pixel intensity for binary mask


def visualize(img, mask=None, rotate=False):
    vis = img.copy().astype("uint8")
    if mask is not None:
        # vis[np.where(mask == False)] = SEAM_COLOR
        vis[mask == False] = np.array([255, 0, 0])
    if rotate:
        vis = _rotate_image(img=vis, clockwise=False)
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis


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
def _carve_single_seam(img, mask):
    width, height = _get_width_and_height(img)
    mask_3d = _convert_to_3d(mask)
    # Delete all the pixels marked False in the mask and resize it to the new image dimensions
    return img[mask_3d].reshape((height, width - 1, 3))


@jit
def _carve_single_seam_grayscale(img, mask):
    h, w = img.shape[:2]
    return img[mask].reshape((h, w - 1))


@jit
def _get_seam_with_lowest_energy(img, pmask=None, rmask=None, mask_thresh=10, weight=10_000_000):
    width, height = _get_width_and_height(img)
    # Minimum energy value seen upto that pixel.

    M = _get_backward_energy_map(img)
    M = cast_from_uin8_to_float(M)
    # Give removal mask priority over protective mask by using larger negative value
    if rmask is not None:
        M[rmask > mask_thresh] = -weight
    if pmask is not None:
        M[pmask > mask_thresh] = weight * 2

    backtrack = np.zeros_like(M, dtype="int")

    # Populate DP matrix
    for i in range(1, height):
        for j in range(0, width):
            if j == 0:
                idx = np.argmin(M[i - 1, j: j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1: j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]
            M[i, j] += min_energy

    # Backtrack to find path
    seam_idx = list()
    mask = np.ones((height, width), dtype="bool")
    j = np.argmin(M[-1])
    for i in range(height - 1, -1, -1):
        mask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), mask


def _carve_seams(img, dx, mask=None, vis=False, rotate=False):
    for _ in range(dx):
        seam_idx, mask = _get_seam_with_lowest_energy(img=img, pmask=pmask)
        if vis:
            visualize(img, mask, rotate=rotate)
        img = _carve_single_seam(img=img, mask=mask)
        if mask is not None:
            mask = _carve_single_seam_grayscale(mask, mask)
    return img, mask


def _insert_seams(img, dx, pmask=None, rmask=None, vis=False, rotate=False):
    copied_img = img.copy()
    copied_pmask = pmask.copy() if pmask is not None else None
    copied_rmask = rmask.copy() if rmask is not None else None

    seams_record = list()
    for _ in range(dx):
        seam_idx, mask = _get_seam_with_lowest_energy(img=copied_img, pmask=copied_pmask)
        if vis:
            visualize(img=copied_img, mask=mask, rotate=rotate)
        seams_record.append(seam_idx)

        copied_img = _carve_single_seam(img=copied_img, mask=mask)
        if copied_pmask is not None:
            copied_pmask = _carve_single_seam_grayscale(img=copied_pmask, mask=mask)
    seams_record.reverse()

    for _ in range(dx):
        seam = seams_record.pop()
        img = add_seam(img, seam)
        if vis:
            visualize(img, rotate=rotate)
        if pmask is not None:
            pmask = add_seam_grayscale(pmask, seam)

        # Update the remaining seam indices
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2
    return img, pmask


def cast_from_uin8_to_float(mask):
    new_mask = mask.copy()
    new_mask = new_mask.astype("float")
    new_mask /= new_mask.max()
    return new_mask


def perform_seam_carving(img, dx, dy, pmask=None, rmask=None, vis=False):
    width, height = _get_width_and_height(img)

    pmask = cast_from_uin8_to_float(pmask)
    rmask = cast_from_uin8_to_float(rmask)

    # img = img.astype(np.float64)
    # assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

    # if pmask is not None:
    #     pmask = pmask.astype(np.float64)

    # output = img

    if dx < 0:
        output, pmask = _carve_seams(img=img, dx=-dx, mask=pmask, vis=vis)
    elif dx > 0:
        output, pmask = _insert_seams(img=img, dx=dx, pmask=pmask, vis=vis, rotate=False)

    # if dy < 0:
    #     output = _rotate_image(img=output, clockwise=True)
    #     if mask is not None:
    #         mask = _rotate_image(img=mask, clockwise=True)
    #     output, mask = _carve_seams(output, -dy, mask, vis, rotate=True)
    #     output = _rotate_image(img=output, clockwise=False)
    # elif dy > 0:
    #     output = _rotate_image(img=output, clockwise=True)
    #     if mask is not None:
    #         mask = _rotate_image(img=mask, clockwise=True)
    #     output, mask = _insert_seams(output, dy, mask, vis, rotate=True)
    #     output = _rotate_image(img=output, clockwise=False)
    return output



def get_rmask(img):
    # canvas = _get_canvas_same_size_as_image(_convert_to_2d(img), black=True)
    width, height = _get_width_and_height(img)
    # with wandImage(
    #     filename="/Users/jongbeomkim/Desktop/workspace/text_renderer/bounding_box_tuning/1119_3752_original.jpg"
    # ) as image:
    with wandImage(width=width, height=height, background=Color("black")) as canvas:
        with Drawing() as ctx:
            for xmin, ymin, xmax, ymax, ori_content, tr_content in bboxes.values:
                text = tr_content

                textbox_width, textbox_height = _get_textbox_width_and_height(ctx=ctx, text=text)
                canvas[ymin: ymin + textbox_height, xmin: xmin + textbox_width] = 255
                # canvas[ymin: ymax, xmin: xmax] = 0
        canvas = _convert_wand_image_to_array(canvas)
    show_image(canvas, img)


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
        dx = int(width * 0.2)
        output = perform_seam_carving(img=img, dy=dy, dx=dx, pmask=pmask, vis=False)
        output = output.astype("uint8")
        save_image(
            img1=output, path=f"/Users/jongbeomkim/Documents/image_inpainting/req2/seam_carving/{path.name}"
        )


    region_mask = _convert_region_score_map_to_region_mask(
        region_score_map=region_score_map, region_score_thresh=200
    )
    pmask = region_mask[:, :, 0]
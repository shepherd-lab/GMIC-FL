# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import os
import argparse
import logging
from multiprocessing import Pool
from functools import partial

import scipy.ndimage
import numpy as np
import cv2

import utilities.pickling as pickling
import utilities.reading_images as reading_images
import utilities.saving_images as saving_images
import utilities.data_handling as data_handling

logger = logging.getLogger(__name__)

_DIR_CACHE = set()


def _ensure_dir(path):
    if path not in _DIR_CACHE:
        os.makedirs(path, exist_ok=True)
        _DIR_CACHE.add(path)


TARGET_H, TARGET_W = 2944, 1920


def resize_and_pad_keep_aspect(img, mode):
    """
    img: 2D numpy array (cropped mammogram, HxW)
    mode: 'left' or 'right' (direction the breast points in the original image)
    Returns HxW == (TARGET_H, TARGET_W) without warping:
      - scale isotropically to fit within 2944x1920
      - place chest wall flush against the chest-wall side (no padding there)
      - pad on the nipple side and top/bottom as needed with zeros
    """
    H, W = img.shape
    # scale to fit within target while preserving aspect ratio
    s = min(TARGET_H / H, TARGET_W / W)
    newH, newW = int(round(H * s)), int(round(W * s))
    resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)

    # create target canvas
    out = np.zeros((TARGET_H, TARGET_W), dtype=img.dtype)

    # vertical placement: center (top/bottom padding split)
    top = (TARGET_H - newH) // 2

    # horizontal placement:
    # keep chest wall flush to its side:
    # - If breast points LEFT, chest wall is on the RIGHT edge of the crop => place flush to RIGHT.
    # - If breast points RIGHT, chest wall is on the LEFT edge => place flush to LEFT.
    if mode == "left":
        # chest wall on right edge => align resized to right
        left = TARGET_W - newW
    else:
        # chest wall on left edge => align resized to left
        left = 0

    out[top:top + newH, left:left + newW] = resized
    return out


def crop_mammogram_one_image_short_path_tolerant(
    scan,
    input_data_folder,
    output_data_folder,
    num_iterations,
    buffer_size,
    logger_fn=None,
):
    """Wrapper that catches failures and returns None for failed images.

    Accepts same arguments as crop_mammogram_one_image_short_path plus logger_fn.
    """
    try:
        return crop_mammogram_one_image_short_path(
            scan,
            input_data_folder,
            output_data_folder,
            num_iterations,
            buffer_size,
            logger_fn=logger_fn,
        )
    except Exception as e:
        _log(logger_fn, f"CROP FAILED: {scan.get('short_file_path', 'unknown')} - {e}", level=logging.ERROR)
        return None


def _log(logger_fn, msg: str, level: int = logging.INFO):
    """
    Safe logger helper: never print; avoids crashing if stdout is broken.
    - If logger_fn is provided (callable), uses it.
    - Else uses module logger.
    """
    if logger_fn is not None:
        try:
            logger_fn(msg)
            return
        except Exception:
            pass
    try:
        logger.log(level, msg)
    except Exception:
        pass


def crop_mammogram(
    input_data_folder,
    exam_list_path,
    cropped_exam_list_path,
    output_data_folder,
    num_processes,
    num_iterations,
    buffer_size,
    error_log_path=None,
    logger_fn=None,
    tolerant=True,
):
    """
    Parallel cropping with fault tolerance:
      - bad images are logged (error_log_path)
      - successful images are kept; failed ones are removed from exam_list
      - metadata is attached only for successful images
    """
    exam_list = pickling.unpickle_from_file(exam_list_path)
    image_list = data_handling.unpack_exam_into_images(exam_list)

    _ensure_dir(output_data_folder)

    if tolerant:
        crop_mammogram_one_image_func = partial(
            crop_mammogram_one_image_short_path_tolerant,
            input_data_folder=input_data_folder,
            output_data_folder=output_data_folder,
            num_iterations=num_iterations,
            buffer_size=buffer_size,
            logger_fn=logger_fn,
        )
    else:
        crop_mammogram_one_image_func = partial(
            crop_mammogram_one_image_short_path,
            input_data_folder=input_data_folder,
            output_data_folder=output_data_folder,
            num_iterations=num_iterations,
            buffer_size=buffer_size,
            logger_fn=logger_fn,
        )

    successes, failures = [], []

    # Use imap_unordered to avoid one failure stalling the whole pool
    if num_processes == 1:
        # Serial path: no pickling, no Pool overhead
        for scan in image_list:
            tag_result = crop_mammogram_one_image_func(scan)
            if not tag_result:
                continue
            tag = tag_result[0]
            if tag == "ok":
                successes.append(tag_result[1])
            elif tag == "fail":
                _, short_id, err = tag_result
                failures.append((short_id, err))
    else:
        with Pool(num_processes) as pool:
            for tag_result in pool.imap_unordered(crop_mammogram_one_image_func, image_list, chunksize=16):
                if not tag_result:
                    continue
                tag = tag_result[0]
                if tag == "ok":
                    successes.append(tag_result[1])
                elif tag == "fail":
                    _, short_id, err = tag_result
                    failures.append((short_id, err))

    total = len(image_list)
    ok = len(successes)
    _log(logger_fn, f"success={ok}/{total} images; failures={total - ok}; out_dir={output_data_folder}")

    # Write a failure log in the parent process (atomic-ish)
    if error_log_path and failures:
        try:
            parent = os.path.dirname(error_log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(error_log_path, "w") as f:
                for sid, err in failures:
                    f.write(f"{sid}\t{err}\n")
            _log(logger_fn, f"wrote failure log: {error_log_path} ({len(failures)} items)")
        except Exception as e:
            _log(logger_fn, f"could not write failure log: {e}", level=logging.WARNING)

    if ok == 0:
        # No images succeeded; still persist the (unmodified) exam_list so the caller can decide to fallback.
        pickling.pickle_to_file(cropped_exam_list_path, exam_list)
        return

    # Build metadata dicts
    wl_pairs, rp_pairs, bp_pairs, ds_pairs = [], [], [], []
    for rec in successes:
        try:
            wl_pairs.append(rec[0])
            rp_pairs.append(rec[1])
            bp_pairs.append(rec[2])
            ds_pairs.append(rec[3])
        except Exception:
            continue
    window_location_dict = dict(wl_pairs)
    rightmost_points_dict = dict(rp_pairs)
    bottommost_points_dict = dict(bp_pairs)
    distance_from_starting_side_dict = dict(ds_pairs)

    # Keep only images that have all required crop metadata
    keep = set(window_location_dict) & set(rightmost_points_dict) & set(bottommost_points_dict)
    if keep:
        for exam in exam_list:
            for view in ["L-CC", "L-MLO", "R-CC", "R-MLO"]:
                if view in exam and exam[view]:
                    keep_idx, new_imgs = [], []
                    for idx, image_id in enumerate(exam[view]):
                        if image_id in keep:
                            new_imgs.append(image_id)
                            keep_idx.append(idx)
                    exam[view] = new_imgs
                    # align any parallel arrays (e.g., original_file_paths)
                    if "original_file_paths" in exam and view in exam["original_file_paths"]:
                        exam["original_file_paths"][view] = [exam["original_file_paths"][view][i] for i in keep_idx]

    # Attach metadata to the (pruned) exam_list
    data_handling.add_metadata(exam_list, "window_location", window_location_dict)
    data_handling.add_metadata(exam_list, "rightmost_points", rightmost_points_dict)
    data_handling.add_metadata(exam_list, "bottommost_points", bottommost_points_dict)
    data_handling.add_metadata(exam_list, "distance_from_starting_side", distance_from_starting_side_dict)

    # Persist
    pickling.pickle_to_file(cropped_exam_list_path, exam_list)


def get_masks_and_sizes_of_connected_components(img_mask):
    """
    Finds connected components from the mask of the image.
    Returns (labeled_mask, {label: pixel_count}) with label 0 = background excluded.
    """
    mask, num_labels = scipy.ndimage.label(img_mask)

    mask_pixels_dict = {}
    # label 0 is background
    for i in range(1, num_labels + 1):
        this_mask = (mask == i)
        pixels = int(np.sum(this_mask))
        if pixels > 0:
            mask_pixels_dict[i] = pixels

    return mask, mask_pixels_dict


def get_mask_of_largest_connected_component(img_mask):
    """
    Finds the largest connected component from the mask of the image.
    Raises ValueError if no foreground components exist.
    """
    mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(img_mask)
    if not mask_pixels_dict:
        raise ValueError("no connected components found in mask")
    largest_mask_index = max(mask_pixels_dict, key=mask_pixels_dict.get)
    largest_mask = (mask == largest_mask_index)
    return largest_mask


def get_edge_values(img, largest_mask, axis):
    """
    Finds the bounding box (start/end) for the largest component along given axis.
    Raises ValueError if the component is empty along that axis.
    """
    assert axis in ["x", "y"]
    has_value = np.any(largest_mask, axis=int(axis == "y"))
    idxs = np.arange(img.shape[int(axis == "x")])[has_value]
    if idxs.size == 0:
        raise ValueError(f"largest component empty along axis={axis}")
    edge_start = idxs[0]
    edge_end = idxs[-1] + 1
    return edge_start, edge_end


def get_bottommost_pixels(img, largest_mask, y_edge_bottom):
    """
    Gets the bottommost nonzero pixels of dilated mask before cropping.
    """
    bottommost_nonzero_y = y_edge_bottom - 1
    bottommost_nonzero_x = np.arange(img.shape[1])[largest_mask[bottommost_nonzero_y, :] > 0]
    return bottommost_nonzero_y, bottommost_nonzero_x


def get_distance_from_starting_side(img, mode, x_edge_left, x_edge_right):
    """
    If we fail to recover the original shape as a result of erosion-dilation
    on the side where the breast starts to appear in the image,
    we record this information.
    """
    if mode == "left":
        return img.shape[1] - x_edge_right
    else:
        return x_edge_left


def include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size):
    """
    Includes buffer in all sides of the image in y-direction
    """
    if y_edge_top > 0:
        y_edge_top -= min(y_edge_top, buffer_size)
    if y_edge_bottom < img.shape[0]:
        y_edge_bottom += min(img.shape[0] - y_edge_bottom, buffer_size)
    return y_edge_top, y_edge_bottom


def include_buffer_x_axis(img, mode, x_edge_left, x_edge_right, buffer_size):
    """
    Includes buffer in only one side of the image in x-direction
    """
    if mode == "left":
        if x_edge_left > 0:
            x_edge_left -= min(x_edge_left, buffer_size)
    else:
        if x_edge_right < img.shape[1]:
            x_edge_right += min(img.shape[1] - x_edge_right, buffer_size)
    return x_edge_left, x_edge_right


def convert_bottommost_pixels_wrt_cropped_image(
    mode,
    bottommost_nonzero_y,
    bottommost_nonzero_x,
    y_edge_top,
    x_edge_right,
    x_edge_left,
):
    """
    Once the image is cropped, adjusts the bottommost pixel values which was originally w.r.t. the original image
    """
    bottommost_nonzero_y -= y_edge_top
    if mode == "left":
        bottommost_nonzero_x = x_edge_right - bottommost_nonzero_x  # not in sorted order anymore
        bottommost_nonzero_x = np.flip(bottommost_nonzero_x, 0)
    else:
        bottommost_nonzero_x -= x_edge_left
    return bottommost_nonzero_y, bottommost_nonzero_x


def get_rightmost_pixels_wrt_cropped_image(mode, largest_mask_cropped, find_rightmost_from_ratio):
    """
    Search the bottom portion of the cropped mask for extreme (left/right) nonzero pixels.
    Raises ValueError if no nonzero pixels exist in the search area.
    """
    ignore_height = int(largest_mask_cropped.shape[0] * find_rightmost_from_ratio)
    rightmost_pixel_search_area = largest_mask_cropped[ignore_height:, :]
    rightmost_pixel_search_area_has_value = np.any(rightmost_pixel_search_area, axis=0)
    cols = np.arange(rightmost_pixel_search_area.shape[1])[rightmost_pixel_search_area_has_value]
    if cols.size == 0:
        raise ValueError("no nonzero pixels in search area")
    rightmost_nonzero_x = cols[-1 if mode == "right" else 0]
    rightmost_nonzero_y = (
        np.arange(rightmost_pixel_search_area.shape[0])[
            rightmost_pixel_search_area[:, rightmost_nonzero_x] > 0
        ]
        + ignore_height
    )

    if mode == "left":
        rightmost_nonzero_x = largest_mask_cropped.shape[1] - rightmost_nonzero_x

    return rightmost_nonzero_y, rightmost_nonzero_x


def crop_img_from_largest_connected(
    img,
    mode,
    erode_dialate=True,
    iterations=100,
    buffer_size=50,
    find_rightmost_from_ratio=1 / 3,
):
    """
    Performs erosion on the mask of the image, selects largest connected component,
    dilates the largest connected component, and draws a bounding box for the result
    with buffers.

    output: a tuple of (window_location, rightmost_points,
                        bottommost_points, distance_from_starting_side)
    """
    assert mode in ("left", "right")

    img_mask = img > 0
    if not np.any(img_mask):
        raise ValueError("no foreground pixels in mask")

    # Erosion in order to remove thin lines in the background
    if erode_dialate:
        img_mask = scipy.ndimage.binary_erosion(img_mask, iterations=iterations)

    # Select mask for largest connected component
    largest_mask = get_mask_of_largest_connected_component(img_mask)

    # Dilation to recover the original mask, excluding the thin lines
    if erode_dialate:
        largest_mask = scipy.ndimage.binary_dilation(largest_mask, iterations=iterations)

    # figure out where to crop
    y_edge_top, y_edge_bottom = get_edge_values(img, largest_mask, "y")
    x_edge_left, x_edge_right = get_edge_values(img, largest_mask, "x")

    # extract bottommost pixel info
    bottommost_nonzero_y, bottommost_nonzero_x = get_bottommost_pixels(img, largest_mask, y_edge_bottom)

    # include maximum 'buffer_size' more pixels on both sides just to make sure we don't miss anything
    y_edge_top, y_edge_bottom = include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size)

    # If cropped image not starting from corresponding edge, they are wrong. Record the distance, will reject if not 0.
    distance_from_starting_side = get_distance_from_starting_side(img, mode, x_edge_left, x_edge_right)

    # include more pixels on either side just to make sure we don't miss anything
    x_edge_left, x_edge_right = include_buffer_x_axis(img, mode, x_edge_left, x_edge_right, buffer_size)

    # convert bottommost pixel locations w.r.t. newly cropped image. Flip if necessary.
    bottommost_nonzero_y, bottommost_nonzero_x = convert_bottommost_pixels_wrt_cropped_image(
        mode,
        bottommost_nonzero_y,
        bottommost_nonzero_x,
        y_edge_top,
        x_edge_right,
        x_edge_left,
    )

    # calculate rightmost point from bottom portion of the image w.r.t. cropped image. Flip if necessary.
    rightmost_nonzero_y, rightmost_nonzero_x = get_rightmost_pixels_wrt_cropped_image(
        mode,
        largest_mask[y_edge_top:y_edge_bottom, x_edge_left:x_edge_right],
        find_rightmost_from_ratio,
    )

    # save window location in medical mode, but everything else in training mode
    return (
        (y_edge_top, y_edge_bottom, x_edge_left, x_edge_right),
        ((rightmost_nonzero_y[0], rightmost_nonzero_y[-1]), rightmost_nonzero_x),
        (bottommost_nonzero_y, (bottommost_nonzero_x[0], bottommost_nonzero_x[-1])),
        distance_from_starting_side,
    )


def image_orientation(horizontal_flip, side):
    """
    Returns the direction where the breast should be facing in the original image.
    """
    assert horizontal_flip in ["YES", "NO"], "Wrong horizontal flip"
    assert side in ["L", "R"], "Wrong side"
    if horizontal_flip == "YES":
        if side == "R":
            return "right"
        else:
            return "left"
    else:
        if side == "R":
            return "left"
        else:
            return "right"


def crop_mammogram_one_image(scan, input_file_path, output_file_path, num_iterations, buffer_size, logger_fn=None):
    """
    Crops a mammogram and saves as PNG.
    On success, returns cropping_info (tuple of 4 items).
    On failure, returns {"failed": True, "error": "..."}.
    """
    image = reading_images.read_image_png(input_file_path)
    try:
        if image is None or getattr(image, "size", 0) == 0 or min(image.shape[:2]) == 0:
            raise ValueError("empty/invalid image array")

        cropping_info = crop_img_from_largest_connected(
            image,
            image_orientation(scan["horizontal_flip"], scan["side"]),
            True,
            num_iterations,
            buffer_size,
            1 / 3,
        )
    except Exception as error:
        _log(
            logger_fn,
            f"FAIL short_id={scan.get('short_file_path')} path={input_file_path} err={error}",
            level=logging.ERROR,
        )
        return {"failed": True, "error": str(error)}

    # existing crop window
    top, bottom, left, right = cropping_info[0]

    # enforce paper size
    mode = image_orientation(scan["horizontal_flip"], scan["side"])
    cropped = image[top:bottom, left:right]

    # [PAD] diagnostic: fraction of the 2944x1920 canvas that is zero-padding introduced by
    # resize_and_pad (isotropic fit + pad, never clips). Logged per image so the PER-SITE
    # padding distribution can be aggregated offline (grep '\[PAD\]'). This characterizes the
    # one real preprocessing artifact of the resize approach -- breast scale varies with the
    # source aspect ratio, and that variation is site-correlated. Decides resize(C') vs
    # upstream variable-crop(B') for the spatial-scale analysis. Logging-only.
    ch, cw = cropped.shape[:2]
    if ch > 0 and cw > 0:
        s = min(TARGET_H / ch, TARGET_W / cw)
        new_h, new_w = int(round(ch * s)), int(round(cw * s))
        pad_frac = 1.0 - (new_h * new_w) / float(TARGET_H * TARGET_W)
        _log(logger_fn,
             f"[PAD] sid={scan.get('short_file_path')} side={scan.get('side')} "
             f"crop={ch}x{cw} scaled={new_h}x{new_w} pad_frac={pad_frac:.4f}")

    final_2944x1920 = resize_and_pad_keep_aspect(cropped, mode)

    target_parent_dir = os.path.split(output_file_path)[0]
    _ensure_dir(target_parent_dir)

    # save standardized image
    saving_images.save_image_as_png(final_2944x1920, output_file_path)

    # return cropping info for aggregation upstream
    return cropping_info


def crop_mammogram_one_image_short_path(
    scan,
    input_data_folder,
    output_data_folder,
    num_iterations,
    buffer_size,
    logger_fn=None,
):
    """
    Wrapper around crop_mammogram_one_image that uses short_file_path.
    Returns:
      ("ok", [(short_id, window_loc), (short_id, rightmost), (short_id, bottommost), (short_id, distance)])
      or ("fail", short_id, error_string)
    """
    short_id = scan["short_file_path"]
    full_input_file_path = os.path.join(input_data_folder, short_id + ".png")
    full_output_file_path = os.path.join(output_data_folder, short_id + ".png")

    if logger_fn:
        try:
            logger_fn(f"START short_id={short_id} in={full_input_file_path}")
        except Exception:
            pass

    result = crop_mammogram_one_image(
        scan=scan,
        input_file_path=full_input_file_path,
        output_file_path=full_output_file_path,
        num_iterations=num_iterations,
        buffer_size=buffer_size,
        logger_fn=logger_fn,
    )

    if isinstance(result, dict) and result.get("failed"):
        return ("fail", short_id, result.get("error", "unknown_error"))

    # success
    cropping_info = result  # tuple of 4 items
    if logger_fn and isinstance(cropping_info, tuple) and len(cropping_info) == 4:
        window_loc, rightmost, bottommost, dist = cropping_info
        try:
            logger_fn(
                f"DONE short_id={short_id} saved={full_output_file_path} "
                f"window={window_loc} rightmost={rightmost} bottommost={bottommost} dist={dist}"
            )
        except Exception:
            pass

    return ("ok", list(zip([short_id] * 4, cropping_info)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background of image and save cropped files")
    parser.add_argument("--input-data-folder", required=True)
    parser.add_argument("--output-data-folder", required=True)
    parser.add_argument("--exam-list-path", required=True)
    parser.add_argument("--cropped-exam-list-path", required=True)
    parser.add_argument("--num-processes", default=10, type=int)
    parser.add_argument("--num-iterations", default=100, type=int)
    parser.add_argument("--buffer-size", default=50, type=int)
    parser.add_argument("--error-log-path", default=None)
    args = parser.parse_args()

    crop_mammogram(
        input_data_folder=args.input_data_folder,
        exam_list_path=args.exam_list_path,
        cropped_exam_list_path=args.cropped_exam_list_path,
        output_data_folder=args.output_data_folder,
        num_processes=args.num_processes,
        num_iterations=args.num_iterations,
        buffer_size=args.buffer_size,
        error_log_path=args.error_log_path,
        tolerant=True,
    )
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

"""
Runs search_windows_and_centers.py and extract_centers.py in the same directory
"""
import argparse
import logging
import traceback
import numpy as np
import os
from collections import Counter
from itertools import repeat
from multiprocessing import Pool

from constants.constants import INPUT_SIZE_DICT

logger = logging.getLogger(__name__)
import utilities.pickling as pickling
import utilities.data_handling as data_handling
import utilities.reading_images as reading_images
import data_loader.loading as loading
import data_loader.calc_optimal_centers as calc_optimal_centers


def extract_center(datum, image):
    """
    Compute the optimal center for an image
    """
    image = loading.flip_image(image, datum["full_view"], datum['horizontal_flip'])
    if datum["view"] == "MLO":
        tl_br_constraint = calc_optimal_centers.get_bottomrightmost_pixel_constraint(
            rightmost_x=datum["rightmost_points"][1],
            bottommost_y=datum["bottommost_points"][0],
        )
    elif datum["view"] == "CC":
        tl_br_constraint = calc_optimal_centers.get_rightmost_pixel_constraint(
            rightmost_x=datum["rightmost_points"][1]
        )
    else:
        raise RuntimeError(datum["view"])
    optimal_center = calc_optimal_centers.get_image_optimal_window_info(
        image,
        com=np.array(image.shape) // 2,
        window_dim=np.array(INPUT_SIZE_DICT[datum["full_view"]]),
        tl_br_constraint=tl_br_constraint,
    )
    return optimal_center["best_center_y"], optimal_center["best_center_x"]


def load_and_extract_center(datum, data_prefix):
    """
    Load image and compute optimal center.

    Returns (short_file_path, (center_y, center_x)) on success, or
    (short_file_path, {"__error__": ...diagnostics...}) on failure -- so one bad image
    cannot kill the whole pool and we can surface WHY it failed (the masked AssertionError
    plus the values that feed the center-window bounds check). Logging-only; no math change.
    """
    sid = datum.get("short_file_path", "?")
    full_image_path = os.path.join(data_prefix, sid + '.png')
    try:
        image = reading_images.read_image_png(full_image_path)
        return sid, extract_center(datum, image)
    except Exception as e:
        try:
            img_shape = tuple(np.asarray(image).shape)  # type: ignore[name-defined]
        except Exception:
            img_shape = None
        return sid, {
            "__error__": f"{type(e).__name__}: {e}",
            "view": datum.get("full_view"),
            "image_shape": img_shape,
            "expected_window_dim": tuple(INPUT_SIZE_DICT.get(datum.get("full_view"), ("?", "?"))),
            "rightmost_points": datum.get("rightmost_points"),
            "bottommost_points": datum.get("bottommost_points"),
            "horizontal_flip": datum.get("horizontal_flip"),
            "traceback": traceback.format_exc(),
        }


def _site_of(short_id):
    """Best-effort site tag from the short_file_path prefix (e.g. 'RSNA_..' -> 'RSNA')."""
    s = str(short_id)
    return s.split("_", 1)[0] if "_" in s else "?"


def get_optimal_centers(data_list, data_prefix, num_processes=1):
    """
    Compute optimal centers for each image in data list. Failures are captured per-image
    (never abort the batch); a summary of count + per-site breakdown + a few full examples
    is logged so a masked Stage-2 AssertionError becomes diagnosable.
    """
    pool = Pool(num_processes)
    results = pool.starmap(load_and_extract_center, zip(data_list, repeat(data_prefix)))
    pool.close(); pool.join()

    centers, failures = {}, []
    for sid, val in results:
        if isinstance(val, dict) and "__error__" in val:
            failures.append((sid, val))
        else:
            centers[sid] = val

    if failures:
        by_site = Counter(_site_of(sid) for sid, _ in failures)
        by_err = Counter(info["__error__"].split(" (")[0] for _, info in failures)
        logger.warning(
            "[CENTERS] %d/%d images FAILED center extraction. by_site=%s by_error=%s",
            len(failures), len(results), dict(by_site), dict(by_err),
        )
        for sid, info in failures[:5]:  # a few full examples (id + inputs + traceback)
            logger.warning(
                "[CENTERS][FAIL] id=%s site=%s view=%s image_shape=%s expected_window=%s "
                "rightmost=%s bottommost=%s flip=%s err=%s",
                sid, _site_of(sid), info["view"], info["image_shape"], info["expected_window_dim"],
                info["rightmost_points"], info["bottommost_points"], info["horizontal_flip"],
                info["__error__"],
            )
            logger.warning("[CENTERS][FAIL][trace] id=%s\n%s", sid, info["traceback"])

    return centers


def main(cropped_exam_list_path, data_prefix, output_exam_list_path, num_processes=1):
    exam_list = pickling.unpickle_from_file(cropped_exam_list_path)
    data_list = data_handling.unpack_exam_into_images(exam_list, cropped=True)
    optimal_centers = get_optimal_centers(
        data_list=data_list,
        data_prefix=data_prefix,
        num_processes=num_processes
    )
    data_handling.add_metadata(exam_list, "best_center", optimal_centers)
    os.makedirs(os.path.dirname(output_exam_list_path), exist_ok=True)
    pickling.pickle_to_file(output_exam_list_path, exam_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute and Extract Optimal Centers')
    parser.add_argument('--cropped-exam-list-path')
    parser.add_argument('--data-prefix')
    parser.add_argument('--output-exam-list-path', required=True)
    parser.add_argument('--num-processes', default=20)
    args = parser.parse_args()

    main(
        cropped_exam_list_path=args.cropped_exam_list_path,
        data_prefix=args.data_prefix,
        output_exam_list_path=args.output_exam_list_path,
        num_processes=int(args.num_processes),
    )
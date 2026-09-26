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
Defines utility functions for managing the dataset.

NOTE: Avoid using print() here; NVFLARE container workers may have broken stdout.
Use an injected logger callback where possible.
"""

import logging
from constants.constants import VIEWS

logger = logging.getLogger(__name__)


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


def unpack_exam_into_images(exam_list, cropped: bool = False, skip_missing: bool = True, logger_fn=None):
    """
    Turn exam_list into image_list.

    Args:
        exam_list: list of exam dicts
        cropped: if True, require cropping metadata and include it in image_dict
        skip_missing: if True and cropped, skip images missing crop metadata; else raise KeyError
        logger_fn: optional callable(str) for safe logging
    """
    image_list = []
    skipped_count = 0

    for i, exam in enumerate(exam_list):
        for view in VIEWS.LIST:
            # Be robust if view key absent or None
            images_for_view = exam.get(view, []) or []
            for j, image in enumerate(images_for_view):
                image_dict = dict(
                    short_file_path=image,
                    horizontal_flip=exam.get("horizontal_flip"),
                    full_view=view,
                    side=view[0],
                    view=view[2:],
                )

                if cropped:
                    # Check if this image has the required metadata
                    has_metadata = (
                        "window_location" in exam
                        and view in exam["window_location"]
                        and j < len(exam["window_location"][view])
                        and "rightmost_points" in exam
                        and view in exam["rightmost_points"]
                        and j < len(exam["rightmost_points"][view])
                        and "bottommost_points" in exam
                        and view in exam["bottommost_points"]
                        and j < len(exam["bottommost_points"][view])
                        and "distance_from_starting_side" in exam
                        and view in exam["distance_from_starting_side"]
                        and j < len(exam["distance_from_starting_side"][view])
                    )

                    if has_metadata:
                        image_dict["window_location"] = exam["window_location"][view][j]
                        image_dict["rightmost_points"] = exam["rightmost_points"][view][j]
                        image_dict["bottommost_points"] = exam["bottommost_points"][view][j]
                        image_dict["distance_from_starting_side"] = exam["distance_from_starting_side"][view][j]
                        image_list.append(image_dict)
                    elif skip_missing:
                        skipped_count += 1
                        _log(
                            logger_fn,
                            f"Skipping {image} - missing cropping metadata (exam_index={i}, view={view}, idx={j})",
                            level=logging.WARNING,
                        )
                    else:
                        raise KeyError(f"Missing cropping metadata for {image} (exam_index={i}, view={view}, idx={j})")
                else:
                    image_list.append(image_dict)

    if cropped and skipped_count > 0:
        _log(logger_fn, f"Skipped {skipped_count} images due to missing cropping metadata", level=logging.WARNING)

    return image_list


def add_metadata(exam_list, additional_metadata_name, additional_metadata_dict):
    """
    Includes new information about images into exam_list.
    """
    for exam in exam_list:
        assert additional_metadata_name not in exam, "this metadata is already included"
        exam[additional_metadata_name] = dict()
        for view in VIEWS.LIST:
            exam[additional_metadata_name][view] = []
            images_for_view = exam.get(view, []) or []
            for j, image in enumerate(images_for_view):
                exam[additional_metadata_name][view].append(additional_metadata_dict[image])
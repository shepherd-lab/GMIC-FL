# ============================================================================
# test_center_neutralized.py - Stage-2 center neutralization (C')
# Run from custom/tests/:  python test_center_neutralized.py
#
# The optimal-center search is neutralized to image-center because it is INERT in this
# fork's resize+pad pipeline. This test proves the invariant that justifies that:
#   the train/eval augmentation output is IDENTICAL for any best_center when the saved
#   image already equals the model window (2944x1920). So image-center is provably as
#   good as any value, and Stage 2 just needs to populate the field.
# (The optimal-center machinery itself is kept, unimported, recoverable for B'.)
# ============================================================================
import os
import sys
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)                     # this tests/ dir (sibling test imports)
sys.path.insert(0, os.path.dirname(THIS_DIR))    # custom/ (bc_executor, fl_utils, model.*, ...)

import data_loader.augmentations as aug  # noqa: E402

H, W = 2944, 1920


def _img():
    img = np.zeros((H, W), dtype=np.float32)
    img[800:2100, W - 760:W] = 800.0                       # breast flush right
    img[1500 - 120:1500 + 120, W - 250 - 120:W - 250 + 120] = 4000.0  # dense mass
    return img


def _aug(center, seed, noise):
    mc = (100, 100) if noise else (0, 0)
    ms = 100 if noise else 0
    return aug.random_augmentation_best_center(
        image=_img(), input_size=(H, W),
        random_number_generator=np.random.RandomState(seed),
        max_crop_noise=mc, max_crop_size_noise=ms,
        auxiliary_image=None, best_center=center, view="R-CC",
    )[0]


def test_augmentation_center_inert():
    """At image==window size, augmentation output is identical for ANY best_center."""
    center = (H // 2, W // 2)
    for other in [(1500, W - 250), (300, 300), (H - 10, 10), (0, 0)]:
        for noise in (False, True):
            a = _aug(center, 7, noise)
            b = _aug(other, 7, noise)   # same seed -> only center differs
            assert np.array_equal(a, b), \
                f"center {center} vs {other} differ (noise={noise}) -> center NOT inert"
    print("[neutralized] augmentation output identical for any center (image==window). OK")


def test_stage2_assigns_image_center():
    """Stage 2 assigns (H//2, W//2) to every image and populates best_center for all views."""
    # Minimal fake exam_list mimicking the cropped structure data_handling expects.
    import utilities.data_handling as dh
    # cropped unpack requires all four metadata dicts present per view (see unpack_exam_into_images)
    exam = {"horizontal_flip": "NO",
            "window_location": {}, "rightmost_points": {},
            "bottommost_points": {}, "distance_from_starting_side": {}}
    for v in ["L-CC", "R-CC", "L-MLO", "R-MLO"]:
        exam[v] = [f"P_{v}"]
        exam["window_location"][v] = [(0, H, 0, W)]
        exam["rightmost_points"][v] = [((0, 0), W - 1)]
        exam["bottommost_points"][v] = [(H - 1, (0, 0))]
        exam["distance_from_starting_side"][v] = [0]
    data_list = dh.unpack_exam_into_images([exam], cropped=True)
    assert len(data_list) == 4, f"expected 4 images, got {len(data_list)}"
    # mirror Stage 2 exactly: build centers from the unpacked data_list
    centers = {d["short_file_path"]: (H // 2, W // 2) for d in data_list}
    dh.add_metadata([exam], "best_center", centers)
    for v in ["L-CC", "R-CC", "L-MLO", "R-MLO"]:
        assert exam["best_center"][v] == [(H // 2, W // 2)], f"{v} center not image-center"
    print(f"[neutralized] Stage-2 assigns image-center ({H//2},{W//2}) to all views. OK")


ALL = [test_augmentation_center_inert, test_stage2_assigns_image_center]

if __name__ == "__main__":
    failures = 0
    for t in ALL:
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
            import traceback; traceback.print_exc()
    print("\n==== %d/%d neutralization tests passed ====" % (len(ALL) - failures, len(ALL)))
    sys.exit(1 if failures else 0)

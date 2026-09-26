"""Refactored GMIC Data Loader

Implements:
  - Two-stage preprocessing with auto-resume
  - Progress manifest + stats
  - Tolerant skip of Stage2
  - Status utilities

NOTE:
  - All direct print() calls removed to avoid OSError: [Errno 5] Input/output error
  - Uses Python logging + optional log_file
"""
from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime
from collections import defaultdict
from typing import Tuple
import shutil
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import imageio.v2 as imageio

from utilities import pickling, data_handling
from constants.constants import VIEWS
import data_loader.augmentations as augmentations
from data_loader.crop_mammogram import crop_mammogram
# NOTE: optimal-center search is currently NEUTRALIZED (see _stage2_extract_centers) because it
# is inert in this fork's resize+pad pipeline. Import kept so reviving it (B': upstream
# variable-crop) is a one-line change in Stage 2, not a re-wire. Do not delete the machinery.
from data_loader.get_optimal_centers import get_optimal_centers  # noqa: F401  (recoverable)

TARGET_H, TARGET_W = 2944, 1920


# Module logger (NVFLARE usually captures this into its logs)
logger = logging.getLogger(__name__)


class GMICDataLoader:
    def __init__(
        self,
        data_path: str,
        image_path: str,
        batch_size: int = 4,
        random_seed: int = 42,
        use_predefined_splits: bool = True,
        val_split: float = 0.2,
        test_split: float = 0.1,
        input_format: str = "csv",
        enable_preprocessing: bool = True,
        output_dir: str = "/workspace/processed_data",
        num_processes: int = 4,
        cache_validation: bool = True,
        train_max_crop_noise: Tuple[int, int] = (0, 0),
        train_max_crop_size_noise: int = 0,
        log_file: str | None = None,
        per_file_logging: bool = False,
        tolerant_missing_metadata: bool = False,
        file_integrity_check: bool = False,
        fail_on_integrity_error: bool = False,
        use_balanced_sampler: bool = False,
        use_augmentation: bool = False,
        augmentation: dict | None = None,
    ):
        """GMIC data loader with optional preprocessing and integrity verification.

        Parameters mirror prior version plus:
          file_integrity_check: verify cropped_images vs exam list when reusing cache.
          fail_on_integrity_error: rebuild pipeline if integrity mismatch (else warn & continue).
        """
        # Core config
        self._rng_train = np.random.RandomState(random_seed)
        self._rng_eval = np.random.RandomState(0)
        self.data_path = data_path
        self.image_path = image_path
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.use_predefined_splits = use_predefined_splits
        self.val_split = val_split
        self.test_split = test_split
        self.input_format = input_format
        self.enable_preprocessing = enable_preprocessing
        self.output_dir = output_dir
        self.num_processes = num_processes
        self.cache_validation = cache_validation
        self.train_max_crop_noise = train_max_crop_noise
        self.train_max_crop_size_noise = train_max_crop_size_noise
        self.use_balanced_sampler = use_balanced_sampler
        self.use_augmentation = use_augmentation
        _aug = augmentation or {}
        self.aug_cfg = {
            "horizontal_flip": bool(_aug.get("horizontal_flip", False)),
            "max_rotation_deg": float(_aug.get("max_rotation_deg", 10.0)),
            "intensity_jitter": float(_aug.get("intensity_jitter", 0.10)),
        }
        # Dedicated augmentation RNG, seeded off random_seed for reproducibility (separate from
        # the crop-noise rng so toggling augmentation doesn't shift the crop-noise stream).
        self._rng_aug = np.random.RandomState(random_seed)
        self.log_file = log_file
        self.per_file_logging = per_file_logging
        self.tolerant_missing_metadata = tolerant_missing_metadata
        # Integrity options
        self.file_integrity_check = file_integrity_check
        self.fail_on_integrity_error = fail_on_integrity_error

        # Input format resolution / preprocessing routing
        os.makedirs(self.output_dir, exist_ok=True)
        if self.input_format == "auto":
            if str(self.data_path).endswith(".csv"):
                self.input_format = "csv"
            elif str(self.data_path).endswith(".pkl"):
                self.input_format = "pkl"
            else:
                raise ValueError(f"Cannot auto-detect format for {self.data_path}")

        if str(self.data_path).endswith(".pkl"):
            self.input_format = "pkl"
            if self.enable_preprocessing:
                self._log("Input already PKL -> disabling preprocessing")
            self.enable_preprocessing = False
            processed_dir = os.path.join(self.output_dir, "cropped_images")
            if os.path.isdir(processed_dir):
                self.image_path = processed_dir

        self._log(f"Format={self.input_format} preproc={self.enable_preprocessing}")

        if self.enable_preprocessing:
            self._initialize_with_preprocessing()
        else:
            self._initialize_without_preprocessing()

    # ---------------- Logging -----------------
    def _log(self, msg: str, level: int = logging.INFO):
        """Safe logging (no print). Also appends to log_file if provided."""
        line = f"[DATALOADER] {datetime.utcnow().strftime('%H:%M:%S')} {msg}"
        try:
            logger.log(level, line)
        except Exception:
            # Never crash due to logging.
            pass

        if self.log_file:
            try:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                with open(self.log_file, "a") as f:
                    f.write(line + "\n")
            except Exception:
                # Never crash due to file logging
                pass

    # ---------------- Preprocessing orchestrator -----------------
    def _initialize_with_preprocessing(self):
        cache_pkl = os.path.join(self.output_dir, "processed_exam_list.pkl")
        cache_info = os.path.join(self.output_dir, "preprocessing_cache_info.json")
        cropped_dir = os.path.join(self.output_dir, "cropped_images")
        cropped_list_path = os.path.join(self.output_dir, "cropped_exam_list.pkl")
        manifest_path = os.path.join(self.output_dir, "preprocessing_progress.json")

        if os.path.exists(cache_pkl) and os.path.exists(cache_info) and self._validate_preprocessing_cache(cache_info):
            self._log("Reuse: full cache")
            self.exam_list = pickling.unpickle_from_file(cache_pkl)
            integrity_ok = True
            if self.file_integrity_check:
                stats = self._verify_cropped_files(self.exam_list, cropped_dir)
                integrity_ok = (stats["missing_count"] == 0)
                if not integrity_ok:
                    if self.fail_on_integrity_error:
                        self._log(f"❌ Integrity mismatch (missing={stats['missing_count']} extra={stats['extra_count']}) -> forcing rebuild", level=logging.WARNING)
                    else:
                        self._log(f"⚠️ Integrity mismatch (missing={stats['missing_count']} extra={stats['extra_count']}) continuing anyway (set fail_on_integrity_error=True to rebuild)", level=logging.WARNING)
            if integrity_ok or not self.fail_on_integrity_error:
                self.data_path = cache_pkl
                if os.path.isdir(cropped_dir):
                    self.image_path = cropped_dir
                self.input_format = "pkl"
                self._create_data_splits()
                return
            else:
                self._log("Proceeding to rebuild due to integrity failure", level=logging.WARNING)

        # Only resume at "Stage2 only" if the Stage-1 cropped PNGs actually exist on disk.
        # Guards the force_preprocessing race (cropped_images/ deleted but cropped_exam_list.pkl
        # survived) and any cache where the PNGs went missing: resuming there would fail center
        # extraction and silently train WITHOUT best-centers. If the crops are gone, fall through
        # to the full pipeline (re-crop) instead. Honors tolerant_missing_metadata to opt out.
        crops_on_disk = os.path.isdir(cropped_dir) and any(
            f.lower().endswith(".png") for f in os.listdir(cropped_dir)
        )
        if os.path.exists(cropped_list_path) and not crops_on_disk and not self.tolerant_missing_metadata:
            self._log(
                f"Stage-1 list present but cropped_images/ is empty/missing ({cropped_dir}); "
                "cannot resume at Stage2 (center extraction needs the crops). Rebuilding from scratch.",
                level=logging.WARNING,
            )
        elif os.path.exists(cropped_list_path):
            try:
                cropped_exam_list = pickling.unpickle_from_file(cropped_list_path)
                if self._cropping_metadata_present(cropped_exam_list):
                    self._log("Resume: Stage2 only")
                    final_exam_list, s2_stats = self._stage2_extract_centers(cropped_exam_list)
                    self.exam_list = final_exam_list
                    self._save_preprocessing_cache()
                    self.data_path = cache_pkl
                    if os.path.isdir(cropped_dir):
                        self.image_path = cropped_dir
                    self.input_format = "pkl"
                    self._create_data_splits()
                    self._update_progress_manifest(manifest_path, "stage2_complete", len(self.exam_list), **s2_stats)
                    return
                if self.tolerant_missing_metadata:
                    self._log("Tolerant: skipping Stage2 (metadata missing)", level=logging.WARNING)
                    self.exam_list = cropped_exam_list
                    self.data_path = cropped_list_path
                    if os.path.isdir(cropped_dir):
                        self.image_path = cropped_dir
                    self.input_format = "pkl"
                    self._create_data_splits()
                    stats = {
                        "stage1_images": self._count_images(cropped_exam_list),
                        "stage2_centers_assigned": 0,
                        "stage2_images_considered": 0,
                        "stage2_rescue_used": False,
                        "stage2_images_dropped": 0,
                        "tolerant_mode_used": True,
                    }
                    self._update_progress_manifest(manifest_path, "stage1_complete_no_centers", len(self.exam_list), **stats)
                    return
                self._log("Resume failed metadata check -> rebuild", level=logging.WARNING)
            except Exception as e:
                self._log(f"Resume read failed: {e}", level=logging.WARNING)

        # Full pipeline
        if self.input_format == "csv":
            self.df = pd.read_csv(self.data_path)
            initial_exam_list = self._convert_csv_to_initial_format()
        elif self.input_format == "raw_pkl":
            initial_exam_list = pickling.unpickle_from_file(self.data_path)
        else:
            raise ValueError(f"Unsupported format for preprocessing: {self.input_format}")

        cropped_exam_list, s1_stats = self._stage1_crop_mammograms(initial_exam_list, return_stats=True)
        self._update_progress_manifest(manifest_path, "stage1_complete", len(cropped_exam_list), **s1_stats)

        final_exam_list, s2_stats = self._stage2_extract_centers(cropped_exam_list)
        self.exam_list = final_exam_list

        self._save_preprocessing_cache()
        self.data_path = cache_pkl
        if os.path.isdir(cropped_dir):
            self.image_path = cropped_dir
        self.input_format = "pkl"

        self._log("Validating final training data...")
        if self._validate_training_data(self.exam_list):
            self._log("Data validation passed")
        else:
            self._log("Data validation found issues - training may be affected", level=logging.WARNING)

        self._create_data_splits()

        merged_stats = {**s1_stats, **s2_stats}
        self._update_progress_manifest(manifest_path, "stage2_complete", len(self.exam_list), **merged_stats)
        self._log("Pipeline complete")

    def _initialize_without_preprocessing(self):
        self._log("📂 Using existing data (no preprocessing)")
        if self.input_format == "csv":
            self.df = pd.read_csv(self.data_path)
            self.exam_list = self._convert_csv_to_gmic_format()
        elif self.input_format == "pkl":
            self.exam_list = pickling.unpickle_from_file(self.data_path)
            self.df = None
            processed_dir = os.path.join(self.output_dir, "cropped_images")
            if os.path.isdir(processed_dir):
                self.image_path = processed_dir
                self._log(f"image_path -> {self.image_path}")
        else:
            raise ValueError(f"Unknown input_format: {self.input_format}")
        self._create_data_splits()

    # ---------- Converters / helpers ----------
    def _validate_labels(self):
        """R1: hard label-contract check on the CSV before conversion.

        Fails loudly if view/exam labels are not strictly {0,1} (or contain NaN), logs
        per-value counts, and asserts each breast (patient, exam, laterality) has its two
        views agreeing on a single label. A disagreement means the data is malformed
        (contralateral-breast mislabeling, etc.) and must surface, not be silently
        resolved -- which also reconciles the two converters (their differing
        last-wins vs sticky-to-1 rules become equivalent once views are guaranteed to agree).
        """
        df = getattr(self, "df", None)
        if df is None:
            return
        for col in ("view_level_label", "exam_level_label"):
            if col not in df.columns:
                continue
            vals = df[col]
            try:
                self._log(f"[labels] {col} value_counts: {vals.value_counts(dropna=False).to_dict()}")
            except Exception:
                pass
            if vals.isna().any():
                raise ValueError(f"[labels] {col} contains NaN; labels must be strictly 0/1")
            bad = set(int(v) if float(v).is_integer() else v for v in vals.unique()) - {0, 1}
            if bad:
                raise ValueError(f"[labels] {col} has non-binary values {bad}; must be strictly 0/1")
        if {"patient_id", "exam_id", "laterality", "view_level_label"}.issubset(df.columns):
            nun = df.groupby(["patient_id", "exam_id", "laterality"])["view_level_label"].nunique()
            bad_breasts = nun[nun > 1]
            if len(bad_breasts) > 0:
                raise ValueError(
                    f"[labels] {len(bad_breasts)} breast(s) have disagreeing view_level_label "
                    f"across their views (e.g. {list(bad_breasts.index[:3])}). Breast-level "
                    f"labels must agree by construction; fix the data."
                )
        self._log("[labels] contract OK (strictly 0/1; breast-level views agree)")

    def _convert_csv_to_gmic_format(self):
        self._validate_labels()
        exam_list = []
        grouped = self.df.groupby(["patient_id", "exam_id"])
        for (patient_id, exam_id), group in grouped:
            exam = {
                "patient_id": patient_id,
                "exam_id": exam_id,
                "horizontal_flip": group.iloc[0].get("horizontal_flip", "NO"),
                "cancer_label": {},
                "best_center": {},
                "file_paths": {},
            }
            for view in ["L-CC", "L-MLO", "R-CC", "R-MLO"]:
                exam[view] = []
                exam["best_center"][view] = []
                exam["file_paths"][view] = []
            lvl = group.iloc[0]["exam_level_label"]
            exam["cancer_label"] = {
                "benign": 1 if lvl == 0 else 0,
                "malignant": 1 if lvl == 1 else 0,
                "left_benign": 0,
                "right_benign": 0,
                "left_malignant": 0,
                "right_malignant": 0,
                "unknown": 0,
            }
            for _, row in group.iterrows():
                laterality = row["laterality"]
                view = row["view"]
                full_view = f"{laterality}-{view}"
                image_id = f"{patient_id}_{exam_id}_{laterality}_{view}"
                exam[full_view].append(image_id)
                exam["best_center"][full_view].append((128, 128))
                exam["file_paths"][full_view].append(row["file_path"])
                view_label = row["view_level_label"]
                if laterality == "L":
                    exam["cancer_label"]["left_malignant"] = int(view_label == 1)
                    exam["cancer_label"]["left_benign"] = int(view_label == 0)
                else:
                    exam["cancer_label"]["right_malignant"] = int(view_label == 1)
                    exam["cancer_label"]["right_benign"] = int(view_label == 0)
            if "split_group" in group.columns:
                exam["split_group"] = group.iloc[0]["split_group"]
            exam_list.append(exam)
        return exam_list

    def _create_data_splits(self):
        all_data = self._unpack_exam_into_images(self.exam_list)
        if self.use_predefined_splits and self._has_predefined_splits():
            self._use_predefined_splits(all_data)
        else:
            self._create_automatic_splits(all_data)

    def _has_predefined_splits(self):
        if self.input_format == "csv" and getattr(self, "df", None) is not None and "split_group" in self.df.columns:
            return True
        return any("split_group" in exam for exam in self.exam_list)

    def _use_predefined_splits(self, all_data):
        mapping = {"train": "train", "dev": "val", "val": "val", "test": "test"}
        self.train_data, self.val_data, self.test_data = [], [], []
        for d in all_data:
            sg = d.get("split_group", "train")
            tgt = mapping.get(str(sg).lower(), "train")
            if tgt == "train":
                self.train_data.append(d)
            elif tgt == "val":
                self.val_data.append(d)
            elif tgt == "test":
                self.test_data.append(d)
        self._log(f"✅ Using predefined splits: train={len(self.train_data)} val={len(self.val_data)} test={len(self.test_data)}")

    def _create_automatic_splits(self, all_data):
        if self.input_format == "csv":
            ids = list({f"{d['patient_id']}_{d['exam_id']}" for d in all_data})
        else:
            ids = list({d["exam_id"] for d in all_data})

        if self.test_split > 0:
            train_val_ids, test_ids = train_test_split(ids, test_size=self.test_split, random_state=self.random_seed)
        else:
            train_val_ids, test_ids = ids, []

        if self.val_split > 0 and len(train_val_ids) > 1:
            train_ids, val_ids = train_test_split(
                train_val_ids,
                test_size=self.val_split / (1 - self.test_split),
                random_state=self.random_seed + 1,
            )
        else:
            train_ids, val_ids = train_val_ids, []

        if self.input_format == "csv":
            self.train_data = [d for d in all_data if f"{d['patient_id']}_{d['exam_id']}" in train_ids]
            self.val_data = [d for d in all_data if f"{d['patient_id']}_{d['exam_id']}" in val_ids]
            self.test_data = [d for d in all_data if f"{d['patient_id']}_{d['exam_id']}" in test_ids]
        else:
            self.train_data = [d for d in all_data if d["exam_id"] in train_ids]
            self.val_data = [d for d in all_data if d["exam_id"] in val_ids]
            self.test_data = [d for d in all_data if d["exam_id"] in test_ids]

        self._log(f"✅ Created automatic splits: train_imgs={len(self.train_data)} val_imgs={len(self.val_data)} test_imgs={len(self.test_data)}")

    def _unpack_exam_into_images(self, exam_list):
        data_list = []
        for exam_idx, exam in enumerate(exam_list):
            for view in VIEWS.LIST:
                if view in exam and exam[view]:
                    for img_idx, image_identifier in enumerate(exam[view]):
                        # best_center is INERT (image is pre-standardized to the model window;
                        # _process_image_to_2944x1920 falls back to image-center when this is None).
                        # Read defensively: a short/missing best_center list (e.g. a stale cache from
                        # a pre-neutralization run) must yield None, never IndexError.
                        bc_for_view = exam.get("best_center", {}).get(view, []) if "best_center" in exam else []
                        best_center = bc_for_view[img_idx] if img_idx < len(bc_for_view) else None
                        datum = {
                            "exam_id": exam_idx,
                            "image_id": image_identifier,
                            "short_file_path": image_identifier,
                            "view": view,
                            "horizontal_flip": exam["horizontal_flip"],
                            "best_center": best_center,
                            "cancer_label": exam["cancer_label"],
                        }
                        if "file_paths" in exam and view in exam["file_paths"] and img_idx < len(exam["file_paths"][view]):
                            datum["full_file_path"] = exam["file_paths"][view][img_idx]
                        else:
                            datum["full_file_path"] = None

                        cl = exam["cancer_label"]
                        datum["exam_level_label"] = cl.get("malignant", 0)
                        datum["view_level_label"] = self._extract_view_level_label(cl, view)

                        if "patient_id" in exam:
                            datum["patient_id"] = exam["patient_id"]
                        if "split_group" in exam:
                            datum["split_group"] = exam["split_group"]
                        data_list.append(datum)
        return data_list

    def _extract_view_level_label(self, cancer_label, view):
        side_code = view.split("-")[0].upper()
        side = "left" if side_code == "L" else "right"
        if cancer_label.get(f"{side}_malignant", 0) == 1:
            return 1
        if cancer_label.get(f"{side}_benign", 0) == 1:
            return 0
        return 0

    # ---------- Cache Validation ----------
    def _validate_preprocessing_cache(self, cache_info_path):
        if not self.cache_validation:
            return True
        try:
            with open(cache_info_path, "r") as f:
                info = json.load(f)
            current_hash = self._get_input_data_hash()
            if info.get("input_hash") != current_hash:
                self._log("⚠️ Input data changed; invalidating cache", level=logging.WARNING)
                return False
            self._log("✅ Preprocessing cache is valid")
            return True
        except Exception as e:
            self._log(f"⚠️ Cache validation error: {e}", level=logging.WARNING)
            return False

    def _validate_training_data(self, exam_list):
        issues = []
        for i, exam in enumerate(exam_list):
            if "best_center" not in exam:
                issues.append(f"Exam {i}: missing best_center")
            else:
                for view in ["L-CC", "L-MLO", "R-CC", "R-MLO"]:
                    if view in exam and exam[view]:
                        if view not in exam["best_center"]:
                            issues.append(f"Exam {i}: missing best_center for {view}")

        if issues:
            self._log(f"Data validation found {len(issues)} issues:", level=logging.WARNING)
            for issue in issues[:10]:
                self._log(f"  {issue}", level=logging.WARNING)

        return len(issues) == 0

    def _get_input_data_hash(self):
        hasher = hashlib.md5()
        with open(self.data_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def _save_preprocessing_cache(self):
        cache_pkl = os.path.join(self.output_dir, "processed_exam_list.pkl")
        cache_info = os.path.join(self.output_dir, "preprocessing_cache_info.json")
        pickling.pickle_to_file(cache_pkl, self.exam_list)
        meta = {
            "input_hash": self._get_input_data_hash(),
            "timestamp": datetime.utcnow().isoformat(),
            "total_exams": len(self.exam_list),
        }
        with open(cache_info, "w") as f:
            json.dump(meta, f, indent=2)
        self._log(f"💾 Cached processed data: {cache_pkl}")

    # ---------- Progress Manifest (auto-resume bookkeeping) ----------
    def _load_progress_manifest(self, path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _update_progress_manifest(self, path, status, exams, **extra):
        try:
            manifest = self._load_progress_manifest(path)
            manifest["status"] = status
            manifest.setdefault("timestamps", {})[status] = datetime.utcnow().isoformat()
            manifest["exams"] = exams
            manifest["input_hash"] = self._get_input_data_hash()
            if extra:
                stats = manifest.setdefault("stats", {})
                for k, v in extra.items():
                    stats[k] = v
            with open(path, "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            self._log(f"⚠️ Could not write progress manifest: {e}", level=logging.WARNING)

    # ---------- Metadata Checks ----------
    def _cropping_metadata_present(self, exam_list):
        required = ["window_location", "rightmost_points", "bottommost_points", "distance_from_starting_side"]
        if not exam_list:
            return False
        for r in required:
            if r not in exam_list[0]:
                return False
        sample = exam_list[0]
        for r in required:
            v = sample.get(r, {})
            if not isinstance(v, dict):
                return False
        return True

    # ---------- Staging & Cropping ----------
    def _convert_csv_to_initial_format(self):
        self._validate_labels()
        exam_list = []
        grouped = self.df.groupby(["patient_id", "exam_id"])
        for (patient_id, exam_id), group in grouped:
            exam = {
                "patient_id": patient_id,
                "exam_id": exam_id,
                "horizontal_flip": group.iloc[0].get("horizontal_flip", "NO"),
                "cancer_label": {},
                "original_file_paths": {},
            }
            for v in ["L-CC", "L-MLO", "R-CC", "R-MLO"]:
                exam[v] = []
                exam["original_file_paths"][v] = []
            lvl = group.iloc[0]["exam_level_label"]
            exam["cancer_label"] = {
                "benign": 1 if lvl == 0 else 0,
                "malignant": 1 if lvl == 1 else 0,
                "left_benign": 0,
                "right_benign": 0,
                "left_malignant": 0,
                "right_malignant": 0,
                "unknown": 0,
            }
            for _, row in group.iterrows():
                laterality = row["laterality"]
                view = row["view"]
                full_view = f"{laterality}-{view}"
                image_id = f"{patient_id}_{exam_id}_{laterality}_{view}"
                exam[full_view].append(image_id)
                exam["original_file_paths"][full_view].append(row["file_path"])
                view_label = row["view_level_label"]
                if laterality == "L":
                    if view_label == 1:
                        exam["cancer_label"]["left_malignant"] = 1
                    else:
                        exam["cancer_label"]["left_benign"] = 1
                else:
                    if view_label == 1:
                        exam["cancer_label"]["right_malignant"] = 1
                    else:
                        exam["cancer_label"]["right_benign"] = 1
            if "split_group" in group.columns:
                exam["split_group"] = group.iloc[0]["split_group"]
            exam_list.append(exam)

        self._log(f"Converted CSV to initial format: {len(exam_list)} exams")
        return exam_list

    def _prepare_staging_area(self, initial_exam_list, staging_dir):
        expected, copied, missing = [], [], []
        for exam in initial_exam_list:
            for v in ["L-CC", "L-MLO", "R-CC", "R-MLO"]:
                if v in exam and exam[v] and "original_file_paths" in exam:
                    for image_id, src in zip(exam[v], exam["original_file_paths"][v]):
                        expected.append(image_id)
                        dst = os.path.join(staging_dir, f"{image_id}.png")
                        if src and os.path.exists(src):
                            try:
                                shutil.copy2(src, dst)
                                copied.append(image_id)
                                if self.per_file_logging:
                                    self._log(f"[STAGING_FILE] id={image_id} status=copied src={src} dst={dst}")
                            except Exception as e:
                                missing.append((image_id, f"copy_error:{e}"))
                                if self.per_file_logging:
                                    self._log(f"[STAGING_FILE] id={image_id} status=copy_error err={e} src={src}", level=logging.WARNING)
                        else:
                            missing.append((image_id, "source_missing"))
                            if self.per_file_logging:
                                self._log(f"[STAGING_FILE] id={image_id} status=source_missing src={src}", level=logging.WARNING)
        self._log(f"[STAGING] expected={len(expected)} copied={len(copied)} missing={len(missing)}")
        if missing:
            self._log(f"[STAGING] sample_missing={', '.join(f'{m[0]}({m[1]})' for m in missing[:10])}", level=logging.WARNING)
        if copied:
            self._log(f"[STAGING] sample_copied={', '.join(copied[:10])}")
        return expected, copied, missing

    def _assert_cropped_schema(self, exam_list):
        required_meta = ["window_location", "rightmost_points", "bottommost_points", "distance_from_starting_side"]
        if not exam_list:
            self._log("[CROP_SCHEMA] empty exam list", level=logging.WARNING)
            return
        sample = exam_list[0]
        missing = [k for k in required_meta if k not in sample]
        if missing:
            self._log(f"[CROP_SCHEMA][WARN] missing keys: {missing}", level=logging.WARNING)
        else:
            self._log("[CROP_SCHEMA] metadata keys present")
        for view in ["L-CC", "L-MLO", "R-CC", "R-MLO"]:
            if view in sample and sample[view] and not isinstance(sample[view], list):
                self._log(f"[CROP_SCHEMA][WARN] view {view} not list type", level=logging.WARNING)

    def _stage1_crop_mammograms(self, initial_exam_list, return_stats=False):
        self._log("Stage 1: Cropping mammograms")
        staging_dir = os.path.join(self.output_dir, "staging")
        cropped_dir = os.path.join(self.output_dir, "cropped_images")
        initial_list_path = os.path.join(self.output_dir, "initial_exam_list.pkl")
        cropped_list_path = os.path.join(self.output_dir, "cropped_exam_list.pkl")
        os.makedirs(staging_dir, exist_ok=True)

        if os.path.exists(cropped_dir) and os.path.exists(cropped_list_path):
            try:
                ce = pickling.unpickle_from_file(cropped_list_path)
                self._assert_cropped_schema(ce)
                self._log(f"Using cached cropped data: {len(ce)} exams")
                stats = {"stage1_images": self._count_images(ce), "stage1_expected_images": None}
                return (ce, stats) if return_stats else ce
            except Exception:
                shutil.rmtree(cropped_dir, ignore_errors=True)
                os.makedirs(cropped_dir, exist_ok=True)

        os.makedirs(cropped_dir, exist_ok=True)
        expected_ids, copied_ids, missing_entries = self._prepare_staging_area(initial_exam_list, staging_dir)
        pickling.pickle_to_file(initial_list_path, initial_exam_list)

        staged_files = [f for f in os.listdir(staging_dir) if f.lower().endswith(".png")]
        self._log(f"[STAGE1] staged_pngs={len(staged_files)} (copied={len(copied_ids)})")

        try:
            crop_mammogram(
                input_data_folder=staging_dir,
                exam_list_path=initial_list_path,
                cropped_exam_list_path=cropped_list_path,
                output_data_folder=cropped_dir,
                num_processes=10,
                num_iterations=100,
                buffer_size=50,
                error_log_path=os.path.join(cropped_dir, "crop_failures.txt"),
                logger_fn=self._crop_logger,
            )
            cropped_exam_list = pickling.unpickle_from_file(cropped_list_path)
            self._assert_cropped_schema(cropped_exam_list)

            produced = [f for f in os.listdir(cropped_dir) if f.lower().endswith(".png")]
            produced_ids = {os.path.splitext(f)[0] for f in produced}
            missing_after = set(expected_ids) - produced_ids

            self._log(f"[STAGE1] produced_pngs={len(produced)}")
            if missing_after:
                self._log(f"[STAGE1] missing_after_count={len(missing_after)} sample={', '.join(list(missing_after)[:10])}", level=logging.WARNING)

            self._log(f"Stage 1 complete: {len(cropped_exam_list)} exams")
            stats = {"stage1_images": self._count_images(cropped_exam_list), "stage1_expected_images": len(expected_ids)}

            if hasattr(self, "filter_failed_exams") and self.filter_failed_exams:
                cropped_exam_list = self._filter_exams_by_success_rate(cropped_exam_list, min_success_rate=0.5)

            return (cropped_exam_list, stats) if return_stats else cropped_exam_list

        except Exception as e:
            self._log(f"Cropping failed: {e}", level=logging.WARNING)
            stats = {"stage1_images": self._count_images(initial_exam_list), "stage1_expected_images": len(expected_ids)}
            return (initial_exam_list, stats) if return_stats else initial_exam_list

    # ---- Crop logger (pickle-safe) ----
    def _crop_logger(self, msg: str):
        if (not self.per_file_logging) and (msg.startswith("START ") or msg.startswith("DONE ")):
            return
        self._log(f"[CROP] {msg}")

    def _stage2_extract_centers(self, cropped_exam_list):
        """Stage 2: assign best_center = image center (NEUTRALIZED optimal-center search).

        WHY NEUTRALIZED (not deleted): the optimal-center search is INERT in this fork's
        resize+pad pipeline -- verified two ways: (1) the train/eval augmentation
        (random_augmentation_best_center) is byte-identical for ANY center when the saved
        image already equals the model window 2944x1920 (max|diff|=0.0), and (2) the K=6
        ROI patches come from the model's saliency map, never from best_center (no
        best_center reference anywhere in model/). Upstream's center IS meaningful because
        upstream feeds VARIABLE-size crops and the augmentation does the final centered crop;
        this fork's added resize_and_pad defeats that. So here best_center only needs to
        populate the field the validators expect; image-center is provably as good as any value.

        The optimal-center machinery (get_optimal_centers / calc_optimal_centers) is left in
        the tree, unimported here, RECOVERABLE: if the per-site padding diagnostic shows the
        resize approach's scale-variability threatens the spatial-scale analysis, we may revert
        to upstream variable-crop (B'), which needs this machinery back. Decision deferred to
        the padding numbers; do not delete until then.
        """
        self._log("Stage 2: center assignment (image-center; optimal-search neutralized)")
        cropped_list_path = os.path.join(self.output_dir, "cropped_exam_list.pkl")
        output_list_path = os.path.join(self.output_dir, "final_exam_list.pkl")
        pickling.pickle_to_file(cropped_list_path, cropped_exam_list)

        data_list = data_handling.unpack_exam_into_images(cropped_exam_list, cropped=True)
        base_stats = {
            "stage2_centers_assigned": 0,
            "stage2_images_considered": len(data_list),
            "stage2_rescue_used": False,
            "stage2_images_dropped": 0,
            "tolerant_mode_used": False,
        }
        if not data_list:
            raise RuntimeError("Stage 2: no images available for center assignment")

        cy, cx = TARGET_H // 2, TARGET_W // 2
        centers = {d["short_file_path"]: (cy, cx) for d in data_list}
        data_handling.add_metadata(cropped_exam_list, "best_center", centers)
        pickling.pickle_to_file(output_list_path, cropped_exam_list)

        base_stats["stage2_centers_assigned"] = len(centers)
        self._log(f"✅ Stage 2 complete: assigned image-center ({cy},{cx}) to {len(centers)} images")
        return cropped_exam_list, base_stats

    # -------------- Info & Stats --------------
    def get_split_info(self):
        info = {
            "train_size": len(self.train_data),
            "val_size": len(self.val_data),
            "test_size": len(self.test_data),
            "total_size": len(self.train_data) + len(self.val_data) + len(self.test_data),
            "input_format": self.input_format,
            "preprocessing_enabled": self.enable_preprocessing,
        }
        if self.input_format == "csv":
            train_p = {d.get("patient_id") for d in self.train_data if "patient_id" in d}
            val_p = {d.get("patient_id") for d in self.val_data if "patient_id" in d}
            test_p = {d.get("patient_id") for d in self.test_data if "patient_id" in d}
            info.update(
                {
                    "train_patients": len(train_p),
                    "val_patients": len(val_p),
                    "test_patients": len(test_p),
                    "total_patients": len(train_p | val_p | test_p),
                }
            )
        return info

    # ---------- Status Utilities ----------
    def print_preprocessing_status(self):
        """Log manifest status (no print)."""
        path = os.path.join(self.output_dir, "preprocessing_progress.json")
        if not os.path.exists(path):
            self._log("No manifest found.", level=logging.WARNING)
            return
        try:
            with open(path, "r") as f:
                manifest = json.load(f)
        except Exception as e:
            self._log(f"Failed to read manifest: {e}", level=logging.WARNING)
            return

        self._log("Preprocessing Status Manifest")
        self._log(f"Status: {manifest.get('status')}")
        self._log(f"Exams: {manifest.get('exams')}")
        self._log(f"Input Hash: {manifest.get('input_hash')}")

        ts = manifest.get("timestamps", {})
        if ts:
            self._log("Timestamps:")
            for k, v in ts.items():
                self._log(f"  {k}: {v}")

        stats = manifest.get("stats", {})
        if stats:
            self._log("Stats:")
            for k, v in stats.items():
                self._log(f"  {k}: {v}")

    def validate_auto_resume(self):
        manifest_path = os.path.join(self.output_dir, "preprocessing_progress.json")
        if not os.path.exists(manifest_path):
            return False, "manifest_missing"
        try:
            with open(manifest_path, "r") as f:
                m = json.load(f)
        except Exception as e:
            return False, f"manifest_read_error:{e}"
        status = m.get("status")
        expected_hash = m.get("input_hash")
        actual_hash = self._get_input_data_hash() if os.path.exists(self.data_path) else None
        if expected_hash and actual_hash and expected_hash != actual_hash:
            return False, "input_hash_mismatch"

        def _exists(name):
            return os.path.exists(os.path.join(self.output_dir, name))

        if status == "stage2_complete":
            if not _exists("processed_exam_list.pkl"):
                return False, "processed_exam_list_missing"
        elif isinstance(status, str) and status.startswith("stage1_complete"):
            if not _exists("cropped_exam_list.pkl"):
                return False, "cropped_exam_list_missing"
        return True, "ok"

    # ---------- Helpers ----------
    def _count_images(self, exam_list):
        c = 0
        for ex in exam_list:
            for v in ["L-CC", "L-MLO", "R-CC", "R-MLO"]:
                if v in ex and isinstance(ex[v], list):
                    c += len(ex[v])
        return c

    def get_class_distribution(self, split="train"):
        data = self.get_data_for_split(split)
        exam_level_counts = defaultdict(int)
        view_level_counts = defaultdict(int)
        for d in data:
            exam_level_counts[d["exam_level_label"]] += 1
            view_level_counts[d["view_level_label"]] += 1
        return {"exam_level": dict(exam_level_counts), "view_level": dict(view_level_counts), "total": len(data)}

    def save_pkl_format(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pickling.pickle_to_file(path, self.exam_list)
        self._log(f"Saved exam list to: {path}")

    def print_summary(self):
        """Log a concise summary (no print)."""
        si = self.get_split_info()
        self._log("============================================================")
        self._log("GMIC DATA LOADER SUMMARY")
        self._log("============================================================")
        self._log(f"SUMMARY format={self.input_format} preproc={self.enable_preprocessing}")
        self._log(f"SUMMARY total_images={si['total_size']} total_exams={len(self.exam_list)}")
        if "total_patients" in si:
            self._log(f"SUMMARY total_patients={si['total_patients']}")
        self._log("Data Splits:")
        self._log(f"  Train: {si['train_size']}")
        if "train_patients" in si:
            self._log(f"    patients={si['train_patients']}")
        self._log(f"  Val:   {si['val_size']}")
        if "val_patients" in si:
            self._log(f"    patients={si['val_patients']}")
        self._log(f"  Test:  {si['test_size']}")
        if "test_patients" in si:
            self._log(f"    patients={si['test_patients']}")
        for split in ["train", "val", "test"]:
            if self.get_data_for_split(split):
                dist = self.get_class_distribution(split)
                self._log(f"{split} dist exam:{dist['exam_level']} view:{dist['view_level']}")

    # -------------- Batching ------------------
    def __len__(self):
        return len(self.train_data)

    def _balanced_index_order(self, samples, rng):
        """WeightedRandomSampler-equivalent index order (with replacement) for the custom
        iterator (there is no torch DataLoader to attach a sampler to). Each view-level
        sample is drawn with prob inversely proportional to its class frequency, so batches
        see a ~even pos/neg mix. Epoch size (len) is preserved.
        """
        labels = np.array([int(s.get("view_level_label", s.get("exam_level_label", 0)))
                           for s in samples])
        n = len(labels)
        if n == 0:
            return list(range(n))
        n_pos = int((labels == 1).sum())
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return list(range(n))  # degenerate: nothing to balance
        w_pos, w_neg = 0.5 / n_pos, 0.5 / n_neg
        weights = np.where(labels == 1, w_pos, w_neg)
        weights = weights / weights.sum()
        order = rng.choice(n, size=n, replace=True, p=weights)
        return order.tolist()

    def get_batch_iterator(self, split: str):
        is_train = split == "train"
        rng = self._rng_train if is_train else self._rng_eval
        samples = self.get_data_for_split(split)
        # Balanced sampling (train only): resample indices by inverse class frequency.
        if is_train and getattr(self, "use_balanced_sampler", False):
            index_order = self._balanced_index_order(samples, rng)
        else:
            index_order = range(len(samples))
        batch_images, batch_targets, batch_meta = [], [], []
        for idx in index_order:
            row = samples[idx]
            path = self._resolve_image_path(row)
            if path is None:
                continue
            try:
                img = _load_image(path, row["view"], row.get("horizontal_flip", "NO"))
                chw = _process_image_to_2944x1920(
                    image=img,
                    view=row["view"],
                    best_center=row.get("best_center"),
                    rng=rng,
                    is_train=is_train,
                    max_crop_noise=self.train_max_crop_noise,
                    max_crop_size_noise=self.train_max_crop_size_noise,
                )
            except Exception as e:
                # R2: a single corrupt/unreadable image must not abort a multi-hour run.
                self._log(f"[image] skipping unreadable/bad image {path}: {e}", level=logging.WARNING)
                continue
            # Conservative augmentation: TRAIN SPLIT ONLY. val/test never reach this branch, so
            # their tensors are byte-identical to prior runs (deterministic eval preserved).
            if is_train and self.use_augmentation:
                aug2d = augmentations.conservative_train_augment(
                    chw[0], self._rng_aug,
                    max_rotation_deg=self.aug_cfg["max_rotation_deg"],
                    intensity_jitter=self.aug_cfg["intensity_jitter"],
                    horizontal_flip=self.aug_cfg["horizontal_flip"],
                    view=row["view"],
                )
                chw = aug2d[None, ...]
            batch_images.append(chw[None, ...])
            target = row.get("view_level_label", row.get("exam_level_label", 0))
            batch_targets.append(int(target))
            batch_meta.append({"exam_id": row.get("exam_id"), "view": row["view"], "path": path})
            if len(batch_images) == self.batch_size:
                yield (
                    torch.from_numpy(np.concatenate(batch_images, axis=0)),
                    torch.tensor(batch_targets, dtype=torch.long),
                    batch_meta,
                )
                batch_images, batch_targets, batch_meta = [], [], []
        if batch_images:
            yield (
                torch.from_numpy(np.concatenate(batch_images, axis=0)),
                torch.tensor(batch_targets, dtype=torch.long),
                batch_meta,
            )

    def _resolve_image_path(self, datum):
        candidates = []
        if datum.get("full_file_path"):
            candidates.append(datum["full_file_path"])
        if self.image_path:
            candidates.append(os.path.join(self.image_path, f"{datum['short_file_path']}.png"))
        candidates.append(os.path.join(self.output_dir, "cropped_images", f"{datum['short_file_path']}.png"))
        candidates.append(os.path.join(self.output_dir, "staging", f"{datum['short_file_path']}.png"))
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _filter_exams_by_success_rate(self, exam_list, min_success_rate=0.5):
        filtered_exams = []
        for exam in exam_list:
            total_images = sum(len(exam[view]) for view in ["L-CC", "L-MLO", "R-CC", "R-MLO"] if view in exam)
            if "window_location" in exam:
                successful_images = sum(
                    len(exam["window_location"][view])
                    for view in ["L-CC", "L-MLO", "R-CC", "R-MLO"]
                    if view in exam["window_location"]
                )
                success_rate = successful_images / total_images if total_images > 0 else 0
                if success_rate >= min_success_rate:
                    filtered_exams.append(exam)
                else:
                    self._log(f"Removing exam {exam.get('exam_id', 'unknown')} - success rate {success_rate:.2f} < {min_success_rate}", level=logging.WARNING)
            else:
                self._log(f"Removing exam {exam.get('exam_id', 'unknown')} - no cropping metadata", level=logging.WARNING)
        self._log(f"Filtered exams: {len(filtered_exams)}/{len(exam_list)} retained")
        return filtered_exams

    # ---------- File Integrity Verification ----------
    def _verify_cropped_files(self, exam_list, cropped_dir):
        if not os.path.isdir(cropped_dir):
            self._log(f"[INTEGRITY] cropped_dir missing: {cropped_dir}", level=logging.WARNING)
            return {"expected": 0, "found": 0, "missing_count": 0, "extra_count": 0, "missing_sample": [], "extra_sample": []}

        expected_ids = set()
        for ex in exam_list:
            for v in ["L-CC", "L-MLO", "R-CC", "R-MLO"]:
                ids = ex.get(v, [])
                if isinstance(ids, list):
                    expected_ids.update(ids)

        produced_files = [f for f in os.listdir(cropped_dir) if f.lower().endswith(".png")]
        produced_ids = {os.path.splitext(f)[0] for f in produced_files}
        missing = sorted(expected_ids - produced_ids)
        extra = sorted(produced_ids - expected_ids)

        stats = {
            "expected": len(expected_ids),
            "found": len(produced_ids),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "missing_sample": missing[:10],
            "extra_sample": extra[:10],
        }

        if stats["missing_count"] == 0 and stats["extra_count"] == 0:
            self._log(f"[INTEGRITY] ✅ All {stats['expected']} expected cropped images present")
        else:
            self._log(f"[INTEGRITY] expected={stats['expected']} found={stats['found']} missing={stats['missing_count']} extra={stats['extra_count']}", level=logging.WARNING)
            if stats["missing_sample"]:
                self._log(f"[INTEGRITY] missing_sample={', '.join(stats['missing_sample'])}", level=logging.WARNING)
            if stats["extra_sample"]:
                self._log(f"[INTEGRITY] extra_sample={', '.join(stats['extra_sample'])}", level=logging.WARNING)

        return stats

    def get_data_for_split(self, split="train"):
        if split == "train":
            return self.train_data
        if split in ("val", "validation"):
            return self.val_data
        if split == "test":
            return self.test_data
        raise ValueError(f"Unknown split: {split}")


def _process_image_to_2944x1920(
    image: np.ndarray,
    view: str,
    best_center,
    rng,
    is_train: bool,
    max_crop_noise=(0, 0),
    max_crop_size_noise=0,
):
    crop_noise = max_crop_noise if is_train else (0, 0)
    size_noise = max_crop_size_noise if is_train else 0
    if best_center is None:
        H, W = image.shape[:2]
        best_center = (H // 2, W // 2)
    cropped, _ = augmentations.random_augmentation_best_center(
        image=image,
        input_size=(TARGET_H, TARGET_W),
        random_number_generator=rng,
        max_crop_noise=crop_noise,
        max_crop_size_noise=size_noise,
        auxiliary_image=None,
        best_center=best_center,
        view=view,
    )
    cropped = cropped.copy().astype(np.float32)
    _standard_normalize_single_image(cropped)
    return _to_chw(cropped)


def _flip_image(image: np.ndarray, view: str, horizontal_flip: str):
    if horizontal_flip == "NO":
        if VIEWS.is_right(view):
            image = np.fliplr(image)
    elif horizontal_flip == "YES":
        if VIEWS.is_left(view):
            image = np.fliplr(image)
    return image


def _standard_normalize_single_image(img: np.ndarray):
    img -= np.mean(img)
    img /= max(np.std(img), 1e-5)


def _read_image_png(path: str):
    return np.array(imageio.imread(path))


def _load_image(image_path: str, view: str, horizontal_flip: str):
    if not image_path.lower().endswith("png"):
        raise RuntimeError(f"Only PNG supported: {image_path}")
    img = _read_image_png(image_path).astype(np.float32)
    return _flip_image(img, view, horizontal_flip)


def _to_chw(img: np.ndarray):
    if img.ndim == 2:
        img = img[None, ...]
    elif img.ndim == 3:
        if img.shape[0] not in (1, 3):
            img = np.transpose(img, (2, 0, 1))
    return img.astype(np.float32)
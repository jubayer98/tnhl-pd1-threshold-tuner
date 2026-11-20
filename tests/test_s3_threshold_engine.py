"""Unit Tests for S3 Threshold Engine & Bundled Core Processing."""

import io
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from s3_threshold_engine import (
    CoreBundle,
    S3DataManager,
    hex_to_rgb,
    natural_sort_key,
    normalize_image,
    pil_to_data_uri,
)


class TestS3ThresholdEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.cores_dir = self.test_dir / "cores"
        self.meta_dir = self.test_dir / "metadata"
        self.cur_dir = self.test_dir / "curations"
        self.cores_dir.mkdir(parents=True)
        self.meta_dir.mkdir(parents=True)
        self.cur_dir.mkdir(parents=True)

        # Create mock core data (100x100 pixels, 10 cells)
        self.H, self.W = 100, 100
        self.raw_pd1 = np.zeros((self.H, self.W), dtype=np.float32)
        self.seg = np.zeros((self.H, self.W), dtype=np.int32)
        self.cell_ids = np.arange(1, 11, dtype=np.int64)

        for i, cid in enumerate(self.cell_ids):
            r = (i % 3) * 30 + 10
            c = (i // 3) * 30 + 10
            self.seg[r : r + 15, c : c + 15] = cid
            self.raw_pd1[r : r + 15, c : c + 15] = 10.0 + i * 5.0  # intensities: 10, 15, 20...

        inner_bounds = np.zeros((self.H, self.W), dtype=bool)
        inner_bounds[self.seg > 0] = True
        packed = np.packbits(inner_bounds)

        pixel_counts = np.bincount(self.seg.ravel(), minlength=12).astype(np.int32)
        crit_p50 = np.array([10.0 + i * 5.0 for i in range(10)], dtype=np.float32)

        # Save mock bundle
        self.mock_npz_path = self.cores_dir / "mock-1_pd1.npz"
        np.savez_compressed(
            self.mock_npz_path,
            raw_pd1=self.raw_pd1,
            seg=self.seg,
            cell_ids=self.cell_ids,
            inner_bounds_packed=packed,
            shape=np.array([self.H, self.W], dtype=np.int32),
            cell_pixel_counts=pixel_counts,
            cell_crit_p50=crit_p50,
        )

        # Save mock metadata
        df_meta = pd.DataFrame([
            {
                "core_id": "mock-1",
                "diagnosis": "Test-Diagnosis",
                "subtype": "Subtype-A",
                "ihc_val": 40.0,
                "ihc_str": "40%",
                "is_valid_ihc": True,
                "our_abs_th": 20.0,
                "our_perc_pos_th": 0.5,
                "kei_abs_th": 15.0,
                "kei_perc_pos_th": 0.5,
            }
        ])
        df_meta.to_csv(self.meta_dir / "cohort_metadata.csv", index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_hex_to_rgb(self):
        self.assertEqual(hex_to_rgb("#00E5FF"), (0, 229, 255))
        self.assertEqual(hex_to_rgb("#FFD700"), (255, 215, 0))

    def test_natural_sort_key(self):
        cores = ["1-10", "1-2", "1-1", "1-20"]
        sorted_cores = sorted(cores, key=natural_sort_key)
        self.assertEqual(sorted_cores, ["1-1", "1-2", "1-10", "1-20"])

    def test_bundle_unpacking_and_evaluation(self):
        npz = np.load(self.mock_npz_path)
        bundle = CoreBundle("mock-1", npz)

        self.assertEqual(bundle.total_cells, 10)
        self.assertEqual(bundle.height, self.H)
        self.assertEqual(bundle.width, self.W)

        # Test threshold evaluation:
        # Intensities: 10, 15, 20, 25, 30, 35, 40, 45, 50, 55
        # At abs_th=22: cells with intensity > 22 (values 25..55, 7 cells) are positive
        is_pos, count, pct = bundle.evaluate_threshold(22.0, 0.5)
        self.assertEqual(count, 7)
        self.assertEqual(pct, 70.0)

        # At abs_th=50: only cell with intensity 55 is positive (1 cell)
        is_pos, count, pct = bundle.evaluate_threshold(50.0, 0.5)
        self.assertEqual(count, 1)
        self.assertEqual(pct, 10.0)

    def test_solve_threshold_for_target_pct(self):
        npz = np.load(self.mock_npz_path)
        bundle = CoreBundle("mock-1", npz)

        # 40% positive cells -> 4 cells positive -> intensities > threshold
        # Intensities: 10, 15, 20, 25, 30, 35, 40, 45, 50, 55
        # Top 4 cells have intensities: 40, 45, 50, 55.
        # Threshold between 35 and 40 (e.g. 35) yields 40% positive cells
        th = bundle.solve_threshold_for_target_pct(40.0, 0.5)
        _, count, pct = bundle.evaluate_threshold(th, 0.5)
        self.assertEqual(pct, 40.0)

    def test_render_panel_and_base64(self):
        npz = np.load(self.mock_npz_path)
        bundle = CoreBundle("mock-1", npz)
        is_pos, _, _ = bundle.evaluate_threshold(20.0, 0.5)

        img = bundle.render_panel(abs_th=20.0, is_pos=is_pos, color_hex="#00FF66")
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (self.W, self.H))

        uri = pil_to_data_uri(img)
        self.assertTrue(uri.startswith("data:image/png;base64,"))

    def test_s3_data_manager_local_save(self):
        cfg = {
            "s3": {
                "bucket": "",
                "local_data_dir": str(self.test_dir),
            }
        }
        dm = S3DataManager(cfg)
        df_meta = dm.load_cohort_metadata()
        self.assertEqual(len(df_meta), 1)
        self.assertEqual(df_meta.iloc[0]["core_id"], "mock-1")

        # Save curation decision
        target = dm.save_curation_decision(
            core_id="mock-1",
            decision_type="Accept Custom Threshold",
            abs_th=25.0,
            perc_pos_th=0.5,
            notes="Verified test cell boundaries",
        )
        self.assertTrue(Path(target).is_file())

        cur_csv = pd.read_csv(target)
        self.assertEqual(len(cur_csv), 1)
        self.assertEqual(cur_csv.iloc[0]["core_id"], "mock-1")
        self.assertEqual(cur_csv.iloc[0]["final_abs_th"], 25.0)
        self.assertNotIn("reviewer", cur_csv.columns)


if __name__ == "__main__":
    unittest.main()

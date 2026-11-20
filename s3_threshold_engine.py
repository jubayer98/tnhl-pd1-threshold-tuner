"""S3-Backed High-Performance Thresholding Engine.

Handles streaming compact .npz core bundles from AWS S3 (or local prepared data cache),
in-memory caching, vectorized single-cell threshold evaluation, discrete IHC auto-matching,
Pillow alpha-blending rendering, and direct decision saving to S3 without reviewer column.
"""

from __future__ import annotations

import base64
import datetime
import io
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import yaml

try:
    import streamlit as st
except ImportError:
    st = None


def natural_sort_key(s: str) -> list[int | str]:
    """Sort strings containing numbers in natural human order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def resolve_path(p: str | Path) -> Path:
    """Resolve relative or absolute path."""
    return Path(p).expanduser().resolve()


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex string '#RRGGBB' to RGB tuple (0-255)."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        return (0, 255, 100)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def normalize_image(img: np.ndarray, pmin: float = 3.0, pmax: float = 99.8, eps: float = 1e-6) -> np.ndarray:
    """Percentile normalization matching spatialproteomics."""
    vmin = float(np.percentile(img, pmin))
    vmax = float(np.percentile(img, pmax))
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = (img - vmin) / ((vmax - vmin) + eps)
    return np.clip(norm, 0.0, 1.0)


def pil_to_data_uri(img: Optional[Image.Image]) -> str:
    """Encode a PIL image to a base64 PNG data URI for high-performance Plotly rendering."""
    if img is None:
        img = Image.new("RGB", (10, 10), (15, 18, 24))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1, optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class CoreBundle:
    """In-memory core bundle unpacked from lightweight .npz bundle."""

    def __init__(self, core_id: str, npz_data: Any):
        self.core_id = core_id

        self.raw_pd1 = npz_data["raw_pd1"].astype(np.float32)
        self.seg = npz_data["seg"].astype(np.int32)
        self.cell_ids = npz_data["cell_ids"].astype(np.int64)
        self.total_cells = len(self.cell_ids)
        self.height, self.width = self.raw_pd1.shape

        # Reconstruct bit-packed inner boundaries
        shape = tuple(npz_data["shape"])
        packed = npz_data["inner_bounds_packed"]
        unpacked = np.unpackbits(packed)[: shape[0] * shape[1]].astype(bool)
        self.inner_bounds = unpacked.reshape(shape)

        self.cell_pixel_counts = npz_data["cell_pixel_counts"].astype(np.int32)
        self.max_cell_id = len(self.cell_pixel_counts) - 1
        self.cell_sizes = np.maximum(self.cell_pixel_counts[self.cell_ids], 1)

        self.cell_crit_p50 = npz_data["cell_crit_p50"].astype(np.float32)
        self.sorted_crit_p50 = np.sort(self.cell_crit_p50)

        # Precompute flat references
        self.seg_flat = self.seg.ravel()
        self.raw_flat = self.raw_pd1.ravel()

        # Cache normalized raw image
        raw_norm = normalize_image(self.raw_pd1, pmin=3.0, pmax=99.8)
        self.raw_uint8 = (raw_norm * 255.0).astype(np.uint8)

        # Cell ID lookup table
        self.id_to_index = np.full(self.max_cell_id + 1, -1, dtype=np.int32)
        self.id_to_index[self.cell_ids] = np.arange(self.total_cells, dtype=np.int32)

    def evaluate_threshold(self, abs_th: float, perc_pos_th: float = 0.5) -> Tuple[np.ndarray, int, float]:
        """Compute positive cells matching spatialproteomics (returns is_pos_mask, count, pct)."""
        if self.total_cells == 0:
            return np.zeros(0, dtype=bool), 0, 0.0

        if abs_th <= 0.0 and perc_pos_th <= 0.0:
            is_pos = np.ones(self.total_cells, dtype=bool)
            return is_pos, self.total_cells, 100.0

        # Fast vectorized path when perc_pos_th == 0.5
        if abs(perc_pos_th - 0.5) < 1e-4:
            is_pos = self.cell_crit_p50 > abs_th
            pos_count = int(np.sum(is_pos))
            pos_pct = (pos_count / max(self.total_cells, 1)) * 100.0
            return is_pos, pos_count, pos_pct

        # General path for arbitrary perc_pos_th
        pos_pixels_per_label = np.bincount(
            self.seg_flat,
            weights=(self.raw_flat > abs_th).astype(np.float32),
            minlength=self.max_cell_id + 1,
        )
        cell_pos_pixels = pos_pixels_per_label[self.cell_ids]
        perc_pos = cell_pos_pixels / self.cell_sizes
        is_pos = perc_pos >= perc_pos_th
        pos_count = int(np.sum(is_pos))
        pos_pct = (pos_count / max(self.total_cells, 1)) * 100.0
        return is_pos, pos_count, pos_pct

    def solve_threshold_for_target_pct(self, target_pct: float, perc_pos_th: float = 0.5) -> float:
        """Find discrete integer threshold minimizing difference with target clinical IHC %."""
        if self.total_cells == 0:
            return 0.0

        max_int = int(min(255, max(50, int(np.ceil(np.percentile(self.raw_pd1, 99.8))))))
        search_range = np.arange(0, max_int + 1, 1, dtype=np.float32)

        if abs(perc_pos_th - 0.5) < 1e-4:
            # Fast vectorized calculation
            idx = np.searchsorted(self.sorted_crit_p50, search_range, side="right")
            pcts = ((len(self.sorted_crit_p50) - idx) / len(self.sorted_crit_p50)) * 100.0
        else:
            pcts = np.array([self.evaluate_threshold(t, perc_pos_th)[2] for t in search_range], dtype=np.float32)

        diffs = np.abs(pcts - target_pct)
        best_idx = int(np.argmin(diffs))
        return float(search_range[best_idx])

    def render_panel(
        self,
        abs_th: Optional[float] = None,
        is_pos: Optional[np.ndarray] = None,
        color_hex: str = "#00FF66",
        alpha: float = 0.08,
        alpha_boundary: float = 1.0,
        show_neg: bool = False,
        neg_color_hex: str = "#E11D48",
        raw_only: bool = False,
    ) -> Image.Image:
        """Render panel with cell outlines using Pillow alpha blending."""
        if raw_only or abs_th is None or is_pos is None:
            gray = self.raw_uint8
            rgb = np.stack([gray, gray, gray], axis=-1)
            return Image.fromarray(rgb)

        filtered = np.maximum(0.0, self.raw_pd1 - abs_th)
        norm_filt = normalize_image(filtered, pmin=3.0, pmax=99.8)
        gray = (norm_filt * 255.0).astype(np.uint8)
        out_rgb = np.stack([gray, gray, gray], axis=-1)

        pos_label_mask = np.zeros(self.max_cell_id + 1, dtype=bool)
        pos_label_mask[self.cell_ids] = is_pos
        pos_pixel_mask = pos_label_mask[self.seg]

        r_pos, g_pos, b_pos = hex_to_rgb(color_hex)

        # Alpha-blend positive cell interior
        if alpha > 0.0:
            inv_a = 1.0 - alpha
            out_rgb[pos_pixel_mask, 0] = (out_rgb[pos_pixel_mask, 0] * inv_a + r_pos * alpha).astype(np.uint8)
            out_rgb[pos_pixel_mask, 1] = (out_rgb[pos_pixel_mask, 1] * inv_a + g_pos * alpha).astype(np.uint8)
            out_rgb[pos_pixel_mask, 2] = (out_rgb[pos_pixel_mask, 2] * inv_a + b_pos * alpha).astype(np.uint8)

        # Alpha-blend positive cell inner boundaries
        pos_bounds = self.inner_bounds & pos_pixel_mask
        if alpha_boundary >= 1.0:
            out_rgb[pos_bounds] = [r_pos, g_pos, b_pos]
        elif alpha_boundary > 0.0:
            inv_ab = 1.0 - alpha_boundary
            out_rgb[pos_bounds, 0] = (out_rgb[pos_bounds, 0] * inv_ab + r_pos * alpha_boundary).astype(np.uint8)
            out_rgb[pos_bounds, 1] = (out_rgb[pos_bounds, 1] * inv_ab + g_pos * alpha_boundary).astype(np.uint8)
            out_rgb[pos_bounds, 2] = (out_rgb[pos_bounds, 2] * inv_ab + b_pos * alpha_boundary).astype(np.uint8)

        # Optional negative cell boundary outline
        if show_neg:
            neg_bounds = self.inner_bounds & (self.seg > 0) & (~pos_pixel_mask)
            r_neg, g_neg, b_neg = hex_to_rgb(neg_color_hex)
            out_rgb[neg_bounds] = [r_neg, g_neg, b_neg]

        return Image.fromarray(out_rgb)


class S3DataManager:
    """Unified storage manager supporting AWS S3 and local prepared data fallback."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.s3_cfg = cfg.get("s3", {})

        if self.s3_cfg.get("use_s3") is False:
            self.bucket = ""
        else:
            secret_bucket = self._lookup_secret("bucket_name", "bucket", "s3_bucket", "aws_bucket")
            cfg_bucket = self.s3_cfg.get("bucket", "").strip()
            self.bucket = secret_bucket or cfg_bucket or "tnhl-pd1-tuner-data"
        self.prefix = (
            self._lookup_secret("s3_prefix", "prefix")
            or self.s3_cfg.get("prefix", "tnhl_pd1_tuner")
        )
        self.region = (
            self._lookup_secret("region_name", "region", "aws_region", "AWS_DEFAULT_REGION")
            or self.s3_cfg.get("region", "eu-central-1")
        )

        # Local directory fallback
        script_dir = Path(__file__).resolve().parent
        local_dir_cfg = self.s3_cfg.get("local_data_dir", "../aws_prepared_data")
        self.local_dir = (script_dir / local_dir_cfg).resolve() if not Path(local_dir_cfg).is_absolute() else Path(local_dir_cfg)

        self.use_s3 = bool(self.bucket)
        self.fs = None
        self.s3_init_error = None

        if self.use_s3:
            try:
                import s3fs
                ak = self._lookup_secret("aws_access_key_id", "access_key_id", "key_id", "AWS_ACCESS_KEY_ID")
                sk = self._lookup_secret("aws_secret_access_key", "secret_access_key", "secret_key", "AWS_SECRET_ACCESS_KEY")
                if ak and sk:
                    self.fs = s3fs.S3FileSystem(
                        key=ak,
                        secret=sk,
                        client_kwargs={"region_name": self.region},
                        default_cache_type="none",
                        use_listings_cache=False,
                    )
                else:
                    self.fs = s3fs.S3FileSystem(
                        client_kwargs={"region_name": self.region},
                        default_cache_type="none",
                        use_listings_cache=False,
                    )
            except Exception as e:
                self.s3_init_error = str(e)
                print(f"Warning: Failed to initialize s3fs ({e}), falling back to local files.")
                self.use_s3 = False

    def _lookup_secret(self, *keys: str) -> Optional[str]:
        """Fetch secret from st.secrets or os.environ by checking multiple alias keys."""
        for k in keys:
            v = os.environ.get(k) or os.environ.get(k.upper())
            if v:
                return str(v).strip()
            if st is not None:
                try:
                    if hasattr(st, "secrets"):
                        if "aws" in st.secrets and k in st.secrets["aws"]:
                            val = st.secrets["aws"].get(k)
                            if val:
                                return str(val).strip()
                        if k in st.secrets:
                            val = st.secrets.get(k)
                            if val:
                                return str(val).strip()
                except Exception:
                    pass
        return None

    def _get_secret(self, key: str) -> Optional[str]:
        """Backward compatibility helper for single key."""
        return self._lookup_secret(key)

    def load_raw_cohort_metadata(self) -> pd.DataFrame:
        """Load raw cohort metadata CSV without merged curations from S3 or local fallback."""
        s3_error = None
        if self.use_s3 and self.fs:
            s3_path = f"{self.bucket}/{self.prefix}/metadata/cohort_metadata.csv"
            try:
                with self.fs.open(s3_path, "rb") as f:
                    df = pd.read_csv(f)
                    df["core_id"] = df["core_id"].astype(str).str.strip()
                    return df
            except Exception as e:
                s3_error = str(e)
                print(f"Warning: S3 read failed ({e}), checking local directory.")
        elif self.s3_init_error:
            s3_error = f"s3fs initialization failed: {self.s3_init_error}"

        local_meta = self.local_dir / "metadata" / "cohort_metadata.csv"
        if local_meta.is_file():
            df = pd.read_csv(local_meta)
            df["core_id"] = df["core_id"].astype(str).str.strip()
            return df

        detail = f" (S3 Error: {s3_error})" if s3_error else ""
        raise FileNotFoundError(
            f"Cohort metadata not found. Checked s3://{self.bucket}/{self.prefix}/metadata/cohort_metadata.csv{detail} "
            f"and local path {local_meta}."
        )

    @staticmethod
    def merge_curations(df_cohort: pd.DataFrame, curations_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Merge saved curation decisions into cohort dataframe."""
        df = df_cohort.copy()
        df["core_id"] = df["core_id"].astype(str).str.strip()

        # Clean string keys in curations_dict
        clean_cur = {str(k).strip(): v for k, v in curations_dict.items()}

        decisions = []
        cur_abs_list = []
        cur_notes_list = []

        for _, row in df.iterrows():
            cid = str(row["core_id"]).strip()
            if cid in clean_cur:
                cinfo = clean_cur[cid]
                dec_val = str(cinfo.get("decision_type", "Reviewed")).strip()
                decisions.append(dec_val if dec_val else "Reviewed")
                cur_abs_list.append(cinfo.get("final_abs_th"))
                cur_notes_list.append(str(cinfo.get("notes", "")).strip())
            else:
                decisions.append("Unreviewed")
                cur_abs_list.append(None)
                cur_notes_list.append("")

        df["cur_decision"] = decisions
        df["cur_abs_th"] = cur_abs_list
        df["cur_notes"] = cur_notes_list
        return df

    def _merge_existing_curations(self, df_cohort: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible helper for merging curations."""
        curations_dict = self.load_curations_dict()
        return self.merge_curations(df_cohort, curations_dict)

    def load_cohort_metadata(self) -> pd.DataFrame:
        """Load cohort metadata CSV with merged curations."""
        df_raw = self.load_raw_cohort_metadata()
        curations_dict = self.load_curations_dict()
        return self.merge_curations(df_raw, curations_dict)

    def load_core_bundle(self, core_id: str) -> CoreBundle:
        """Load and unpack .npz core bundle from S3 or local directory."""
        # Check local cache first for instant load
        local_npz = self.local_dir / "cores" / f"{core_id}_pd1.npz"
        if local_npz.is_file():
            npz = np.load(local_npz)
            return CoreBundle(core_id, npz)

        # Stream from S3
        if self.use_s3 and self.fs:
            s3_path = f"{self.bucket}/{self.prefix}/cores/{core_id}_pd1.npz"
            with self.fs.open(s3_path, "rb") as f:
                bio = io.BytesIO(f.read())
                npz = np.load(bio)
                return CoreBundle(core_id, npz)

        raise FileNotFoundError(f"Core bundle '{core_id}' not found locally or on S3.")

    def load_curations_dict(self) -> Dict[str, Dict[str, Any]]:
        """Load all curated decisions into a dictionary keyed by core_id."""
        if self.use_s3 and self.fs:
            try:
                self.fs.invalidate_cache()
            except Exception:
                pass
            s3_path = f"{self.bucket}/{self.prefix}/curations/curated_pd1_thresholds.csv"
            try:
                with self.fs.open(s3_path, "rb") as f:
                    df = pd.read_csv(f)
                if "core_id" in df.columns and len(df) > 0:
                    df["core_id"] = df["core_id"].astype(str).str.strip()
                    if "reviewer" in df.columns:
                        df = df.drop(columns=["reviewer"])
                    return {str(k).strip(): v for k, v in df.set_index("core_id").to_dict(orient="index").items()}
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Warning: Failed to read S3 curations ({e})")

        local_cur = self.local_dir / "curations" / "curated_pd1_thresholds.csv"
        if local_cur.is_file():
            try:
                df = pd.read_csv(local_cur)
                if "core_id" in df.columns and len(df) > 0:
                    df["core_id"] = df["core_id"].astype(str).str.strip()
                    if "reviewer" in df.columns:
                        df = df.drop(columns=["reviewer"])
                    return {str(k).strip(): v for k, v in df.set_index("core_id").to_dict(orient="index").items()}
            except Exception as e:
                print(f"Warning: Failed to read local curations ({e})")

        return {}

    def save_curation_decision(
        self,
        core_id: str,
        decision_type: str,
        abs_th: Optional[float],
        perc_pos_th: float,
        notes: str,
    ) -> str:
        """Save curation decision directly to S3 or local CSV without reviewer column."""
        cid_str = str(core_id).strip()
        record = {
            "core_id": cid_str,
            "decision_type": str(decision_type).strip(),
            "final_abs_th": float(abs_th) if abs_th is not None else np.nan,
            "final_perc_pos_th": float(perc_pos_th),
            "notes": str(notes).strip(),
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 1. Update on S3 if active
        if self.use_s3 and self.fs:
            s3_path = f"{self.bucket}/{self.prefix}/curations/curated_pd1_thresholds.csv"
            try:
                try:
                    self.fs.invalidate_cache()
                except Exception:
                    pass

                try:
                    with self.fs.open(s3_path, "rb") as f:
                        df = pd.read_csv(f)
                    if "reviewer" in df.columns:
                        df = df.drop(columns=["reviewer"])
                    df["core_id"] = df["core_id"].astype(str).str.strip()
                    df = df[df["core_id"] != cid_str]
                    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                except (FileNotFoundError, Exception):
                    df = pd.DataFrame([record])

                if "reviewer" in df.columns:
                    df = df.drop(columns=["reviewer"])

                df["core_id"] = df["core_id"].astype(str).str.strip()
                df = df.sort_values("core_id", key=lambda col: [natural_sort_key(x) for x in col]).reset_index(drop=True)
                with self.fs.open(s3_path, "wb") as f:
                    df.to_csv(f, index=False)

                try:
                    self.fs.invalidate_cache()
                except Exception:
                    pass
                return f"s3://{s3_path}"
            except Exception as e:
                print(f"Warning: S3 save failed ({e}), saving locally.")

        # 2. Local fallback save
        local_cur_dir = self.local_dir / "curations"
        local_cur_dir.mkdir(parents=True, exist_ok=True)
        local_csv = local_cur_dir / "curated_pd1_thresholds.csv"

        if local_csv.is_file():
            try:
                df = pd.read_csv(local_csv)
                if "reviewer" in df.columns:
                    df = df.drop(columns=["reviewer"])
                df["core_id"] = df["core_id"].astype(str).str.strip()
                df = df[df["core_id"] != cid_str]
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            except Exception:
                df = pd.DataFrame([record])
        else:
            df = pd.DataFrame([record])

        if "reviewer" in df.columns:
            df = df.drop(columns=["reviewer"])

        df["core_id"] = df["core_id"].astype(str).str.strip()
        df = df.sort_values("core_id", key=lambda col: [natural_sort_key(x) for x in col]).reset_index(drop=True)
        df.to_csv(local_csv, index=False)
        return str(local_csv)

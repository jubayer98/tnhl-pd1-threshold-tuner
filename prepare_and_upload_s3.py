#!/usr/bin/env python3
"""PD-1 Data Packaging & AWS S3 Upload Script.

Extracts only the PD-1 marker data, segmentation masks, and precomputed inner
cell boundaries for all 198 study cores (filtered by is_minor_cts_prediction == True)
into compact ~3 MB .npz bundles (~600 MB total for all 198 cores) and uploads
them to an AWS S3 bucket for cloud hosting.

Usage:
    # 1. Package a test core locally:
    python prepare_and_upload_s3.py --cores 1-4

    # 2. Package all 198 cores locally:
    python prepare_and_upload_s3.py

    # 3. Package and upload to S3:
    python prepare_and_upload_s3.py --upload --bucket my-codex-bucket
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries
import xarray as xr
import yaml


def natural_sort_key(s: str) -> list[int | str]:
    """Sort strings containing numbers in natural human order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def resolve_path(p: str | Path) -> Path:
    """Resolve relative or absolute path."""
    return Path(p).expanduser().resolve()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_ihc_value(val: Any) -> Tuple[Optional[float], str, bool]:
    """Parse pathology IHC clinical value."""
    if pd.isna(val) or val is None:
        return None, "Not Available", False
    s = str(val).strip()
    if not s or s.lower() in ["nan", "none", "na", "n/a", "unknown"]:
        return None, "Not Available", False
    if "<" in s:
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if m:
            v = float(m.group(1))
            return max(0.0, v - 0.5), f"<{v:g}%", True
        return None, s, False
    if ">" in s:
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if m:
            v = float(m.group(1))
            return min(100.0, v + 0.5), f">{v:g}%", True
        return None, s, False
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        v = float(m.group(1))
        return v, f"{v:g}%", True
    return None, s, False


def find_zarr_stores(zarr_base: Path) -> Dict[str, Path]:
    """Index all available core Zarr stores."""
    zarr_map: Dict[str, Path] = {}
    if not zarr_base.is_dir():
        return zarr_map
    for p in zarr_base.rglob("*.zarr"):
        core_name = p.stem.replace(".zarr", "").strip()
        zarr_map[core_name] = p
    return zarr_map


def find_our_threshold_csv(base_dir: Path, core_name: str) -> Optional[Path]:
    """Find UKD categorical threshold CSV for a core."""
    if not base_dir.is_dir():
        return None
    for p in base_dir.rglob("*.csv"):
        if p.stem.strip() == core_name:
            return p
    return None


def index_cohort_metadata(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Index the cohort metadata for all 198 cores where is_minor_cts_prediction == 1.0."""
    local_src = cfg.get("local_source", {})
    qc_path = resolve_path(local_src["qc_table"])
    df_qc = pd.read_excel(qc_path, sheet_name=local_src.get("qc_sheet", "main"), header=local_src.get("qc_header_row", 1))

    # Match IHC column name
    col_n = [c for c in df_qc.columns if "col_n" in str(c).lower() or "pd1_ihc" in str(c).lower()]
    col_n_name = col_n[0] if col_n else "PD1_IHC"

    # Keisuke thresholds
    kei_path = resolve_path(local_src["keisuke_thresholds"])
    df_kei = pd.read_csv(kei_path)
    df_kei["sample"] = df_kei["sample"].astype(str).str.strip()
    df_kei_pd1 = df_kei[df_kei["channel"] == "PD1"].copy()
    kei_dict = df_kei_pd1.set_index("sample").to_dict(orient="index")

    # Zarr stores
    zarr_base = resolve_path(local_src["zarr_dir"])
    zarr_map = find_zarr_stores(zarr_base)

    # UKD thresholds
    our_th_base = resolve_path(local_src["our_thresholds_dir"])

    records = []
    for _, row in df_qc.iterrows():
        c_name = str(row["core_name"]).strip()
        is_minor = row.get("is_minor_cts_prediction")
        is_minor_bool = bool(
            is_minor is True or is_minor == 1 or is_minor == 1.0 or str(is_minor).strip().lower() in ["true", "1"]
        )
        if not is_minor_bool:
            continue
        if c_name not in zarr_map:
            continue

        raw_ihc = row.get(col_n_name, row.get("PD1_IHC"))
        num_ihc, disp_ihc, is_valid_ihc = parse_ihc_value(raw_ihc)

        # Tneo neoplastic phenotype and PD-1 ratio
        tneo_def = str(row.get("tneo_def", "")).strip() if pd.notna(row.get("tneo_def")) else "N/A"
        tneo_markers = str(row.get("tneo_markers", "")).strip() if pd.notna(row.get("tneo_markers")) else "N/A"
        raw_tneo_ratio = row.get("T Neo PD1 ratio_4th")
        try:
            tneo_ratio = float(raw_tneo_ratio) if pd.notna(raw_tneo_ratio) else None
        except (ValueError, TypeError):
            tneo_ratio = None
        tneo_str = f"{tneo_ratio * 100:.1f}%" if tneo_ratio is not None else "N/A"

        k_info = kei_dict.get(c_name, {})
        k_abs = k_info.get("lower")
        k_perc = k_info.get("threshold", 0.5)

        our_csv = find_our_threshold_csv(our_th_base, c_name)
        o_abs, o_perc = None, 0.5
        if our_csv and our_csv.is_file():
            try:
                df_o = pd.read_csv(our_csv)
                if "channel" in df_o.columns and "PD1" in df_o["channel"].values:
                    row_pd1 = df_o[df_o["channel"] == "PD1"].iloc[0]
                    o_abs = row_pd1.get("absolute_threshold")
                    o_perc = row_pd1.get("perc_pos_threshold", 0.5)
            except Exception:
                pass

        records.append({
            "core_id": c_name,
            "core_name": c_name,
            "diagnosis": str(row.get("diagnosis", "Unknown")).strip(),
            "subtype": str(row.get("subtype", "Unknown")).strip(),
            "ihc_val": num_ihc,
            "ihc_str": disp_ihc,
            "is_valid_ihc": is_valid_ihc,
            "tneo_def": tneo_def,
            "tneo_markers": tneo_markers,
            "tneo_pd1_ratio": tneo_ratio,
            "tneo_pd1_str": tneo_str,
            "zarr_path": str(zarr_map[c_name]),
            "our_abs_th": float(o_abs) if pd.notna(o_abs) else None,
            "our_perc_pos_th": float(o_perc) if pd.notna(o_perc) else 0.5,
            "kei_abs_th": float(k_abs) if pd.notna(k_abs) else None,
            "kei_perc_pos_th": float(k_perc) if pd.notna(k_perc) else 0.5,
        })

    df_meta = pd.DataFrame(records)
    df_meta = df_meta.sort_values("core_id", key=lambda col: [natural_sort_key(x) for x in col]).reset_index(drop=True)
    return df_meta


def package_core(core_id: str, zarr_path: Path, output_file: Path) -> Dict[str, Any]:
    """Open Zarr store, extract only PD-1 channel + mask, precompute boundaries, and save bundle."""
    t0 = time.time()
    ds = xr.open_zarr(str(zarr_path), consolidated=False)

    raw_channels = list(ds.coords["channels"].values)
    if "PD1" in raw_channels:
        pd1_layer = ds["_raw_image"].sel(channels="PD1").values
    else:
        candidates = [c for c in raw_channels if str(c).strip().lower() == "pd1"]
        if not candidates:
            ds.close()
            raise ValueError(f"Channel 'PD1' not found in {zarr_path}")
        pd1_layer = ds["_raw_image"].sel(channels=candidates[0]).values

    raw_pd1 = pd1_layer.astype(np.float32)
    seg = ds["_segmentation"].values.astype(np.int32)
    height, width = raw_pd1.shape

    if "cells" in ds.coords:
        cell_ids = ds.coords["cells"].values.astype(np.int64)
    else:
        unique_labels = np.unique(seg)
        cell_ids = unique_labels[unique_labels > 0].astype(np.int64)

    total_cells = len(cell_ids)
    max_cell_id = int(np.max(seg)) if seg.size > 0 else 0

    # Precompute inner boundaries (mode='inner' on segmentation mask)
    inner_bounds = find_boundaries(seg, mode="inner")
    inner_bounds_packed = np.packbits(inner_bounds)

    # Pixel count per cell
    seg_flat = seg.ravel()
    raw_flat = raw_pd1.ravel()
    cell_pixel_counts = np.bincount(seg_flat, minlength=max_cell_id + 1).astype(np.int32)

    # Precompute p50 critical cutoffs for instant response curve & IHC matching
    crit_vals = np.zeros(total_cells, dtype=np.float32)
    if total_cells > 0:
        order = np.argsort(seg_flat)
        sorted_labels = seg_flat[order]
        sorted_raw = raw_flat[order]
        starts = np.searchsorted(sorted_labels, cell_ids, side="left")
        ends = np.searchsorted(sorted_labels, cell_ids, side="right")

        for idx in range(total_cells):
            s = starts[idx]
            e = ends[idx]
            n_px = e - s
            if n_px <= 0:
                crit_vals[idx] = 0.0
                continue
            k = int(np.ceil(0.5 * n_px))
            k = max(1, min(k, n_px))
            target_idx = e - k
            cell_vals = np.partition(sorted_raw[s:e], target_idx - s)
            crit_vals[idx] = cell_vals[target_idx - s]

    ds.close()

    # Save compressed bundle
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        raw_pd1=raw_pd1,
        seg=seg,
        cell_ids=cell_ids,
        inner_bounds_packed=inner_bounds_packed,
        shape=np.array([height, width], dtype=np.int32),
        cell_pixel_counts=cell_pixel_counts,
        cell_crit_p50=crit_vals,
    )

    elapsed = time.time() - t0
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    return {
        "core_id": core_id,
        "cells": total_cells,
        "shape": f"{height}x{width}",
        "size_mb": round(file_size_mb, 2),
        "time_sec": round(elapsed, 2),
    }


def upload_to_s3(
    local_dir: Path,
    bucket: str,
    prefix: str,
    key_id: Optional[str] = None,
    secret_key: Optional[str] = None,
    region: Optional[str] = None,
):
    """Upload prepared cores and metadata to AWS S3 using s3fs."""
    try:
        import s3fs
    except ImportError:
        print("ERROR: 's3fs' is required for S3 upload. Install with: pip install s3fs")
        sys.exit(1)

    print(f"\n--- Uploading to AWS S3: s3://{bucket}/{prefix}/ ---")

    # Resolve credentials from arguments, environment variables, or .streamlit/secrets.toml
    key_id = key_id or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = region or os.environ.get("AWS_DEFAULT_REGION") or "eu-central-1"

    if not key_id or not secret_key:
        secrets_file = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
        if secrets_file.is_file():
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    tomllib = None
            if tomllib is not None:
                try:
                    with open(secrets_file, "rb") as f:
                        secrets_data = tomllib.load(f)
                    aws_sec = secrets_data.get("aws", {})
                    key_id = key_id or aws_sec.get("aws_access_key_id")
                    secret_key = secret_key or aws_sec.get("aws_secret_access_key")
                    region = region or aws_sec.get("region_name", "eu-central-1")
                except Exception:
                    pass

    try:
        if key_id and secret_key:
            fs = s3fs.S3FileSystem(key=key_id, secret=secret_key, client_kwargs={"region_name": region})
        else:
            fs = s3fs.S3FileSystem(client_kwargs={"region_name": region})

        # Test listing bucket to verify authentication and bucket existence early
        fs.ls(bucket)
    except Exception as e:
        err_msg = str(e)
        print("\n" + "!" * 70)
        print("AWS S3 AUTHENTICATION ERROR:")
        print(f"  {err_msg}")
        print("!" * 70)
        print("\nTo fix this, please provide your AWS credentials using ONE of the following options:\n")
        print("Option 1: Export environment variables in your terminal:")
        print("  export AWS_ACCESS_KEY_ID='your-access-key-id'")
        print("  export AWS_SECRET_ACCESS_KEY='your-secret-access-key'")
        print("  export AWS_DEFAULT_REGION='eu-central-1'")
        print(f"  python prepare_and_upload_s3.py --upload --bucket {bucket}\n")
        print("Option 2: Pass them directly via CLI flags:")
        print(f"  python prepare_and_upload_s3.py --upload --bucket {bucket} \\")
        print("    --key-id 'your-access-key-id' --secret-key 'your-secret-access-key' --region 'eu-central-1'\n")
        print("Option 3: Save them in .streamlit/secrets.toml:")
        print("  [aws]")
        print("  bucket_name = \"" + bucket + "\"")
        print("  aws_access_key_id = \"your-access-key-id\"")
        print("  aws_secret_access_key = \"your-secret-access-key\"")
        print("  region_name = \"eu-central-1\"")
        print("!" * 70 + "\n")
        sys.exit(1)

    # Upload metadata
    meta_file = local_dir / "metadata" / "cohort_metadata.csv"
    if meta_file.is_file():
        s3_meta_path = f"{bucket}/{prefix}/metadata/cohort_metadata.csv"
        print(f"Uploading metadata: {s3_meta_path}")
        fs.put(str(meta_file), s3_meta_path)

    # Upload core bundles
    core_files = list((local_dir / "cores").glob("*_pd1.npz"))
    total = len(core_files)
    print(f"Uploading {total} core bundles...")

    for i, cf in enumerate(core_files, 1):
        s3_core_path = f"{bucket}/{prefix}/cores/{cf.name}"
        fs.put(str(cf), s3_core_path)
        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] Uploaded {cf.name} -> s3://{s3_core_path}")

    print(f"\nUpload complete! All {total} cores uploaded to s3://{bucket}/{prefix}/")


def main():
    parser = argparse.ArgumentParser(description="Package PD-1 Zarr data into lightweight bundles and upload to AWS S3.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default="../aws_prepared_data", help="Local directory to store prepared bundles")
    parser.add_argument("--cores", nargs="+", help="Specific core IDs to process (e.g. --cores 1-4 1-1)")
    parser.add_argument("--limit", type=int, help="Limit number of cores to process")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing bundles")
    parser.add_argument("--upload", action="store_true", help="Upload packaged data to AWS S3 after processing")
    parser.add_argument("--bucket", type=str, help="AWS S3 bucket name (overrides config)")
    parser.add_argument("--prefix", type=str, default="tnhl_pd1_tuner", help="S3 prefix folder")
    parser.add_argument("--key-id", type=str, help="AWS Access Key ID")
    parser.add_argument("--secret-key", type=str, help="AWS Secret Access Key")
    parser.add_argument("--region", type=str, default="eu-central-1", help="AWS Region (default: eu-central-1)")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    cfg_path = (script_dir / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)

    output_dir = (script_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    cores_dir = output_dir / "cores"
    meta_dir = output_dir / "metadata"
    cores_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PD-1 Lightweight Data Packaging for AWS S3")
    print(f"Output Directory: {output_dir}")
    print("=" * 70)

    # 1. Index cohort metadata (198 cores)
    print("\n[1/3] Indexing cohort metadata (is_minor_cts_prediction == 1.0)...")
    meta_csv_path = meta_dir / "cohort_metadata.csv"
    if meta_csv_path.is_file() and not args.force:
        df_meta = pd.read_csv(meta_csv_path)
        print(f"Loaded existing cohort index: {meta_csv_path} ({len(df_meta)} cores)")
    else:
        df_meta = index_cohort_metadata(cfg)
        print(f"Found {len(df_meta)} eligible cores.")
        df_meta.to_csv(meta_csv_path, index=False)
        print(f"Saved cohort index: {meta_csv_path} ({len(df_meta)} cores)")

    # 2. Filter cores if requested
    if args.cores:
        df_meta = df_meta[df_meta["core_id"].isin(args.cores)]
        print(f"Filtered to requested {len(df_meta)} cores: {args.cores}")
    elif args.limit:
        df_meta = df_meta.head(args.limit)
        print(f"Limited to first {len(df_meta)} cores.")

    # 3. Package each core
    print(f"\n[2/3] Packaging {len(df_meta)} cores into compressed .npz bundles...")
    total_size = 0.0
    processed = 0

    for idx, row in df_meta.iterrows():
        cid = row["core_id"]
        zpath = Path(row["zarr_path"])
        out_npz = cores_dir / f"{cid}_pd1.npz"

        if out_npz.is_file() and not args.force:
            sz = out_npz.stat().st_size / (1024 * 1024)
            total_size += sz
            processed += 1
            print(f"  [{processed}/{len(df_meta)}] Core {cid}: Already exists ({sz:.2f} MB), skipping.")
            continue

        try:
            stats = package_core(cid, zpath, out_npz)
            total_size += stats["size_mb"]
            processed += 1
            print(f"  [{processed}/{len(df_meta)}] Core {cid}: {stats['cells']:,} cells, {stats['shape']}, {stats['size_mb']} MB in {stats['time_sec']}s")
        except Exception as e:
            print(f"  ERROR packaging Core {cid}: {e}", file=sys.stderr)

    print("\n[3/3] Packaging Complete!")
    print(f"  Total Processed Cores : {processed}")
    print(f"  Total Bundle Size     : {total_size:.2f} MB (~{total_size/1024:.2f} GB)")
    print(f"  Average Size / Core   : {total_size/max(processed, 1):.2f} MB")

    # 4. Upload to S3 if requested
    if args.upload:
        bucket = args.bucket or cfg.get("s3", {}).get("bucket")
        if not bucket:
            print("ERROR: --upload requested but no S3 bucket specified. Provide --bucket <name> or set in config.yaml.")
            sys.exit(1)
        prefix = args.prefix or cfg.get("s3", {}).get("prefix", "tnhl_pd1_tuner")
        upload_to_s3(
            local_dir=output_dir,
            bucket=bucket,
            prefix=prefix,
            key_id=args.key_id,
            secret_key=args.secret_key,
            region=args.region,
        )
    else:
        print("\nTo upload to AWS S3, run:")
        print(f"  python prepare_and_upload_s3.py --upload --bucket <your-s3-bucket> --prefix {args.prefix}")


if __name__ == "__main__":
    main()

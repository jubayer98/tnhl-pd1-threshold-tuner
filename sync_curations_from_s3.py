#!/usr/bin/env python3
"""Sync Curation Decisions from AWS S3 to Local Mac & Export Step 4 CSVs.

Usage:
    # 1. Download curated CSV from S3 to local outputs:
    python sync_curations_from_s3.py --bucket my-codex-bucket

    # 2. Download and export updated Step 4 Snakemake CSVs:
    python sync_curations_from_s3.py --bucket my-codex-bucket --export-step4-dir ../outputs/step4_curated_csvs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_our_threshold_csv(base_dir: Path, core_name: str) -> Optional[Path]:
    if not base_dir.is_dir():
        return None
    for p in base_dir.rglob("*.csv"):
        if p.stem.strip() == core_name:
            return p
    return None


def export_step4_csv(
    core_id: str,
    abs_th: float,
    perc_pos_th: float,
    step4_dir: Path,
    our_th_base: Path,
) -> Path:
    """Export or update the core's Step 4 categorical threshold CSV."""
    tma = core_id.split("-")[0] if "-" in core_id else ""
    tma_dir = step4_dir / tma
    tma_dir.mkdir(parents=True, exist_ok=True)
    target_csv = tma_dir / f"{core_id}.csv"

    orig_csv = find_our_threshold_csv(our_th_base, core_id)
    if orig_csv and orig_csv.is_file():
        df = pd.read_csv(orig_csv)
        if "channel" in df.columns:
            if "PD1" in df["channel"].values:
                df.loc[df["channel"] == "PD1", "absolute_threshold"] = abs_th
                df.loc[df["channel"] == "PD1", "perc_pos_threshold"] = perc_pos_th
                df.loc[df["channel"] == "PD1", "status"] = "curated_custom"
            else:
                new_row = {
                    "channel": "PD1",
                    "sample": core_id,
                    "status": "curated_custom",
                    "absolute_threshold": abs_th,
                    "perc_pos_threshold": perc_pos_th,
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(target_csv, index=False)
    else:
        df = pd.DataFrame([
            {
                "channel": "PD1",
                "sample": core_id,
                "status": "curated_custom",
                "absolute_threshold": abs_th,
                "perc_pos_threshold": perc_pos_th,
            }
        ])
        df.to_csv(target_csv, index=False)

    return target_csv


def main():
    parser = argparse.ArgumentParser(description="Download reviewer curations from AWS S3 and export Step 4 CSVs.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--bucket", type=str, help="AWS S3 bucket name")
    parser.add_argument("--prefix", type=str, default="tnhl_pd1_tuner", help="S3 prefix folder")
    parser.add_argument("--key-id", type=str, help="AWS Access Key ID")
    parser.add_argument("--secret-key", type=str, help="AWS Secret Access Key")
    parser.add_argument("--region", type=str, default="eu-central-1", help="AWS Region (default: eu-central-1)")
    parser.add_argument("--out-csv", type=str, default="outputs/curated_pd1_thresholds.csv", help="Local destination CSV")
    parser.add_argument("--export-step4-dir", type=str, help="Directory to export Step 4 Snakemake CSVs")

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    cfg = load_config(script_dir / args.config)

    bucket = args.bucket or cfg.get("s3", {}).get("bucket")
    prefix = args.prefix or cfg.get("s3", {}).get("prefix", "tnhl_pd1_tuner")
    key_id = args.key_id or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = args.secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = args.region or os.environ.get("AWS_DEFAULT_REGION") or "eu-central-1"

    if not key_id or not secret_key:
        secrets_file = script_dir / ".streamlit" / "secrets.toml"
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

    out_csv = (script_dir / args.out_csv).resolve() if not Path(args.out_csv).is_absolute() else Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_cur = None

    if bucket:
        try:
            import s3fs
            if key_id and secret_key:
                fs = s3fs.S3FileSystem(key=key_id, secret=secret_key, client_kwargs={"region_name": region})
            else:
                fs = s3fs.S3FileSystem(client_kwargs={"region_name": region})
            s3_path = f"{bucket}/{prefix}/curations/curated_pd1_thresholds.csv"
            print(f"Fetching from AWS S3: s3://{s3_path}")
            if fs.exists(s3_path):
                with fs.open(s3_path, "rb") as f:
                    df_cur = pd.read_csv(f)
                if "reviewer" in df_cur.columns:
                    df_cur = df_cur.drop(columns=["reviewer"])
                df_cur.to_csv(out_csv, index=False)
                print(f"Downloaded {len(df_cur)} curation decisions to: {out_csv}")
            else:
                print(f"No curation table found at s3://{s3_path}")
        except Exception as e:
            print(f"Error fetching from S3: {e}")

    # Fallback to local file if S3 was not used or failed
    if df_cur is None:
        local_src = (script_dir / cfg.get("s3", {}).get("local_data_dir", "../aws_prepared_data") / "curations" / "curated_pd1_thresholds.csv").resolve()
        if local_src.is_file():
            df_cur = pd.read_csv(local_src)
            if "reviewer" in df_cur.columns:
                df_cur = df_cur.drop(columns=["reviewer"])
            df_cur.to_csv(out_csv, index=False)
            print(f"Copied {len(df_cur)} local curation decisions to: {out_csv}")
        else:
            print("No local curations found.")
            sys.exit(0)

    # Export Step 4 CSVs if requested
    if args.export_step4_dir:
        step4_dir = Path(args.export_step4_dir).resolve()
        step4_dir.mkdir(parents=True, exist_ok=True)
        our_th_base = Path(cfg.get("local_source", {}).get("our_thresholds_dir", "")).resolve()

        print(f"\nExporting Step 4 Snakemake CSVs to: {step4_dir}")
        exported = 0
        for _, row in df_cur.iterrows():
            cid = str(row["core_id"])
            abs_th = row.get("final_abs_th")
            perc_pos = row.get("final_perc_pos_th", 0.5)
            if pd.notna(abs_th) and float(abs_th) > 0.0:
                export_step4_csv(cid, float(abs_th), float(perc_pos), step4_dir, our_th_base)
                exported += 1

        print(f"Successfully exported {exported} Step 4 CSVs.")


if __name__ == "__main__":
    main()

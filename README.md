# Cloud PD-1 Interactive Threshold Tuner (AWS S3 + Streamlit Cloud)

A 24/7 online pathology review application for PD-1 single-cell threshold tuning, backed by **AWS S3** and hosted on **Streamlit Community Cloud** (or Hugging Face Spaces) with a **permanent, fixed HTTPS URL**.

---

## Key Capabilities

1. **Lightweight S3 Cloud Storage**:
   - Extracts only the PD-1 channel, cell segmentation mask, and precomputed inner cell boundaries for all **198 study cores** (`is_minor_cts_prediction == 1.0`).
   - Compresses the raw 50–100+ GB Zarr dataset into ~3–5 MB `.npz` bundles (**~600 MB total** for all 198 cores), costing <$0.02 / month on AWS S3.
2. **Permanent Fixed Public URL**:
   - Hosted on Streamlit Community Cloud (`https://<your-app>.streamlit.app`) with automatic SSL, zero VPN, and no local hosting requirements.
3. **Synchronized 4-Panel Inspection**:
   - Side-by-side comparison (Raw, UKD, Keisuke, Custom) in natural 1:1 circular aspect ratio.
   - Synchronized zoom & pan (`🔎 Box Zoom`, `✥ Pan`, `↺ Reset View`).
   - High-resolution `⛶ Fullscreen` modal dialogs with instant PNG download.
4. **Streamlined Threshold Tuning**:
   - Continuous `0` to `255` slider with discrete step 1.
   - 1-click presets: `Use UKD`, `Use Keisuke`, `Use Midpoint`, `Auto-Match Clinical IHC %`.
   - Vectorized Threshold vs. Positivity Response Curve (<1 ms).
   - Structured metadata banner (`CORE ID`, `DIAGNOSIS`, `TOTAL CELLS`).
   - Light and Dark browser theme adaptation.
5. **Direct S3 Decision Sync**:
   - Saves reviewer decisions directly to `s3://<bucket>/tnhl_pd1_tuner/curations/curated_pd1_thresholds.csv` without a `reviewer` column.
   - Local sync script pulls decisions back to your Mac and exports Step 4 Snakemake CSVs.

---

## Directory Structure

```text
aws_threshold_tuner/
├── app.py                      # Main Streamlit web application
├── s3_threshold_engine.py      # S3 streaming, vector evaluation & rendering engine
├── prepare_and_upload_s3.py    # Local Zarr packaging & S3 upload script
├── sync_curations_from_s3.py   # Pulls S3 curations & exports Step 4 CSVs
├── config.yaml                 # Configuration (S3 bucket, prefixes, colors)
├── requirements.txt            # Lightweight cloud dependencies
├── .streamlit/
│   └── secrets.toml.example    # Template for AWS S3 credentials
└── tests/
    └── test_s3_threshold_engine.py # Automated unit tests
```

---

## Step-by-Step Deployment Guide

### Step 1: Package and Upload PD-1 Data to AWS S3 (Run locally on your Mac)

```bash
cd /Users/jubayer/Development/tnhl-codex/scripts/interactive_threshold_tuner/aws_threshold_tuner

# 1. Package all 198 study cores locally into aws_prepared_data/ (~600 MB total):
python prepare_and_upload_s3.py

# 2. Upload the prepared bundles and metadata to your AWS S3 bucket:
python prepare_and_upload_s3.py --upload --bucket <your-s3-bucket-name>
```

### Step 2: Push to GitHub

```bash
git add .
git commit -m "Add AWS S3 PD-1 threshold tuner"
git push origin main
```

### Step 3: Deploy on Streamlit Community Cloud (Fixed URL)

1. Open [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and set the main file path to:
   `scripts/interactive_threshold_tuner/aws_threshold_tuner/app.py`
4. Under **"App URL"**, enter your desired permanent fixed URL:
   `https://tnhl-pd1-tuner.streamlit.app`
5. Open **"Advanced settings"** $\rightarrow$ **"Secrets"**, and paste your AWS credentials:
   ```toml
   [aws]
   aws_access_key_id = "AKIA..."
   aws_secret_access_key = "..."
   region_name = "eu-central-1"
   s3_bucket = "<your-s3-bucket-name>"
   s3_prefix = "tnhl_pd1_tuner"
   ```
6. Click **"Deploy!"**. Within 60 seconds, your app will be online 24/7.

### Step 4: Share with Pathologists / Reviewers

Send the permanent URL `https://tnhl-pd1-tuner.streamlit.app` to Andreas or other collaborators. They can review cores on any computer or tablet without VPN or software installation.

### Step 5: Pull Reviewed Decisions Back to Local Pipeline

Whenever you want to ingest the approved cutoffs back into the Snakemake workflow on your Mac:

```bash
cd /Users/jubayer/Development/tnhl-codex/scripts/interactive_threshold_tuner/aws_threshold_tuner
python sync_curations_from_s3.py --bucket <your-s3-bucket-name> --export-step4-dir ../outputs/step4_curated_csvs
```

---

## Running Automated Tests

```bash
cd /Users/jubayer/Development/tnhl-codex/scripts/interactive_threshold_tuner/aws_threshold_tuner
/opt/homebrew/Caskroom/miniforge/base/envs/tnhl-analysis/bin/python -m unittest discover -s tests -v
```

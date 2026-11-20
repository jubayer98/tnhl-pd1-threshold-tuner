#!/usr/bin/env python3
"""Cloud-Native PD-1 Single-Cell Threshold Tuner (Streamlit Web Application).

24/7 Cloud-Hosted Application backed by AWS S3 (or local prepared data cache):
- Streams lightweight ~3-5 MB PD-1 bundles on-demand in ~250ms
- Vectorized single-cell thresholding and discrete IHC matching in <1ms
- Synchronized 4-panel comparison in natural 1:1 circular aspect ratios (Raw, UKD, Keisuke, Custom)
- Smooth, lag-free zoom & pan across all panels + fullscreen modal inspection
- Continuous 0-255 cutoff slider with discrete step 1
- Saves decisions directly to AWS S3 (no reviewer column)
- Automatically adapts to Light and Dark browser themes
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

from s3_threshold_engine import (
    CoreBundle,
    S3DataManager,
    natural_sort_key,
    pil_to_data_uri,
    resolve_path,
)

# ------------------------------------------------------------------------------
# Page Setup & Styling (Adaptive to Light & Dark Theme)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="PD-1 Interactive Threshold Tuner (Cloud)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Generous top padding so the core header text is never clipped by Streamlit header */
    .block-container {
        padding-top: 4.2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Adaptive Core Stats Banner */
    .core-stats-banner {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: stretch;
        margin-top: 0 !important;
        margin-bottom: 14px !important;
        padding: 8px 12px;
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        border-radius: 8px;
    }
    .stat-chip {
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: var(--background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.22) !important;
        border-radius: 6px;
        padding: 8px 18px;
        min-width: 150px;
    }
    .stat-chip-core {
        border-left: 3.5px solid #00B4D8 !important;
    }
    .stat-chip-diag {
        border-left: 3.5px solid #8B5CF6 !important;
        flex-grow: 1;
    }
    .stat-chip-cells {
        border-left: 3.5px solid #10B981 !important;
    }
    .stat-chip-label {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-color) !important;
        opacity: 0.70;
        margin-bottom: 2px;
    }
    .stat-chip-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-color) !important;
    }
    .stat-chip-val-highlight {
        font-size: 1.25rem;
        font-weight: 800;
        color: #00B4D8 !important;
    }
    .stat-chip-sub {
        font-size: 0.85rem;
        font-weight: 400;
        color: var(--text-color) !important;
        opacity: 0.70;
    }
    .metric-card {
        background-color: var(--secondary-background-color) !important;
        border-radius: 6px;
        padding: 10px 18px !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        margin-bottom: 12px !important;
        color: var(--text-color) !important;
    }
    .metric-card-title {
        color: var(--text-color) !important;
        font-weight: 600;
        font-size: 0.90rem;
    }
    .metric-card-sub {
        color: var(--text-color) !important;
        opacity: 0.70;
        font-size: 0.8rem;
    }
    .badge-cyan { color: #00B4D8 !important; font-weight: bold; }
    .badge-gold { color: #D97706 !important; font-weight: bold; }
    .badge-green { color: #10B981 !important; font-weight: bold; }
    .badge-coral { color: #E11D48 !important; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; margin-bottom: 4px; }
    
    h3 {
        margin-top: 14px !important;
        margin-bottom: 8px !important;
        font-size: 1.2rem !important;
        color: var(--text-color) !important;
    }

    .panel-header-card {
        font-size: 0.88rem;
        font-weight: 700;
        padding: 7px 10px;
        border-radius: 6px;
        margin-bottom: 4px;
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        text-align: center;
        letter-spacing: 0.02em;
    }
    .panel-header-raw {
        color: #0F172A !important;
        border-top: 3px solid #64748B !important;
    }
    .panel-header-ukd {
        color: #0284C7 !important;
        border-top: 3px solid #0284C7 !important;
    }
    .panel-header-kei {
        color: #B45309 !important;
        border-top: 3px solid #B45309 !important;
    }
    .panel-header-custom {
        color: #047857 !important;
        border-top: 3px solid #047857 !important;
    }

    /* Response Curve Axis, Ticks & Legend styling: High-contrast Light Mode */
    div[data-testid="stPlotlyChart"] .g-xtitle text,
    div[data-testid="stPlotlyChart"] .g-ytitle text,
    div[data-testid="stPlotlyChart"] .xtick text,
    div[data-testid="stPlotlyChart"] .ytick text,
    div[data-testid="stPlotlyChart"] .legend text,
    div[data-testid="stPlotlyChart"] .legendtext,
    div[data-testid="stPlotlyChart"] text.xtitle,
    div[data-testid="stPlotlyChart"] text.ytitle {
        fill: #0F172A !important;
        color: #0F172A !important;
    }

    .stPlotlyChart .xgrid,
    .stPlotlyChart .ygrid {
        stroke: rgba(0, 0, 0, 0.08) !important;
    }
    .stPlotlyChart .zerolinelayer path {
        stroke: rgba(0, 0, 0, 0.22) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def get_cached_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_data_manager() -> S3DataManager:
    cfg = get_cached_config()
    return S3DataManager(cfg)


@st.cache_data(show_spinner="Loading cohort metadata from cloud storage...")
def get_base_cohort_dataframe() -> pd.DataFrame:
    dm = get_data_manager()
    return dm.load_raw_cohort_metadata()


@st.cache_data(ttl=5, show_spinner=False)
def get_curations_dict() -> Dict[str, Dict[str, Any]]:
    dm = get_data_manager()
    return dm.load_curations_dict()


def get_cohort_dataframe() -> pd.DataFrame:
    df_base = get_base_cohort_dataframe()
    curations_dict = get_curations_dict()
    return S3DataManager.merge_curations(df_base, curations_dict)


@st.cache_resource(show_spinner=False, max_entries=8)
def get_core_bundle(core_id: str) -> CoreBundle:
    dm = get_data_manager()
    return dm.load_core_bundle(core_id)


# Caches the static response curve per core (vectorized, <1ms)
@st.cache_data(show_spinner=False)
def get_cached_response_curve(core_id: str, perc_pos_th: float, max_val: float):
    bundle = get_core_bundle(core_id)
    if abs(perc_pos_th - 0.5) < 1e-4 and len(bundle.sorted_crit_p50) > 0:
        sweep_ths = np.arange(0.0, max_val + 1.0, 1.0)
        idx = np.searchsorted(bundle.sorted_crit_p50, sweep_ths, side="right")
        sweep_pcts = ((len(bundle.sorted_crit_p50) - idx) / len(bundle.sorted_crit_p50) * 100.0).tolist()
        return sweep_ths, sweep_pcts
    sweep_ths = np.linspace(0.0, max_val, 70)
    sweep_pcts = [bundle.evaluate_threshold(float(t), perc_pos_th)[2] for t in sweep_ths]
    return sweep_ths, sweep_pcts


# Caches the static panels (Raw, UKD, Keisuke) and base64 URIs per core
@st.cache_resource(show_spinner=False, max_entries=8)
def get_cached_static_images(
    core_id: str,
    ukd_abs: Optional[float],
    kei_abs: Optional[float],
    perc_pos_th: float,
    show_neg: bool,
    hide_masks: bool,
    alpha_fill: float,
    alpha_bound: float,
    color_ukd: str,
    color_kei: str,
):
    bundle = get_core_bundle(core_id)

    img_raw = bundle.render_panel(raw_only=True)
    uri_raw = pil_to_data_uri(img_raw)

    if ukd_abs is not None:
        is_pos_ukd, _, _ = bundle.evaluate_threshold(ukd_abs, perc_pos_th)
        img_ukd = bundle.render_panel(
            abs_th=ukd_abs,
            is_pos=is_pos_ukd,
            color_hex=color_ukd,
            alpha=alpha_fill if not hide_masks else 0.0,
            alpha_boundary=alpha_bound if not hide_masks else 0.0,
            show_neg=show_neg and not hide_masks,
        )
        uri_ukd = pil_to_data_uri(img_ukd)
    else:
        img_ukd = None
        uri_ukd = None

    if kei_abs is not None:
        is_pos_kei, _, _ = bundle.evaluate_threshold(kei_abs, perc_pos_th)
        img_kei = bundle.render_panel(
            abs_th=kei_abs,
            is_pos=is_pos_kei,
            color_hex=color_kei,
            alpha=alpha_fill if not hide_masks else 0.0,
            alpha_boundary=alpha_bound if not hide_masks else 0.0,
            show_neg=show_neg and not hide_masks,
        )
        uri_kei = pil_to_data_uri(img_kei)
    else:
        img_kei = None
        uri_kei = None

    return img_raw, img_ukd, img_kei, uri_raw, uri_ukd, uri_kei


# High-resolution inspection modal
@st.dialog("🔬 High-Resolution Image Inspection", width="large")
def show_fullscreen_modal(pil_image: Image.Image, panel_title: str, subtitle: str = ""):
    st.markdown(f"### {panel_title}")
    if subtitle:
        st.caption(subtitle)
    st.image(pil_image, use_container_width=True)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    st.download_button(
        "📥 Download High-Res PNG",
        data=buf.getvalue(),
        file_name=f"{st.session_state.current_core_id}_{panel_title.replace(' ', '_').lower()}.png",
        mime="image/png",
    )


cfg = get_cached_config()
dm = get_data_manager()
df_cohort = get_cohort_dataframe()

# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------
if "current_core_id" not in st.session_state:
    default_cid = cfg.get("local_source", {}).get("default_core", "1-1")
    if default_cid in df_cohort["core_id"].values:
        st.session_state.current_core_id = default_cid
    else:
        st.session_state.current_core_id = df_cohort["core_id"].iloc[0]

if "last_core_id" not in st.session_state:
    st.session_state.last_core_id = st.session_state.current_core_id

if "custom_abs_th" not in st.session_state:
    st.session_state.custom_abs_th = 15.0

if "perc_pos_th" not in st.session_state:
    st.session_state.perc_pos_th = 0.50

if "show_neg" not in st.session_state:
    st.session_state.show_neg = False

if "hide_masks" not in st.session_state:
    st.session_state.hide_masks = False

if "zoom_reset_token" not in st.session_state:
    st.session_state.zoom_reset_token = 0

if "nav_dragmode" not in st.session_state:
    st.session_state.nav_dragmode = "zoom"

# Reset review notes whenever the core changes
if st.session_state.current_core_id != st.session_state.last_core_id:
    st.session_state.last_core_id = st.session_state.current_core_id
    note_key = f"notes_{st.session_state.current_core_id}"
    st.session_state[note_key] = ""

# ------------------------------------------------------------------------------
# Sidebar: Navigation & Controls
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 PD-1 Threshold Tuner")
    storage_badge = "☁️ AWS S3" if dm.use_s3 else "📁 Local Cache"
    st.caption(f"Cohort: {len(df_cohort)} cores | Storage: **{storage_badge}**")

    # Filter options
    filter_choice = st.selectbox(
        "Filter Cores:",
        ["All Cores", "Unreviewed Only", "Reviewed Only", "High Discrepancy (|UKD - Kei| > 15)", "Filter by Diagnosis"],
        index=0,
    )

    df_filtered = df_cohort.copy()
    if filter_choice == "Unreviewed Only":
        df_filtered = df_filtered[df_filtered["cur_decision"].astype(str).str.strip() == "Unreviewed"]
    elif filter_choice == "Reviewed Only":
        df_filtered = df_filtered[df_filtered["cur_decision"].astype(str).str.strip() != "Unreviewed"]
    elif filter_choice == "High Discrepancy (|UKD - Kei| > 15)":
        mask = (df_filtered["our_abs_th"].notna()) & (df_filtered["kei_abs_th"].notna())
        diff = np.abs(df_filtered["our_abs_th"] - df_filtered["kei_abs_th"])
        df_filtered = df_filtered[mask & (diff >= 15.0)]
    elif filter_choice == "Filter by Diagnosis":
        diag_list = sorted(df_cohort["diagnosis"].unique())
        selected_diag = st.selectbox("Diagnosis:", diag_list)
        df_filtered = df_filtered[df_filtered["diagnosis"] == selected_diag]

    if df_filtered.empty:
        st.warning("No cores match current filter.")
        df_filtered = df_cohort

    core_list = df_filtered["core_id"].tolist()
    if st.session_state.current_core_id not in core_list:
        st.session_state.current_core_id = core_list[0]

    current_idx = core_list.index(st.session_state.current_core_id)

    # Core Selector
    def format_core_label(cid: str) -> str:
        matching = df_cohort[df_cohort["core_id"] == cid]
        if matching.empty:
            return str(cid)
        row = matching.iloc[0]
        cur_icon = "✅" if str(row.get("cur_decision", "Unreviewed")).strip() != "Unreviewed" else "⏳"
        return f"{cur_icon} {cid} | IHC: {row['ihc_str']} | {row['diagnosis']}"

    def on_core_dropdown_change():
        new_cid = st.session_state.selected_core_dropdown
        st.session_state.current_core_id = new_cid
        st.session_state[f"notes_{new_cid}"] = ""

    # Keep dropdown widget state synchronized with current_core_id
    if "selected_core_dropdown" not in st.session_state or st.session_state.selected_core_dropdown != st.session_state.current_core_id:
        st.session_state.selected_core_dropdown = st.session_state.current_core_id

    st.selectbox(
        "Select Core:",
        options=core_list,
        format_func=format_core_label,
        key="selected_core_dropdown",
        on_change=on_core_dropdown_change,
    )

    # Prev / Next navigation via on_click callbacks (zero ghosting / flicker)
    def go_prev_core():
        if current_idx > 0:
            prev_id = core_list[current_idx - 1]
            st.session_state.current_core_id = prev_id
            st.session_state.selected_core_dropdown = prev_id
            st.session_state[f"notes_{prev_id}"] = ""

    def go_next_core():
        if current_idx < len(core_list) - 1:
            next_id = core_list[current_idx + 1]
            st.session_state.current_core_id = next_id
            st.session_state.selected_core_dropdown = next_id
            st.session_state[f"notes_{next_id}"] = ""

    col_prev, col_next = st.columns(2)
    with col_prev:
        st.button("⬅ Prev", disabled=current_idx == 0, on_click=go_prev_core)
    with col_next:
        st.button("Next ➡", disabled=current_idx >= len(core_list) - 1, on_click=go_next_core)

    st.markdown("---")
    st.subheader("🎚️ Threshold Controls")

    current_row = df_cohort[df_cohort["core_id"] == st.session_state.current_core_id].iloc[0]
    bundle = get_core_bundle(current_row["core_id"])

    # Quick Presets: One option per line
    st.write("**Quick Presets (one click):**")

    ukd_lbl = f"{current_row['our_abs_th']:.1f}" if pd.notna(current_row["our_abs_th"]) else "N/A"
    if st.button(f"Use UKD ({ukd_lbl})", help="Set threshold to UKD pipeline cutoff"):
        if pd.notna(current_row["our_abs_th"]):
            st.session_state.custom_abs_th = float(current_row["our_abs_th"])
            st.rerun()

    kei_lbl = f"{current_row['kei_abs_th']:.1f}" if pd.notna(current_row["kei_abs_th"]) else "N/A"
    if st.button(f"Use Keisuke ({kei_lbl})", help="Set threshold to Keisuke reviewer cutoff"):
        if pd.notna(current_row["kei_abs_th"]):
            st.session_state.custom_abs_th = float(current_row["kei_abs_th"])
            st.rerun()

    if st.button("Use Midpoint (UKD / Keisuke)", help="Average of UKD and Keisuke cutoffs"):
        if pd.notna(current_row["our_abs_th"]) and pd.notna(current_row["kei_abs_th"]):
            st.session_state.custom_abs_th = float(
                round(0.5 * (current_row["our_abs_th"] + current_row["kei_abs_th"]), 1)
            )
            st.rerun()

    ihc_lbl = current_row["ihc_str"]
    if st.button(f"Auto-Match Clinical IHC % ({ihc_lbl})", help="Analytical solver matching Clinical IHC %"):
        if current_row["is_valid_ihc"] and current_row["ihc_val"] is not None:
            solved_th = bundle.solve_threshold_for_target_pct(
                float(current_row["ihc_val"]), float(st.session_state.perc_pos_th)
            )
            st.session_state.custom_abs_th = float(solved_th)
            st.rerun()

    st.markdown("---")

    # Continuous Slider (0 to 255, step 1)
    custom_slider = st.slider(
        "Custom abs_threshold:",
        min_value=0.0,
        max_value=255.0,
        value=float(min(255.0, max(0.0, st.session_state.custom_abs_th))),
        step=1.0,
    )
    if custom_slider != st.session_state.custom_abs_th:
        st.session_state.custom_abs_th = custom_slider

    # Display toggles
    st.markdown("---")
    st.write("**Display Toggles:**")
    st.session_state.show_neg = st.checkbox("Show Negative Cell Outlines (Red)", value=st.session_state.show_neg)
    st.session_state.hide_masks = st.checkbox("Hide All Masks (Inspect Raw Signal)", value=st.session_state.hide_masks)

    with st.expander("⚙️ Advanced Settings"):
        st.session_state.perc_pos_th = st.slider(
            "perc_pos_threshold:",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state.perc_pos_th),
            step=0.05,
        )
        alpha_fill = st.slider("Mask Interior Alpha:", 0.0, 0.4, float(cfg["plot"].get("mask_alpha", 0.08)), 0.02)
        alpha_bound = st.slider("Mask Boundary Alpha:", 0.2, 1.0, float(cfg["plot"].get("mask_alpha_boundary", 1.0)), 0.1)

# ------------------------------------------------------------------------------
# Main Content: Header & Metrics
# ------------------------------------------------------------------------------
core_id = current_row["core_id"]
ihc_str = current_row["ihc_str"]
diagnosis = current_row["diagnosis"]
subtype = current_row["subtype"]
total_cells = bundle.total_cells

ukd_abs = current_row["our_abs_th"]
kei_abs = current_row["kei_abs_th"]
custom_abs = float(st.session_state.custom_abs_th)
perc_pos = float(st.session_state.perc_pos_th)

# Live evaluations
is_pos_ukd, n_ukd, pct_ukd = bundle.evaluate_threshold(ukd_abs or 999.0, perc_pos) if ukd_abs else (None, 0, 0.0)
is_pos_kei, n_kei, pct_kei = bundle.evaluate_threshold(kei_abs or 999.0, perc_pos) if kei_abs else (None, 0, 0.0)
is_pos_custom, n_custom, pct_custom = bundle.evaluate_threshold(custom_abs, perc_pos)

# Top Core Stat Banner (Structured, modern metadata badges)
st.markdown(
    f"""
    <div class="core-stats-banner">
        <div class="stat-chip stat-chip-core">
            <span class="stat-chip-label">Core ID</span>
            <span class="stat-chip-val-highlight">{core_id}</span>
        </div>
        <div class="stat-chip stat-chip-diag">
            <span class="stat-chip-label">Pathology Diagnosis</span>
            <span class="stat-chip-val">{diagnosis} <span class="stat-chip-sub">({subtype})</span></span>
        </div>
        <div class="stat-chip stat-chip-cells">
            <span class="stat-chip-label">Total Segmented Cells</span>
            <span class="stat-chip-val" style="color: #10B981;">{total_cells:,}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m_c1, m_c2, m_c3, m_c4 = st.columns(4)
with m_c1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-card-title"><b>Clinical IHC PD-1</b></div>'
        f'<h2 style="color: var(--text-color); margin: 2px 0;">{ihc_str}</h2>'
        f'<span class="metric-card-sub">Raw image</span></div>',
        unsafe_allow_html=True,
    )
with m_c2:
    ukd_label = f"abs_th={ukd_abs:.1f}" if ukd_abs is not None else "N/A"
    st.markdown(
        f'<div class="metric-card"><div class="metric-card-title"><b>UKD Pipeline (<span class="badge-cyan">Cyan</span>)</b></div>'
        f'<h2 style="color:#00B4D8; margin: 2px 0;">{pct_ukd:.1f}%</h2>'
        f'<span class="metric-card-sub">{n_ukd:,} cells ({ukd_label})</span></div>',
        unsafe_allow_html=True,
    )
with m_c3:
    kei_label = f"abs_th={kei_abs:.1f}" if kei_abs is not None else "N/A"
    st.markdown(
        f'<div class="metric-card"><div class="metric-card-title"><b>Keisuke Reviewer (<span class="badge-gold">Gold</span>)</b></div>'
        f'<h2 style="color:#D97706; margin: 2px 0;">{pct_kei:.1f}%</h2>'
        f'<span class="metric-card-sub">{n_kei:,} cells ({kei_label})</span></div>',
        unsafe_allow_html=True,
    )
with m_c4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-card-title"><b>Interactive Custom (<span class="badge-green">Lime</span>)</b></div>'
        f'<h2 style="color:#10B981; margin: 2px 0;">{pct_custom:.1f}%</h2>'
        f'<span class="metric-card-sub">{n_custom:,} cells (abs_th={custom_abs:.1f})</span></div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# 4-Panel Synchronized Comparison (Strict 1:1 Aspect Ratio + Linked Zoom/Pan)
# ------------------------------------------------------------------------------
sync_hdr_col1, sync_hdr_col2, sync_hdr_col3 = st.columns([2.5, 1.1, 0.8])
with sync_hdr_col1:
    st.markdown("### 🔬 Synchronized 4-Panel Comparison")
    st.caption("💡 *Tip: Drag on any image to zoom in simultaneously across all 4 panels. Double-click any image to reset.*")
with sync_hdr_col2:
    mode_selection = st.radio(
        "Navigation Mode:",
        ["🔎 Box Zoom", "✥ Pan"],
        horizontal=True,
        label_visibility="collapsed",
        key="nav_mode_radio",
    )
    st.session_state.nav_dragmode = "zoom" if "Zoom" in mode_selection else "pan"
with sync_hdr_col3:
    if st.button("↺ Reset View", use_container_width=True, help="Reset zoom back to full view"):
        st.session_state.zoom_reset_token = st.session_state.get("zoom_reset_token", 0) + 1
        st.rerun()

color_ukd = cfg["plot"].get("ukd_mask_color", "#00B4D8")
color_kei = cfg["plot"].get("keisuke_mask_color", "#D97706")

# 1. Fetch cached static images & base64 URIs (0ms on slider movement)
img_raw, img_ukd, img_kei, uri_raw, uri_ukd, uri_kei = get_cached_static_images(
    core_id=core_id,
    ukd_abs=ukd_abs,
    kei_abs=kei_abs,
    perc_pos_th=perc_pos,
    show_neg=st.session_state.show_neg,
    hide_masks=st.session_state.hide_masks,
    alpha_fill=alpha_fill,
    alpha_bound=alpha_bound,
    color_ukd=color_ukd,
    color_kei=color_kei,
)

# 2. Only render the Custom panel that changed (~35ms)
img_custom = bundle.render_panel(
    abs_th=custom_abs,
    is_pos=is_pos_custom,
    color_hex=cfg["plot"].get("custom_pos_color", "#10B981"),
    alpha=alpha_fill if not st.session_state.hide_masks else 0.0,
    alpha_boundary=alpha_bound if not st.session_state.hide_masks else 0.0,
    show_neg=st.session_state.show_neg and not st.session_state.hide_masks,
)
uri_custom = pil_to_data_uri(img_custom)

H, W = img_raw.height, img_raw.width
blank_uri = pil_to_data_uri(Image.new("RGB", (W, H), (15, 18, 24)))

ukd_hdr = f"Panel 2: UKD ({ukd_abs:.1f})" if ukd_abs is not None else "Panel 2: UKD (Unreviewed)"
kei_hdr = f"Panel 3: Keisuke ({kei_abs:.1f})" if kei_abs is not None else "Panel 3: Keisuke (Unreviewed)"
custom_hdr = f"Panel 4: Custom ({custom_abs:.1f})"

# Distinct native HTML headers aligned perfectly above the 4 panels
col_h1, col_h2, col_h3, col_h4 = st.columns(4)
with col_h1:
    st.markdown('<div class="panel-header-card panel-header-raw">Panel 1: Raw PD-1 Staining</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown(f'<div class="panel-header-card panel-header-ukd">{ukd_hdr}</div>', unsafe_allow_html=True)
with col_h3:
    st.markdown(f'<div class="panel-header-card panel-header-kei">{kei_hdr}</div>', unsafe_allow_html=True)
with col_h4:
    st.markdown(f'<div class="panel-header-card panel-header-custom">{custom_hdr}</div>', unsafe_allow_html=True)

fig_panels = make_subplots(
    rows=1,
    cols=4,
    horizontal_spacing=0.015,
)

# Add images as high-performance base64 data URIs
fig_panels.add_trace(go.Image(source=uri_raw, x0=0, y0=0, dx=1, dy=1, hoverinfo="none"), row=1, col=1)
fig_panels.add_trace(go.Image(source=uri_ukd if uri_ukd else blank_uri, x0=0, y0=0, dx=1, dy=1, hoverinfo="none"), row=1, col=2)
fig_panels.add_trace(go.Image(source=uri_kei if uri_kei else blank_uri, x0=0, y0=0, dx=1, dy=1, hoverinfo="none"), row=1, col=3)
fig_panels.add_trace(go.Image(source=uri_custom, x0=0, y0=0, dx=1, dy=1, hoverinfo="none"), row=1, col=4)

# Strictly enforce 1:1 circular aspect ratio, shared zoom/pan, and bounded range
for i in range(1, 5):
    xname = "xaxis" if i == 1 else f"xaxis{i}"
    yname = "yaxis" if i == 1 else f"yaxis{i}"

    if i == 1:
        fig_panels.layout[xname].update(
            range=[0, W],
            minallowed=0,
            maxallowed=W,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            constrain="domain",
        )
        fig_panels.layout[yname].update(
            range=[H, 0],
            minallowed=0,
            maxallowed=H,
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            constrain="domain",
        )
    else:
        fig_panels.layout[xname].update(
            matches="x",
            range=[0, W],
            minallowed=0,
            maxallowed=W,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            constrain="domain",
        )
        fig_panels.layout[yname].update(
            matches="y",
            range=[H, 0],
            minallowed=0,
            maxallowed=H,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            constrain="domain",
        )

fig_panels.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=6, b=6),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    dragmode=st.session_state.nav_dragmode,
    uirevision=f"{core_id}_{st.session_state.get('zoom_reset_token', 0)}",
)

panel_config = {
    "displayModeBar": False,
    "scrollZoom": True,
    "doubleClick": "reset",
}

st.plotly_chart(fig_panels, use_container_width=True, config=panel_config)

# Fullscreen modal inspection buttons below each corresponding panel
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    if st.button("⛶ Fullscreen", key="modal_btn_raw", use_container_width=True):
        show_fullscreen_modal(img_raw, "Panel 1: Raw PD-1 Staining", f'Core "{core_id}" | Clinical IHC: {ihc_str}')
with col_m2:
    if img_ukd is not None:
        if st.button("⛶ Fullscreen", key="modal_btn_ukd", use_container_width=True):
            show_fullscreen_modal(img_ukd, ukd_hdr, f"Cutoff: {ukd_abs} | Positive: {n_ukd:,} cells ({pct_ukd:.1f}%) [Cyan]")
    else:
        st.button("⛶ Fullscreen", key="modal_btn_ukd_dis", use_container_width=True, disabled=True)
with col_m3:
    if img_kei is not None:
        if st.button("⛶ Fullscreen", key="modal_btn_kei", use_container_width=True):
            show_fullscreen_modal(img_kei, kei_hdr, f"Cutoff: {kei_abs} | Positive: {n_kei:,} cells ({pct_kei:.1f}%) [Gold]")
    else:
        st.button("⛶ Fullscreen", key="modal_btn_kei_dis", use_container_width=True, disabled=True)
with col_m4:
    if st.button("⛶ Fullscreen", key="modal_btn_custom", use_container_width=True):
        show_fullscreen_modal(img_custom, custom_hdr, f"Cutoff: {custom_abs:.1f} | Positive: {n_custom:,} cells ({pct_custom:.1f}%) [Lime Green]")

# ------------------------------------------------------------------------------
# Threshold vs. Positivity Response Curve (High-Contrast Light Mode)
# ------------------------------------------------------------------------------
st.markdown("### 📈 Threshold vs. Positivity Response Curve")

curve_color = "#0284C7"       # Vibrant Deep Blue
axis_title_col = "#0F172A"    # High-contrast Pitch Black / Slate-900
axis_tick_col = "#0F172A"     # High-contrast Pitch Black / Slate-900
legend_col = "#0F172A"        # High-contrast Pitch Black
grid_col = "rgba(0, 0, 0, 0.08)"
zeroline_col = "rgba(0, 0, 0, 0.22)"

# High-contrast WCAG-compliant badges on light canvas
c_kei = "#B45309"             # Dark Amber Gold
bg_kei = "rgba(245, 158, 11, 0.15)"
c_custom = "#047857"          # Dark Emerald
bg_custom = "rgba(16, 185, 129, 0.15)"
c_ukd = "#0284C7"             # Dark Sky Blue / Cyan
bg_ukd = "rgba(0, 180, 216, 0.15)"
c_ihc = "#BE123C"             # Dark Crimson Rose
bg_ihc = "rgba(225, 29, 72, 0.15)"

# Fetch cached response curve (vectorized, <1ms)
sweep_ths, sweep_pcts = get_cached_response_curve(
    core_id=core_id,
    perc_pos_th=perc_pos,
    max_val=255.0,
)

fig_curve = go.Figure()
fig_curve.add_trace(
    go.Scatter(
        x=sweep_ths,
        y=sweep_pcts,
        mode="lines",
        name="Positive Cells (%)\u00A0\u00A0\u00A0\u00A0",
        line=dict(color=curve_color, width=2.5),
    )
)

# Legend proxies for threshold markers
if ukd_abs is not None:
    fig_curve.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=f"UKD ({ukd_abs:.1f})\u00A0\u00A0\u00A0\u00A0",
            line=dict(color=c_ukd, dash="dash", width=2),
        )
    )
if kei_abs is not None:
    fig_curve.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=f"Keisuke ({kei_abs:.1f})\u00A0\u00A0\u00A0\u00A0",
            line=dict(color=c_kei, dash="dash", width=2),
        )
    )
fig_curve.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        name=f"Custom ({custom_abs:.1f})\u00A0\u00A0\u00A0\u00A0",
        line=dict(color=c_custom, width=3),
    )
)
if current_row["is_valid_ihc"] and current_row["ihc_val"] is not None:
    fig_curve.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=f"Clinical IHC ({float(current_row['ihc_val']):.1f}%)",
            line=dict(color=c_ihc, dash="dot", width=2),
        )
    )

# Staggered vertical markers with non-overlapping pill badges
# Level 1 (y=88): Keisuke (Amber Gold)
if kei_abs is not None:
    fig_curve.add_vline(x=kei_abs, line=dict(color=c_kei, dash="dash", width=2))
    xanc = "left" if kei_abs < 35 else ("right" if kei_abs > 220 else "center")
    fig_curve.add_annotation(
        x=kei_abs,
        y=88,
        text=f"<b>Keisuke ({kei_abs:.1f})</b>",
        showarrow=False,
        xanchor=xanc,
        width=140,
        align="center",
        font=dict(color=c_kei, size=11),
        bgcolor=bg_kei,
        bordercolor=c_kei,
        borderwidth=1.5,
        borderpad=6,
    )

# Level 2 (y=68): Custom (Lime Green)
fig_curve.add_vline(x=custom_abs, line=dict(color=c_custom, width=3))
xanc = "left" if custom_abs < 35 else ("right" if custom_abs > 220 else "center")
fig_curve.add_annotation(
    x=custom_abs,
    y=68,
    text=f"<b>Custom ({custom_abs:.1f})</b>",
    showarrow=False,
    xanchor=xanc,
    width=140,
    align="center",
    font=dict(color=c_custom, size=11),
    bgcolor=bg_custom,
    bordercolor=c_custom,
    borderwidth=1.5,
    borderpad=6,
)

# Level 3 (y=48): UKD (Cyan)
if ukd_abs is not None:
    fig_curve.add_vline(x=ukd_abs, line=dict(color=c_ukd, dash="dash", width=2))
    xanc = "left" if ukd_abs < 35 else ("right" if ukd_abs > 220 else "center")
    fig_curve.add_annotation(
        x=ukd_abs,
        y=48,
        text=f"<b>UKD ({ukd_abs:.1f})</b>",
        showarrow=False,
        xanchor=xanc,
        width=140,
        align="center",
        font=dict(color=c_ukd, size=11),
        bgcolor=bg_ukd,
        bordercolor=c_ukd,
        borderwidth=1.5,
        borderpad=6,
    )

# Clinical IHC Reference (Rose Red)
if current_row["is_valid_ihc"] and current_row["ihc_val"] is not None:
    ihc_val = float(current_row["ihc_val"])
    fig_curve.add_hline(y=ihc_val, line=dict(color=c_ihc, dash="dot", width=2))
    yanc = "bottom" if ihc_val < 85 else "top"
    fig_curve.add_annotation(
        x=210,
        y=ihc_val,
        text=f"<b>Clinical IHC ({ihc_val:.1f}%)</b>",
        showarrow=False,
        xanchor="right",
        yanchor=yanc,
        width=180,
        align="center",
        font=dict(color=c_ihc, size=11),
        bgcolor=bg_ihc,
        bordercolor=c_ihc,
        borderwidth=1.5,
        borderpad=6,
    )

fig_curve.update_layout(
    height=270,
    margin=dict(l=40, r=40, t=35, b=35),
    xaxis_title="PD-1 Intensity Cutoff (abs_threshold)",
    yaxis_title="Positive Cells (%)",
    xaxis=dict(
        title=dict(font=dict(color=axis_title_col, size=12)),
        tickfont=dict(color=axis_tick_col, size=11),
        range=[0, 255],
        showgrid=True,
        gridcolor=grid_col,
        zerolinecolor=zeroline_col,
    ),
    yaxis=dict(
        title=dict(font=dict(color=axis_title_col, size=12)),
        tickfont=dict(color=axis_tick_col, size=11),
        range=[0, 100],
        showgrid=True,
        gridcolor=grid_col,
        zerolinecolor=zeroline_col,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="center",
        x=0.5,
        font=dict(color=legend_col, size=11),
        itemclick=False,
        itemdoubleclick=False,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_curve, use_container_width=True, config={"displayModeBar": False}, theme=None)

# ------------------------------------------------------------------------------
# Reviewer Decision Card
# ------------------------------------------------------------------------------
st.markdown("### 🔬 Record Reviewer Decision")

note_key = f"notes_{core_id}"
if note_key not in st.session_state:
    st.session_state[note_key] = ""

with st.container():
    c_dec1, c_dec2 = st.columns([1, 2])

    with c_dec1:
        decision_choice = st.radio(
            "Select Decision:",
            options=["Accept UKD", "Accept Keisuke", "Accept Custom Threshold", "Flag / Exclude Core"],
            index=2,
        )

        if decision_choice == "Accept UKD":
            final_abs = ukd_abs
        elif decision_choice == "Accept Keisuke":
            final_abs = kei_abs
        elif decision_choice == "Accept Custom Threshold":
            final_abs = custom_abs
        else:
            final_abs = None

    with c_dec2:
        review_notes = st.text_area(
            "Reviewer Notes / Rational:",
            value=st.session_state.get(note_key, ""),
            key=note_key,
            placeholder="Enter rationale for threshold decision or specific core observations (cleared on core change)...",
            height=90,
        )

    if st.button("💾 Save Decision to Cloud", type="primary"):
        save_target = dm.save_curation_decision(
            core_id=core_id,
            decision_type=decision_choice,
            abs_th=final_abs,
            perc_pos_th=perc_pos,
            notes=review_notes,
        )

        # Clear cached curations so new curation is merged immediately
        get_curations_dict.clear()

        # Cache the feedback message across rerun
        if final_abs is not None:
            st.session_state.save_feedback = ("success", f"Saved decision for Core **{core_id}**! Final threshold: **{final_abs:.1f}**")
        else:
            st.session_state.save_feedback = ("warning", f"Core **{core_id}** flagged as Excluded.")

        st.rerun()

    # Display feedback message if present
    if "save_feedback" in st.session_state:
        msg_type, msg_text = st.session_state.pop("save_feedback")
        if msg_type == "success":
            st.success(msg_text)
        else:
            st.warning(msg_text)

# ------------------------------------------------------------------------------
# Curated Progress Table
# ------------------------------------------------------------------------------
curations_dict = get_curations_dict()
if curations_dict:
    df_cur = pd.DataFrame.from_dict(curations_dict, orient="index").reset_index()
    if "index" in df_cur.columns:
        df_cur = df_cur.rename(columns={"index": "core_id"})
    with st.expander("📊 Review Progress & Saved Curations", expanded=False):
        st.write(f"Total Reviewed Cores: **{len(df_cur)} / {len(df_cohort)}**")
        st.dataframe(df_cur, use_container_width=True)
        csv_bytes = df_cur.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Curated CSV",
            data=csv_bytes,
            file_name="curated_pd1_thresholds.csv",
            mime="text/csv",
        )

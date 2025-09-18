import argparse
import hashlib
import logging
import shutil
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, UTC
from pathlib import Path
from config import PROCESSED_DIR, SNAPSHOT_DIR

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def md5_of_file(path, block_size=2**20):
    """Generate an MD5 checksum for a file by reading in chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_raw_file(src_path: str, snapshot_dir: str) -> str:
    """
    Copy raw file to snapshot_dir with timestamp and write checksum meta.
    Returns path to copied snapshot.
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    basename = os.path.basename(src_path)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root, ext = os.path.splitext(basename)
    dest_name = f"{root}_snapshot_{ts}{ext}"
    dest_path = os.path.join(snapshot_dir, dest_name)

    shutil.copy2(src_path, dest_path)
    checksum = md5_of_file(dest_path)
    meta = {
        "original": os.path.abspath(src_path),
        "snapshot": os.path.abspath(dest_path),
        "md5": checksum,
        "timestamp_utc": ts
    }
    meta_path = dest_path + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Created snapshot {dest_path} and meta {meta_path}")

    return dest_path


def load_raw_csv(path: str) -> pd.DataFrame:
    p = Path(str(path).strip()).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    logger.info(f"Loading CSV from {p}")
    df = pd.read_csv(p)
    logger.info(f"Loaded dataframe with shape {df.shape}")

    return df


def profile_df(df: pd.DataFrame) -> dict:
    """Return a small profiling dict (shape, dtypes, missing, uniques, numeric summary)."""
    profile = {}
    profile["shape"] = f"{df.shape[0]} rows × {df.shape[1]} columns"
    profile["dtypes"] = df.dtypes.apply(lambda x: x.name).to_dict()
    profile["missing_counts"] = df.isna().sum().sort_values(ascending=False).to_dict()
    profile["n_unique"] = df.nunique().to_dict()
    # numeric summary
    num = df.select_dtypes(include=[np.number])
    profile["numeric_summary"] = num.describe().to_dict()
    return profile


def minimal_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal deterministic cleaning
    - standardize column names
    - replace common NA strings with np.nan
    - drop totally empty columns
    - drop exact duplicates
    """
    df = df.copy()

    # Standardize column names
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Replace obvious NA placeholders
    df.replace({"NA": np.nan, "N/A": np.nan, "": np.nan}, inplace=True)

    # Drop columns that are fully null
    null_frac = df.isna().mean()
    cols_all_null = null_frac[null_frac >= 1.0].index.tolist()
    if cols_all_null:
        logger.info(f"Dropping columns fully null: {cols_all_null}")
        df.drop(columns=cols_all_null, inplace=True)

    # Drop exact duplicates
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_after = len(df)
    if n_after < n_before:
        logger.info(f"Dropped {n_before - n_after} duplicate rows")
        
    return df


def quick_imputations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe, common imputations for Ames dataset.
    - LotFrontage: median per Neighborhood
    - MasVnrArea, TotalBsmtSF, GarageArea: fill 0
    - Electrical: fill mode
    - GarageCars: 0 if GarageArea == 0 else median
    """
    df = df.copy()

    # Impute LotFrontage
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = (
            df.groupby("Neighborhood")["LotFrontage"]
            .transform(lambda x: x.fillna(x.median()))
        )
        # fallback global median
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # Fill zeros
    for col in ["MasVnrArea", "TotalBsmtSF", "GarageArea"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Electrical imputation
    if "Electrical" in df.columns and df["Electrical"].isna().any():
        mode_val = df["Electrical"].mode()
        if not mode_val.empty:
            df["Electrical"] = df["Electrical"].fillna(mode_val.iloc[0])

    # GarageCars
    if "GarageCars" in df.columns:
        mask = df["GarageCars"].isna()
        if mask.any():
            if "GarageArea" in df.columns:
                df.loc[mask & (df["GarageArea"] == 0), "GarageCars"] = 0
            df["GarageCars"] = df["GarageCars"].fillna(df["GarageCars"].median())

    return df


def tag_outliers_iqr(df: pd.DataFrame, cols=None, factor: float = 1.5) -> pd.DataFrame:
    """
    Tag outliers using IQR rule. Adds int columns '<col>_is_outlier' for numeric cols.
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - factor * iqr
            hi = q3 + factor * iqr
            out_col = f"{col}_is_outlier"
            df[out_col] = ((df[col] < lo) | (df[col] > hi)).astype(int)
        except Exception:
            continue
    return df


def cap_outliers_pct(
    df: pd.DataFrame,
    cols=None,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """Cap numeric columns at given percentiles."""
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        try:
            lo = df[col].quantile(lower_pct)
            hi = df[col].quantile(upper_pct)
            df[col] = df[col].clip(lower=lo, upper=hi)
        except Exception:
            continue
    return df


def write_profile_json(profile: dict, out_path: str):
    """Save a profile dict as JSON on disk."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)
    logger.info(f"Saved profile to {out_path}")


def run_pipeline(
    raw_csv_path: str,
    snapshot_dir: str,
    processed_path: str,
    minimal_cleaning_enabled: bool = True,
    quick_impute: bool = True,
    tag_outliers: bool = True,
    cap_outliers: bool = False,
    save_profile_json: bool = True,
    save_csv: bool = True,
):
    # 1) Snapshot raw
    snapshot_path = snapshot_raw_file(raw_csv_path, snapshot_dir)

    # 2) Load
    df = load_raw_csv(snapshot_path)

    # 3) Profile BEFORE any changes
    profile_before = profile_df(df)
    if save_profile_json:
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        write_profile_json(profile_before, processed_path + ".profile_before.json")

    # 4) Minimal cleaning
    if minimal_cleaning_enabled:
        df = minimal_cleaning(df)

    # 5) Quick impute
    if quick_impute:
        df = quick_imputations(df)

    # 6) Tag outliers
    if tag_outliers:
        df = tag_outliers_iqr(df)

    # 7) Cap outliers
    if cap_outliers:
        df = cap_outliers_pct(df)

    # 8) Profile AFTER
    profile_after = profile_df(df)
    if save_profile_json:
        write_profile_json(profile_after, processed_path + ".profile_after.json")

    # 9) Save processed
    if save_csv:
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        logger.info(f"Saved processed CSV to {processed_path} with shape {df.shape}")
    else:
        logger.info("Skipping CSV save as requested.")

    return {
        "snapshot_path": snapshot_path,
        "processed_path": processed_path,
        "profile_before": profile_before,
        "profile_after": profile_after,
    }


def parse_args_and_run():
    ap = argparse.ArgumentParser(
        description="Data intake and cleaning pipeline for Ames Housing (run from repo root)."
    )

    # Required/paths
    ap.add_argument("--raw-csv", required=True, help="Path to raw CSV file")
    ap.add_argument("--snapshot-dir", default=SNAPSHOT_DIR, help="Path to snapshot directory")
    ap.add_argument(
        "--out-csv",
        default=os.path.join(PROCESSED_DIR, "ames_cleaned.csv"),
        help="Path to processed CSV file",
    )

    # Minimal cleaning (default ON)
    g_clean = ap.add_mutually_exclusive_group()
    g_clean.add_argument(
        "--minimal-cleaning",
        dest="minimal_cleaning_enabled",
        action="store_true",
        help="Do basic cleaning (standardize names, drop empty cols, drop duplicates)",
    )
    g_clean.add_argument(
        "--no-minimal-cleaning",
        dest="minimal_cleaning_enabled",
        action="store_false",
        help="Skip basic cleaning",
    )
    ap.set_defaults(minimal_cleaning_enabled=True)

    # Quick imputations (default ON)
    g_impute = ap.add_mutually_exclusive_group()
    g_impute.add_argument(
        "--impute",
        dest="quick_impute",
        action="store_true",
        help="Enable quick imputations (LotFrontage by neighborhood median, etc.)",
    )
    g_impute.add_argument(
        "--no-impute",
        dest="quick_impute",
        action="store_false",
        help="Disable quick imputations",
    )
    ap.set_defaults(quick_impute=True)

    # Tag outliers (default ON)
    g_tag = ap.add_mutually_exclusive_group()
    g_tag.add_argument(
        "--tag-outliers",
        dest="tag_outliers",
        action="store_true",
        help="Add IQR-based outlier tag columns",
    )
    g_tag.add_argument(
        "--no-tag-outliers",
        dest="tag_outliers",
        action="store_false",
        help="Do not add outlier tags",
    )
    ap.set_defaults(tag_outliers=True)

    # Cap outliers (default OFF)
    g_cap = ap.add_mutually_exclusive_group()
    g_cap.add_argument(
        "--cap-outliers",
        dest="cap_outliers",
        action="store_true",
        help="Cap numeric columns at 1st/99th percentiles",
    )
    g_cap.add_argument(
        "--no-cap-outliers",
        dest="cap_outliers",
        action="store_false",
        help="Do not cap outliers",
    )
    ap.set_defaults(cap_outliers=False)

    # Optional saves (opt-in)
    ap.add_argument(
        "--save-profile",
        dest="save_profile_json",
        action="store_true",
        help="Save before/after profile JSON files",
    )
    ap.add_argument(
        "--save-csv",
        action="store_true",
        help="Save the processed CSV",
    )

    args = ap.parse_args()

    run_pipeline(
        raw_csv_path=args.raw_csv,
        snapshot_dir=args.snapshot_dir,
        processed_path=args.out_csv,
        minimal_cleaning_enabled=args.minimal_cleaning_enabled,
        quick_impute=args.quick_impute,
        tag_outliers=args.tag_outliers,
        cap_outliers=args.cap_outliers,
        save_profile_json=args.save_profile_json,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    parse_args_and_run()
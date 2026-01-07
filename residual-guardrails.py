#!/usr/bin/env python3
"""
MATRIXDM — Pantheon+ residual-structure diagnostics (PV / sky geometry / optional κ)

Purpose
-------
Reproducible diagnostic pipeline for the analyses reported in the associated paper.

        "Background-Consistent Residual Structure in Type Ia Supernovae"

The script constructs a fixed-reference Hubble-residual vector, projects out a conservative
low-rank nuisance basis, and evaluates whether the remaining ordering-dependent residual
structure correlates with LSS-adjacent proxies (peculiar velocity terms; optional κ).

Method summary
------------------------------
1) Define fixed-reference distance-modulus residuals
     r(z) = μ - μ_ref(z; H0_ref, Ωm_ref)
   where μ_ref is a flat-ΛCDM reference used only to define residuals.

2) Remove a low-rank nuisance subspace (if available in the table)
   using weighted least squares on:
     - intercept
     - z-scored HOST_LOGMASS, x1, c
     - survey one-hot (categories with sufficient counts)
   yielding r_struct = r - r_scale.

3) Collapse duplicate entries (if an identifier is available)
   via inverse-variance weighted means on z and r_struct.

4) Optional background-consistency guardrail
   If a DESI BAO chain directory is provided, draw (H0, Ωm) samples from the
   chain-implied Gaussian approximation and quantify the induced dispersion in
   μ_ref(z) relative to the baseline reference.

5) Dependence tests
   Compute weighted |slope| tests of r_struct against:
     - PV (if present)
     - PV uncertainty (if present)
     - redshift z (sanity)
     - κ at SN sky positions (if a HEALPix κ map is provided)
   Significance is calibrated using permutation nulls (shuffled x).

Inputs (expected relative to PROJECT_ROOT)
-----------------------------------------
Pantheon+SH0ES:
  data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat
  data/pantheonplus/redshift_pv/all_redshifts_PVs.csv

Optional:
  --bao-chain-dir   path to DESI BAO chain directory containing chain.*.txt and chain.updated.yaml
  --joint-bestfit   path to a bestfit.bestfit.txt with a commented header and a numeric row
  --kappa-map       HEALPix κ map (.fits readable by healpy, or .npy)
  --ned-csv         external geometry-only RA/Dec table (does not change μ or z)

Project root discovery
----------------------
PROJECT_ROOT is detected by searching upward from this file for a directory
containing both `data/` and `MATRIXDM/`. You may override detection by setting
one of: PHASE_OS_ROOT, PHASEOS_ROOT, PROJECT_ROOT.

Run
---
python3 MATRIXDM/dmhunt.py [options]

Outputs
-------
MATRIXDM/out/
  pantheon_dm_master.csv
  pantheon_dm_summary.json
  perm_*.npy
  diagnostic_residual_vs_proxies.png
  background_guardrail_effect.png

Reproducibility note
--------------------
All randomness is controlled by --seed. No model fitting to cosmological parameters is
performed; μ_ref is a fixed reference used only for residual definition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy import trapezoid as trapz
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Project root (never rely on CWD)
#
# This repo isn't strictly laid out everywhere, so compute
# PROJECT_ROOT robustly by searching upward for a directory that contains both
#   - data/
#   - MATRIXDM/
#
# You can override detection via env vars:
#   PHASE_OS_ROOT, PHASEOS_ROOT, or PROJECT_ROOT
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve()


def _detect_project_root(start: Path) -> Path:
    env_root = (
        os.environ.get("PHASE_OS_ROOT")
        or os.environ.get("PHASEOS_ROOT")
        or os.environ.get("PROJECT_ROOT")
    )
    if env_root:
        cand = Path(env_root).expanduser().resolve()
        if cand.exists():
            return cand

    for cand in (start,) + tuple(start.parents):
        if (cand / "data").is_dir() and (cand / "MATRIXDM").is_dir():
            return cand

    # Last-resort fallback: keep previous behavior, but make it explicit.
    return start.parents[2]


PROJECT_ROOT = _detect_project_root(HERE)

DATA_ROOT = PROJECT_ROOT / "data" / "pantheonplus"
DISTMOD_DIR = DATA_ROOT / "distance_moduli"
PV_DIR = DATA_ROOT / "redshift_pv"
DM_ROOT = PROJECT_ROOT / "MATRIXDM"
OUTDIR = DM_ROOT / "out"
OUTDIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Reference mu(z) (fixed reference; never refit during conditioning)
# -----------------------------------------------------------------------------

def mu_ref_flat_lcdm(z: np.ndarray, H0: float = 70.0, Om0: float = 0.3) -> np.ndarray:
    """
    Flat LCDM reference using simple trapezoid integration.
    Good enough for *residual definition*.
    """
    c_km_s = 299792.458
    z = np.asarray(z, dtype=float)
    z = np.clip(z, 0.0, None)

    mu = np.zeros_like(z, dtype=float)
    for i, zi in enumerate(z):
        if zi <= 0:
            mu[i] = np.nan
            continue
        n = 2048 if zi > 0.2 else 512
        grid = np.linspace(0.0, zi, n)
        Ez = np.sqrt(Om0 * (1.0 + grid) ** 3 + (1.0 - Om0))
        integ = trapz(1.0 / Ez, grid)
        dC = (c_km_s / H0) * integ  # Mpc
        dL = (1.0 + zi) * dC
        mu[i] = 5.0 * np.log10(dL) + 25.0
    return mu


# -----------------------------------------------------------------------------
# DESI BAO background-consistency guardrail helpers
# -----------------------------------------------------------------------------
from typing import Tuple
def load_bao_chain_cov(chain_dir: Path) -> Tuple[float, float, np.ndarray]:
    """
    Load DESI BAO chains and estimate mean/covariance for (H0, Omegam).

    Correctly handles iminuit-style chains where:
      - chain.*.txt files are numeric only
      - parameter ordering is defined in chain.updated.yaml
    """
    files = sorted(chain_dir.glob("chain.*.txt"))
    if not files:
        raise FileNotFoundError(f"No chain.*.txt files found in {chain_dir}")

    yaml_path = chain_dir / "chain.updated.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            "DESI BAO chains require chain.updated.yaml to map parameter ordering."
        )

    # --- load parameter ordering from YAML (Cobaya / iminuit semantics) ---
    try:
        import yaml  # type: ignore
        with open(yaml_path, "r") as f:
            y = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError("Failed to read chain.updated.yaml") from e

    if "params" not in y:
        raise KeyError("chain.updated.yaml missing top-level 'params' section")

    param_names = []
    for pname, pinfo in y["params"].items():
        if isinstance(pinfo, dict) and pinfo.get("drop", False):
            continue
        param_names.append(pname)

    if len(param_names) == 0:
        raise ValueError("No usable parameters found in chain.updated.yaml params section")

    # Identify indices
    def _find_idx(names, keys):
        for k in keys:
            for i, n in enumerate(names):
                if n.lower() == k.lower():
                    return i
        return None

    H0_idx = _find_idx(param_names, ["H0", "h0"])
    Om_idx = _find_idx(param_names, ["omegam", "omega_m", "Om0"])

    if H0_idx is None or Om_idx is None:
        raise KeyError(
            f"Could not locate H0 / omegam in parameter list: {param_names}\n"
            f"YAML–chain misalignment: resolved ordering = {param_names}"
        )

    # --- load numeric chains ---
    mats = []
    for f in files:
        df = pd.read_csv(f, sep=r"\s+", comment="#", header=None)
        mats.append(df.to_numpy(dtype=float))

    M = np.vstack(mats)
    if M.ndim != 2 or M.shape[1] <= max(H0_idx, Om_idx):
        raise ValueError(
            f"Chain matrix shape {M.shape} incompatible with parameter indices "
            f"H0_idx={H0_idx}, Om_idx={Om_idx}"
        )

    vals = M[:, [H0_idx, Om_idx]]
    mu = np.nanmean(vals, axis=0)
    cov = np.nan_to_num(np.cov(vals, rowvar=False))

    return float(mu[0]), float(mu[1]), cov


# -----------------------------------------------------------------------------
# DESI+Pantheon+Planck(+ACT) best-fit background loader
# -----------------------------------------------------------------------------
def load_joint_bestfit(path: Path) -> Tuple[float, float]:
    """
    Load a DESI+Pantheon+Planck(+ACT) iminuit best-fit file and extract (H0, omegam).

    These files store column names in a commented header line and a single numeric row.
    """
    if not path.exists():
        raise FileNotFoundError(f"Best-fit file not found: {path}")

    # Read raw lines
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Extract header from last comment line
    header = None
    for ln in lines:
        if ln.startswith("#"):
            header = ln.lstrip("#").strip()

    if header is None:
        raise KeyError("No commented header line found in best-fit file")

    colnames = header.split()

    # Extract first non-comment numeric line
    data_line = next(ln for ln in lines if not ln.startswith("#"))
    values = data_line.split()

    if len(values) != len(colnames):
        raise ValueError(
            f"Header/data length mismatch: {len(colnames)} cols vs {len(values)} values"
        )

    row = {colnames[i].lower(): float(values[i]) for i in range(len(colnames))}

    if "h0" not in row or "omegam" not in row:
        raise KeyError(
            f"Expected 'H0' and 'omegam' in best-fit header. "
            f"Found keys: {list(row.keys())}"
        )

    return float(row["h0"]), float(row["omegam"])


# -----------------------------------------------------------------------------
# Parsing loaders (robust to minor schema differences)
# -----------------------------------------------------------------------------

def load_pantheon_shoes_dat(path: Path) -> pd.DataFrame:
    """
    Pantheon_SH0ES.dat is typically whitespace-delimited with a header/comment lines.
    We parse with pandas in a tolerant way.

    Expected to contain at least:
      - some SN identifier column
      - z (zHD or zCMB or zHEL, etc.)
      - mu and muerr (or equivalent)
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing Pantheon distance moduli file: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        engine="python",
    )

    # normalize column names (lowercase)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _resolve_column(
    df: pd.DataFrame,
    candidates: List[str],
    contains_any: Optional[List[str]] = None,
) -> Optional[str]:
    """Resolve a column name case-insensitively with a small heuristic fallback."""
    norm_to_actual: Dict[str, str] = {}
    for c in df.columns:
        cs = str(c).strip()
        norm_to_actual[cs.lower()] = cs

    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_to_actual:
            return norm_to_actual[key]

    if contains_any:
        needles = [s.lower() for s in contains_any]
        hits = [c for c in df.columns if any(n in str(c).lower() for n in needles)]
        if len(hits) == 1:
            return str(hits[0]).strip()

    return None


def load_redshift_pv_csv(path: Path) -> pd.DataFrame:
    """
    all_redshifts_PVs.csv: expect an identifier + redshift variants and peculiar velocity info.

    We don’t assume exact columns; we’ll try to detect them.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing Pantheon redshift/PV file: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# -----------------------------------------------------------------------------
# Key matching / merging
# -----------------------------------------------------------------------------

def pick_best_key(left: pd.DataFrame, right: pd.DataFrame) -> Optional[str]:
    """
    Heuristic: find a common identifier column between the two frames.
    """
    candidates = [
        "CID", "cid",
        "SNID", "snid",
        "ID", "id",
        "NAME", "Name", "name",
        "SN", "sn",
        "objname", "OBJNAME",
    ]
    lset = set(left.columns)
    rset = set(right.columns)
    for c in candidates:
        if c in lset and c in rset:
            return c

    # fallback: any exact common column with mostly unique-ish values
    common = sorted(list(lset.intersection(rset)))
    for c in common:
        if left[c].nunique(dropna=True) > 0.5 * len(left) and right[c].nunique(dropna=True) > 0.5 * len(right):
            return c
    return None


def merge_pantheon_with_pv(
    dm_df: pd.DataFrame,
    pv_df: pd.DataFrame,
    strict: bool = False,
) -> pd.DataFrame:
    key = pick_best_key(dm_df, pv_df)
    if key is None:
        msg = (
            "Could not find a common SN identifier column between Pantheon_SH0ES.dat "
            "and all_redshifts_PVs.csv. Add/rename an ID column or pass --strict-join to fail."
        )
        if strict:
            raise ValueError(msg)
        # non-strict: return dm_df unchanged
        return dm_df.copy()

    # Ensure string key for robust join
    out = dm_df.copy()
    out[key] = out[key].astype(str)
    pv2 = pv_df.copy()
    pv2[key] = pv2[key].astype(str)

    merged = out.merge(pv2, on=key, how="left", suffixes=("", "_pv"))
    return merged


# -----------------------------------------------------------------------------
# Duplicate collapse (MATRIX invariants-preserving)
# -----------------------------------------------------------------------------
def collapse_duplicates(
    df: pd.DataFrame,
    id_col: Optional[str],
    z_col: str,
    r_struct_col: str,
    w_col: str,
) -> pd.DataFrame:
    """
    Collapse duplicate SN entries into a single representative row.

    MATRIX invariants preserved:
      - ordering is defined by z only (re-sorted after collapse)
      - structure is combined, not refit (operate on r_struct)
      - inverse-variance weighting is respected
      - no new scale modes are introduced

    Rule:
      - group by id_col (if None, return df unchanged)
      - z: weighted mean (weights = w)
      - r_struct: weighted mean (weights = w)
      - weight: sum of weights
      - all other columns: take first (metadata only)
    """
    if id_col is None or id_col not in df.columns:
        return df.copy()

    rows = []
    for _, g in df.groupby(id_col, sort=False):
        if len(g) == 1:
            rows.append(g.iloc[0])
            continue

        w = g[w_col].to_numpy(dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if np.sum(w) == 0:
            w = np.ones(len(g), dtype=float)

        z_bar = np.sum(w * g[z_col].to_numpy(dtype=float)) / np.sum(w)
        r_bar = np.sum(w * g[r_struct_col].to_numpy(dtype=float)) / np.sum(w)
        w_sum = float(np.sum(w))

        row = g.iloc[0].copy()
        row[z_col] = z_bar
        row[r_struct_col] = r_bar
        row[w_col] = w_sum
        rows.append(row)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out = out.sort_values(z_col).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Nuisance (scale-mode) projection
# -----------------------------------------------------------------------------

def zscore(v: np.ndarray) -> np.ndarray:
    m = np.nanmean(v)
    s = np.nanstd(v)
    if not np.isfinite(s) or s == 0:
        return v * 0.0
    return (v - m) / s


def build_design_matrix(df: pd.DataFrame, min_category_count: int = 20) -> Tuple[np.ndarray, List[str]]:
    """
    Build a conservative nuisance basis:
      - intercept
      - host mass (if present)
      - SALT2 x1, c (if present)
      - survey one-hot (if present)
    """
    X_parts: List[np.ndarray] = [np.ones((len(df), 1), dtype=float)]
    names: List[str] = ["intercept"]

    # numeric candidates
    for c in ["HOST_LOGMASS", "x1", "c"]:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            X_parts.append(zscore(v).reshape(-1, 1))
            names.append(f"z_{c}")

    # survey one-hot
    for sname in ["survey", "SURVEY", "Survey"]:
        if sname in df.columns:
            s = df[sname].astype(str).fillna("NA")
            vc = s.value_counts()
            keep = set(vc[vc >= min_category_count].index.tolist())
            s2 = s.apply(lambda x: x if x in keep else "OTHER")
            cats = sorted(s2.unique().tolist())
            if len(cats) > 1:
                ref = cats[0]
                for cat in cats[1:]:
                    X_parts.append((s2 == cat).to_numpy(dtype=float).reshape(-1, 1))
                    names.append(f"{sname}[{cat}]")
            break

    X = np.concatenate(X_parts, axis=1)
    return X, names


def weighted_least_squares(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    if X.size == 0:
        return np.zeros((0,), dtype=float)
    W = w.reshape(-1, 1)
    A = (X.T * W.T) @ X
    A = A + ridge * np.eye(A.shape[0])
    b = (X.T * W.T) @ y
    return np.linalg.solve(A, b)


# -----------------------------------------------------------------------------
# Dependence tests (permutation-calibrated)
# -----------------------------------------------------------------------------

def weighted_slope(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> float:
    mx = np.sum(w * x) / np.sum(w)
    my = np.sum(w * y) / np.sum(w)
    xc = x - mx
    yc = y - my
    denom = np.sum(w * xc * xc)
    if denom <= 0:
        return float("nan")
    return float(np.sum(w * xc * yc) / denom)


def perm_pvalue_abs_slope(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    n_perm: int,
    seed: int,
) -> Tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = abs(weighted_slope(y, x, w))
    stats = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        xp = x.copy()
        rng.shuffle(xp)
        stats[i] = abs(weighted_slope(y, xp, w))
    p = (1.0 + np.sum(stats >= obs)) / (n_perm + 1.0)
    return float(obs), float(p), stats


# -----------------------------------------------------------------------------
# Optional HEALPix map sampling
# -----------------------------------------------------------------------------

def load_healpix_map(path: Path, nside: Optional[int] = None) -> Tuple[np.ndarray, int]:
    ext = path.suffix.lower()
    if ext == ".npy":
        m = np.load(path)
        if nside is None:
            npix = int(m.shape[0])
            nside_inf = int(round(math.sqrt(npix / 12.0)))
            if 12 * nside_inf * nside_inf != npix:
                raise ValueError(f"Cannot infer NSIDE from npix={npix}; pass --kappa-nside.")
            nside = nside_inf
        return m.astype(float), int(nside)

    try:
        import healpy as hp  # type: ignore
    except Exception as e:
        raise RuntimeError("healpy is required to read FITS HEALPix maps. Install healpy or use .npy.") from e

    m = hp.read_map(str(path), verbose=False)
    nside2 = hp.get_nside(m)
    return np.asarray(m, dtype=float), int(nside2)


def sample_healpix_at_radec(m: np.ndarray, nside: int, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    try:
        import healpy as hp  # type: ignore
    except Exception as e:
        raise RuntimeError("healpy is required for HEALPix sampling. Install healpy.") from e

    ra = np.deg2rad(ra_deg.astype(float))
    dec = np.deg2rad(dec_deg.astype(float))
    theta = 0.5 * np.pi - dec
    phi = ra
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    return m[pix]


# -----------------------------------------------------------------------------
# NED geometry helpers
# -----------------------------------------------------------------------------

def _norm_sn_name(s: str) -> str:
    if s is None:
        return ""
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def load_ned_geometry_csv(
    path: Path,
    name_col: str,
    ra_col: str,
    dec_col: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"NED CSV not found: {path}")

    df = pd.read_csv(path)

    for c in (name_col, ra_col, dec_col):
        if c not in df.columns:
            raise KeyError(
                f"Missing column '{c}' in NED CSV. Available columns: {list(df.columns)}"
            )

    out = df[[name_col, ra_col, dec_col]].copy()
    out.columns = ["_ned_name", "_ned_ra", "_ned_dec"]

    out["_ned_key"] = out["_ned_name"].map(_norm_sn_name)
    out["_ned_ra"] = pd.to_numeric(out["_ned_ra"], errors="coerce")
    out["_ned_dec"] = pd.to_numeric(out["_ned_dec"], errors="coerce")

    out = out.dropna(subset=["_ned_ra", "_ned_dec"]).reset_index(drop=True)
    return out

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--H0-ref", type=float, default=70.0)
    ap.add_argument("--Om0-ref", type=float, default=0.3)
    ap.add_argument("--strict-join", action="store_true")

    ap.add_argument("--kappa-map", type=str, default=None, help="Optional HEALPix κ (mass proxy) map: .fits or .npy")
    ap.add_argument("--kappa-nside", type=int, default=None)
    ap.add_argument("--bao-chain-dir", type=str, default=None,
                    help="Optional DESI BAO chain directory (containing chain.*.txt) to define background guardrail")
    ap.add_argument("--bao-n-samples", type=int, default=25,
                    help="Number of background samples drawn from BAO chain for guardrail test")
    ap.add_argument(
        "--joint-bestfit",
        type=str,
        default=None,
        help="Optional DESI+Pantheon+Planck(+ACT) bestfit.bestfit.txt for hard background anchoring",
    )
    ap.add_argument(
        "--ned-csv",
        type=str,
        default=None,
        help="Optional NED object CSV providing RA/DEC to override geometry (geometry-only).",
    )
    ap.add_argument(
        "--ned-name-col",
        type=str,
        default="Object Name",
        help="Column name in NED CSV for object identifier.",
    )
    ap.add_argument(
        "--ned-ra-col",
        type=str,
        default="RA",
        help="Column name in NED CSV for RA in degrees.",
    )
    ap.add_argument(
        "--ned-dec-col",
        type=str,
        default="Dec",
        help="Column name in NED CSV for Dec in degrees.",
    )

    args = ap.parse_args()

    # Resolve input paths from your hierarchy
    dm_path = DISTMOD_DIR / "Pantheon_SH0ES.dat"
    pv_path = PV_DIR / "all_redshifts_PVs.csv"

    dm_df = load_pantheon_shoes_dat(dm_path)
    pv_df = load_redshift_pv_csv(pv_path)
    df = merge_pantheon_with_pv(dm_df, pv_df, strict=args.strict_join)
    # Capture SN identifier column explicitly
    ID_COL = pick_best_key(dm_df, pv_df)

    # Optional NED geometry override (geometry-only)
    if args.ned_csv is not None:
        ned_df = load_ned_geometry_csv(
            Path(args.ned_csv),
            name_col=args.ned_name_col,
            ra_col=args.ned_ra_col,
            dec_col=args.ned_dec_col,
        )

        # Identify a Pantheon-side identifier for name matching
        pantheon_id_cols = [c for c in ["SNID", "IAUC", "Name", "NAME", "CID", "ID"] if c in df.columns]
        if not pantheon_id_cols:
            raise KeyError(
                "No usable identifier column found in Pantheon data to join NED geometry."
            )

        pid = pantheon_id_cols[0]
        df["_ned_key"] = df[pid].astype(str).map(_norm_sn_name)

        df = df.merge(
            ned_df[["_ned_key", "_ned_ra", "_ned_dec"]],
            on="_ned_key",
            how="left",
        )

        # Override geometry only (never μ or z)
        if "RA" in df.columns and "DEC" in df.columns:
            df["RA"] = pd.to_numeric(df["RA"], errors="coerce")
            df["DEC"] = pd.to_numeric(df["DEC"], errors="coerce")
            df["RA"] = df["RA"].where(np.isfinite(df["RA"]), df["_ned_ra"])
            df["DEC"] = df["DEC"].where(np.isfinite(df["DEC"]), df["_ned_dec"])
        else:
            df["RA"] = df["_ned_ra"]
            df["DEC"] = df["_ned_dec"]

        df["sky_source"] = np.where(
            np.isfinite(df["_ned_ra"]) & np.isfinite(df["_ned_dec"]),
            "NED",
            "Pantheon",
        )

    # Resolve key columns (Pantheon schema varies by release / processing)
    Z_COL = _resolve_column(
        df,
        candidates=["zCMB", "zHD", "zHEL", "z", "Z"],
        contains_any=["zcmb", "zhd", "zhel", "z"],
    )
    MU_COL = _resolve_column(
        df,
        candidates=["MU_SH0ES", "MU_SHOES", "MU", "mu", "distmod", "distance_modulus"],
        contains_any=["mu"],
    )
    MUERR_COL = _resolve_column(
        df,
        candidates=[
            "MU_SH0ES_ERR_DIAG",
            "MU_SH0ES_ERR",
            "MUERR_SHOES",
            "MUERR",
            "muerr",
            "MU_ERR",
            "mu_err",
            "dmu",
            "sigma_mu",
        ],
        contains_any=["muerr", "mu_err", "dmu", "sigma"],
    )

    # Optional (used only if you run κ-map tests)
    RA_COL = _resolve_column(df, candidates=["RA"], contains_any=["ra"])
    DEC_COL = _resolve_column(df, candidates=["DEC"], contains_any=["dec"])

    missing = [name for name, col in [("z", Z_COL), ("mu", MU_COL), ("muerr", MUERR_COL)] if col is None]
    if missing:
        cols_preview = ", ".join(map(str, list(df.columns)[:40]))
        more = "" if len(df.columns) <= 40 else f" (+{len(df.columns) - 40} more)"
        raise KeyError(
            f"Missing required column(s) {missing} in {dm_path.name}. "
            f"Available columns: {cols_preview}{more}"
        )

    z = pd.to_numeric(df[Z_COL], errors="coerce").to_numpy(dtype=float)  # type: ignore[index]
    mu = pd.to_numeric(df[MU_COL], errors="coerce").to_numpy(dtype=float)  # type: ignore[index]
    muerr = pd.to_numeric(df[MUERR_COL], errors="coerce").to_numpy(dtype=float)  # type: ignore[index]

    # Sort by redshift before all downstream analysis
    order = np.argsort(z)
    z = z[order]
    mu = mu[order]
    muerr = muerr[order]
    df = df.iloc[order].reset_index(drop=True)

    good = np.isfinite(z) & np.isfinite(mu) & np.isfinite(muerr) & (muerr > 0)
    df = df.loc[good].reset_index(drop=True)
    z = z[good]; mu = mu[good]; muerr = muerr[good]
    w = (muerr ** -2).astype(float)

    # Residuals vs fixed reference
    mu_ref = mu_ref_flat_lcdm(z, H0=args.H0_ref, Om0=args.Om0_ref)
    r = mu - mu_ref

    # Remove scale modes
    X, feat_names = build_design_matrix(df)
    beta = weighted_least_squares(X, r, w)
    r_scale = X @ beta
    r_struct = r - r_scale

    # Attach weights for collapse
    df["w_invvar"] = w
    df["r_struct"] = r_struct
    df["z_used"] = z

    # Collapse duplicate SN entries (preserves MATRIX invariants)
    df = collapse_duplicates(
        df,
        id_col=ID_COL,
        z_col="z_used",
        r_struct_col="r_struct",
        w_col="w_invvar",
    )

    # Recompute reference and residuals on the collapsed set
    z = df["z_used"].to_numpy(dtype=float)
    w = df["w_invvar"].to_numpy(dtype=float)

    mu_ref_collapsed = mu_ref_flat_lcdm(z, H0=args.H0_ref, Om0=args.Om0_ref)

    # r_struct is already collapsed correctly; scale and raw residuals
    # are no longer meaningful per-row, so keep them aligned by recomputing
    # r_mu as (r_struct + r_scale_collapsed), with r_scale collapsed to zero.
    r_struct = df["r_struct"].to_numpy(dtype=float)
    r_scale_collapsed = np.zeros_like(r_struct)
    r_mu_collapsed = r_struct + r_scale_collapsed

    # --- BAO background consistency guardrail ---
    guardrail = None
    if args.bao_chain_dir is not None:
        chain_dir = Path(args.bao_chain_dir)
        H0_mu, Om_mu, cov = load_bao_chain_cov(chain_dir)

        rng = np.random.default_rng(args.seed + 99)
        samples = rng.multivariate_normal(
            mean=np.array([H0_mu, Om_mu], dtype=float),
            cov=cov,
            size=int(args.bao_n_samples),
        )

        # Measure how much r_struct changes under BAO-allowed backgrounds on the collapsed dataset
        deltas = []
        for H0_s, Om_s in samples:
            # Guardrail must operate on the collapsed dataset only
            mu_ref_s = mu_ref_flat_lcdm(z, H0=H0_s, Om0=Om_s)

            # r_struct was already defined on the collapsed set with scale removed.
            # Under a background shift, the only allowed change is a uniform
            # redefinition of mu_ref(z), so the structural delta is:
            delta_struct = mu_ref_collapsed - mu_ref_s

            deltas.append(float(np.nanstd(delta_struct)))

        guardrail = {
            "H0_mean": H0_mu,
            "Om0_mean": Om_mu,
            "H0_Om0_cov": cov.tolist(),
            "r_struct_delta_std_mean": float(np.mean(deltas)),
            "r_struct_delta_std_max": float(np.max(deltas)),
            "n_samples": int(len(deltas)),
        }

    # --- Joint (DESI+Pantheon+Planck+ACT) hard background anchor ---
    joint_guardrail = None
    if args.joint_bestfit is not None:
        bf_path = Path(args.joint_bestfit)
        H0_j, Om_j = load_joint_bestfit(bf_path)

        # Reference under joint best-fit
        mu_ref_joint = mu_ref_flat_lcdm(z, H0=H0_j, Om0=Om_j)

        # Structural delta relative to baseline reference
        delta_joint = mu_ref_collapsed - mu_ref_joint

        joint_guardrail = {
            "H0_joint": H0_j,
            "Om0_joint": Om_j,
            "delta_mu_std": float(np.nanstd(delta_joint)),
            "delta_mu_max": float(np.nanmax(np.abs(delta_joint))),
        }

    # Prepare output table
    df_out = df.copy()
    df_out["mu_ref"] = mu_ref_collapsed
    df_out["r_mu"] = r_mu_collapsed
    df_out["r_scale"] = r_scale_collapsed
    df_out["r_struct"] = r_struct

    # --- DM-adjacent proxy tests ---
    tests: Dict[str, Dict[str, float]] = {}

    # 1) Peculiar velocity proxy (if present)
    PV_COL = None
    for cand in ["PV", "VPEC", "vpec", "pv"]:
        if cand in df_out.columns:
            PV_COL = cand
            break

    PVERR_COL = None
    for cand in ["vpecerr", "PVERR", "VPECERR", "pverr"]:
        if cand in df_out.columns:
            PVERR_COL = cand
            break

    if PV_COL is not None:
        pv = pd.to_numeric(df_out[PV_COL], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(pv)
        if np.sum(m) >= 20:
            obs, p, null = perm_pvalue_abs_slope(r_struct[m], pv[m], w[m], n_perm=args.n_perm, seed=args.seed + 1)
            tests["r_struct_vs_pv"] = {"abs_slope": obs, "p_perm": p, "n": float(np.sum(m))}
            np.save(OUTDIR / "perm_rstruct_vs_pv_abs_slope.npy", null)
            df_out["pv_used"] = pv
        else:
            tests["r_struct_vs_pv"] = {"abs_slope": float("nan"), "p_perm": float("nan"), "n": float(np.sum(m))}
    else:
        tests["r_struct_vs_pv"] = {"abs_slope": float("nan"), "p_perm": float("nan"), "n": 0.0}

    # 2) PV uncertainty proxy (often tracks local flow / environment + selection)
    if PVERR_COL is not None:
        pverr = pd.to_numeric(df_out[PVERR_COL], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(pverr)
        if np.sum(m) >= 20:
            obs, p, null = perm_pvalue_abs_slope(r_struct[m], pverr[m], w[m], n_perm=args.n_perm, seed=args.seed + 2)
            tests["r_struct_vs_pverr"] = {"abs_slope": obs, "p_perm": p, "n": float(np.sum(m))}
            np.save(OUTDIR / "perm_rstruct_vs_pverr_abs_slope.npy", null)
            df_out["pverr_used"] = pverr
        else:
            tests["r_struct_vs_pverr"] = {"abs_slope": float("nan"), "p_perm": float("nan"), "n": float(np.sum(m))}
    else:
        tests["r_struct_vs_pverr"] = {"abs_slope": float("nan"), "p_perm": float("nan"), "n": 0.0}

    # 3) Optional κ map (CMB lensing / mass proxy)
    if args.kappa_map is not None:
        if RA_COL not in df_out.columns or DEC_COL not in df_out.columns:
            tests["r_struct_vs_kappa"] = {"abs_slope": float("nan"), "p_perm": float("nan"), "n": 0.0}
        else:
            kpath = Path(args.kappa_map)
            mapp, nside = load_healpix_map(kpath, nside=args.kappa_nside)
            ra = pd.to_numeric(df_out[RA_COL], errors="coerce").to_numpy(dtype=float)
            dec = pd.to_numeric(df_out[DEC_COL], errors="coerce").to_numpy(dtype=float)
            mm = np.isfinite(ra) & np.isfinite(dec)
            kap = np.full(len(df_out), np.nan, dtype=float)
            kap[mm] = sample_healpix_at_radec(mapp, nside, ra[mm], dec[mm])

            m = np.isfinite(kap)
            if np.sum(m) >= 20:
                obs, p, null = perm_pvalue_abs_slope(r_struct[m], kap[m], w[m], n_perm=args.n_perm, seed=args.seed + 3)
                tests["r_struct_vs_kappa"] = {"abs_slope": obs, "p_perm": p, "n": float(np.sum(m))}
                np.save(OUTDIR / "perm_rstruct_vs_kappa_abs_slope.npy", null)
                df_out["kappa_used"] = kap
            else:
                tests["r_struct_vs_kappa"] = {"abs_slope": float("nan"), "p_perm": float("nan"), "n": float(np.sum(m))}
    else:
        tests["r_struct_vs_kappa"] = {"abs_slope": float("nan"), "p_perm": float("nan"), "n": 0.0}

    # 4) Sanity: residual structure vs redshift (should usually show structure)
    obs, p, null = perm_pvalue_abs_slope(r_struct, z, w, n_perm=args.n_perm, seed=args.seed + 4)
    tests["r_struct_vs_z"] = {"abs_slope": obs, "p_perm": p, "n": float(len(z))}
    np.save(OUTDIR / "perm_rstruct_vs_z_abs_slope.npy", null)

    # --- Diagnostic image: residual structure vs proxies with null envelopes ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    def _plot_panel(ax, x, y, w, title, stat):
        if x is None or np.sum(np.isfinite(x)) < 20:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center")
            ax.set_title(title)
            return
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=15, alpha=0.6)
        slope = weighted_slope(y[m], x[m], w[m])
        xx = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 100)
        yy = slope * (xx - np.average(x[m], weights=w[m])) + np.average(y[m], weights=w[m])
        ax.plot(xx, yy)
        ax.set_title(f"{title}\n|slope|={stat['abs_slope']:.3g}, p={stat['p_perm']:.3g}")

    _plot_panel(axes[0], z, r_struct, w, "r_struct vs z", tests["r_struct_vs_z"])

    pv = df_out[PV_COL].to_numpy(dtype=float) if PV_COL else None
    _plot_panel(axes[1], pv, r_struct, w, "r_struct vs PV", tests["r_struct_vs_pv"])

    pverr = df_out[PVERR_COL].to_numpy(dtype=float) if PVERR_COL else None
    _plot_panel(axes[2], pverr, r_struct, w, "r_struct vs PV error", tests["r_struct_vs_pverr"])

    # κ sky coverage diagnostic instead of correlation plot
    ax = axes[3]
    ax.clear()

    if args.kappa_map is not None and RA_COL in df_out.columns and DEC_COL in df_out.columns:
        ra = pd.to_numeric(df_out[RA_COL], errors="coerce").to_numpy(dtype=float)
        dec = pd.to_numeric(df_out[DEC_COL], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(ra) & np.isfinite(dec)

        ax.scatter(
            ra[m], dec[m],
            s=20,
            alpha=0.7,
        )
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.set_title(
            f"SN sky positions (κ overlap)\n"
            f"n={np.sum(m)} usable"
        )
    else:
        ax.text(0.5, 0.5, "no κ geometry", ha="center", va="center")
        ax.set_title("κ sky coverage")

    plt.tight_layout()
    plt.savefig(OUTDIR / "diagnostic_residual_vs_proxies.png", dpi=150)
    plt.close(fig)

    # --- Background guardrail visualization (normalized, quantile-aware) ---
    if guardrail is not None:
        # Normalize by observed structural scatter
        obs_struct_std = float(np.nanstd(r_struct))
        deltas = np.asarray(deltas, dtype=float) / obs_struct_std

        q16, q50, q84, q95 = np.percentile(deltas, [16, 50, 84, 95])

        # Ensure visually separable reference lines even under extreme clustering
        eps = 0.02 * q50 if q50 > 0 else 1e-4
        q95_plot = q95 + eps

        # Explicit x-range so tightly clustered distributions are visible
        xmax = float(np.max(deltas))

        joint_norm = None
        if joint_guardrail is not None:
            joint_norm = joint_guardrail["delta_mu_std"] / obs_struct_std
            xmax = max(xmax, joint_norm)

        xmax = 1.2 * xmax if xmax > 0 else 1e-3

        fig = plt.figure(figsize=(6, 4))
        plt.hist(
            deltas,
            bins=15,
            range=(0, xmax),
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            zorder=1,
            label="BAO-consistent backgrounds",
        )

        plt.axvline(
            q50,
            color="black",
            linestyle="-",
            linewidth=2.5,
            label="median",
            zorder=5,
        )
        plt.axvline(
            q95_plot,
            color="black",
            linestyle=(0, (3, 3)),
            linewidth=2,
            label="95% quantile",
        )

        if joint_guardrail is not None:
            joint_norm = joint_guardrail["delta_mu_std"] / obs_struct_std
            plt.plot(
                joint_norm, 0,
                marker="v",
                markersize=8,
                color="darkred",
                label="joint best-fit",
            )

        plt.xlim(0, xmax)
        plt.xlabel("fraction of observed structure explained")
        plt.ylabel("count")
        plt.title("Background-consistency guardrail")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(OUTDIR / "background_guardrail_effect.png", dpi=150)
        plt.close(fig)

    # Save artifacts
    out_csv = OUTDIR / "pantheon_dm_master.csv"
    df_out.to_csv(out_csv, index=False)

    summary = {
        "project_root": str(PROJECT_ROOT),
        "inputs": {
            "pantheon_dm": str(dm_path),
            "pantheon_pv": str(pv_path),
            "kappa_map": args.kappa_map,
            "ned_csv": args.ned_csv,
        },
        "n_used": int(len(df_out)),
        "reference": {"H0_ref": float(args.H0_ref), "Om0_ref": float(args.Om0_ref)},
        "nuisance_features": feat_names,
        "beta": [float(x) for x in beta.tolist()],
        "duplicate_collapse": {
            "id_column": ID_COL,
            "rule": "inverse-variance weighted mean on r_struct and z",
        },
        "tests": tests,
        "outputs": {
            "table": str(out_csv),
            "outdir": str(OUTDIR),
            "images": [
                "diagnostic_residual_vs_proxies.png",
                "background_guardrail_effect.png"
            ]
        },
        "value_column": MU_COL,
        "redshift_column": Z_COL,
        "bao_guardrail": guardrail,
        "joint_background_guardrail": joint_guardrail,
        "sky_geometry": {
            "source_column": "sky_source" if "sky_source" in df_out.columns else None
        },
    }
    out_json = OUTDIR / "pantheon_dm_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("MATRIXDM Pantheon DM-trace run complete")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  wrote: {out_csv}")
    print(f"  wrote: {out_json}")
    print("  tests:")
    for k, v in tests.items():
        print(f"    - {k}: abs_slope={v['abs_slope']:.4g} p={v['p_perm']:.4g} n={int(v['n'])}")


if __name__ == "__main__":
    main()
# Background-Consistent Residual Structure in Type Ia Supernovae

This repository contains the minimal, paper-facing analysis code used to generate the figures and quantitative claims in the manuscript:

Background-Consistent Residual Structure in Type Ia Supernovae  
Kevin Shepheard (2026)  
Independent Researcher

DOI: https://doi.org/10.5281/zenodo.18167688

The purpose of this code is not to provide a general cosmological inference framework, but to cleanly, conservatively, and transparently demonstrate that supernova residual structure persists under tightly constrained background cosmologies.

---

## Scope and intent

This repository is designed to answer one narrow question:

Can the observed residual structure in Type Ia supernova distances be explained by allowed background cosmological freedom?

Accordingly, the code:

- Enforces strict background consistency using BAO and joint CMB constraints
- Separates background-induced effects from ordering-dependent residual structure
- Produces only the artifacts required to support the paper’s claims

---

## What the code does

The analysis pipeline:

- Loads the Pantheon+SH0ES Type Ia supernova dataset
- Constructs a fixed reference distance–redshift relation μ_ref(z)
- Computes Hubble residuals without refitting the background
- Projects out low-rank calibration (scale) modes:
  - host-galaxy stellar mass
  - light-curve stretch (x1)
  - color (c)
  - survey terms (if present)
- Defines an ordering-dependent residual structure statistic
- Constructs a background-consistency guardrail using:
  - DESI BAO likelihood chains
  - optional joint BAO+CMB best-fit cosmologies
- Tests correlations of residual structure against:
  - redshift
  - peculiar velocity and PV uncertainty
  - sky geometry and optional CMB lensing convergence (κ)
- Produces two diagnostic figures used directly in the paper:
  - residual structure vs environmental proxies
  - background-consistency guardrail visualization

All outputs are written to a single deterministic run directory.

---

## What the code does not do

- It does not fit or optimize cosmological models
- It does not infer a new value of H0
- It does not attribute residual structure to a specific physical mechanism
- It does not modify the background expansion history

The analysis is strictly empirical and diagnostic.

---

## Requirements

- Python 3.9 or newer
- NumPy
- Pandas
- Matplotlib
- PyYAML (for DESI chain parsing)

Optional:
- healpy (only if using CMB lensing κ maps)

---

## Running the analysis

Example invocation:

python MATRIXDM/dmhunt.py \
  --kappa-map data/COM_Lensing_4096_R3.00/kappa_mv_map.fits \
  --kappa-nside 2048 \
  --bao-chain-dir data/desi/bao_cosmo_params \
  --bao-n-samples 500 \
  --joint-bestfit data/dr6-lensing/bestfit.bestfit.txt

This produces:

- pantheon_dm_master.csv
- pantheon_dm_summary.json
- diagnostic_residual_vs_proxies.png
- background_guardrail_effect.png
- permutation null distributions (perm_*.npy)

---

## Reproducibility notes

- All random processes use fixed seeds
- The reference μ_ref(z) is never refit
- Duplicate supernova entries are collapsed using invariant-preserving rules
- Permutation tests preserve marginal distributions while destroying ordering
- The pipeline is deterministic given the stated inputs

These design choices are intentional and documented in the paper.

---

## License

This repository is released under the MIT License.

You are free to use, modify, and redistribute the code with attribution.  
No warranty is provided.

---

## Citation

If you use or reference this code, please cite:

Shepheard, K. (2026). Background-Consistent Residual Structure in Type Ia Supernovae (Version 1.0.0).  
Zenodo. https://doi.org/10.5281/zenodo.18167688

---

## Contact

Questions, critiques, and replication attempts are welcome via GitHub issues.
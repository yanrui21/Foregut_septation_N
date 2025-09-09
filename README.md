# Convergent Flow–Mediated Mesenchymal Force in Foregut Splitting
Code & notebooks accompanying: **Yan et al.** “Convergent flow-mediated mesenchymal force drives embryonic foregut constriction and splitting.” *bioRxiv* (2025).

> This repository organizes the analysis used in the study into four task-focused folders with example inputs to help you validate your setup.

## Repository structure
- **Foregut_Epithelium_Segmentation/** — Jupyter notebook using **Segment Anything Model 2 (SAM2)** to segment the foregut epithelium from surrounding mesenchyme. Includes example inputs.
- **PIV_strain_rate/** — Jupyter notebook that post-processes **PIVLab** velocity fields to compute strain rate and divergence maps. Includes example inputs.
- **Single-cell-tracking-and-trajectory-analysis/** — MATLAB utilities for TrackMate-exported trajectories: directionality, persistence, MSD with power-law fitting, and **vbSPT** state inference. Includes an example input.
- **Golgi_Nuclei_Vectors/** — MATLAB utilities to compute nucleus→Golgi vectors and angle statistics (with a user-defined 0° reference) from dual-labeled, TrackMate-segmented data. Includes example inputs.

---

## Environments & dependencies

### Python (for Jupyter notebooks)
- Python ≥ 3.10, JupyterLab/Notebook
- Packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `opencv-python`, `tifffile`
- **Segmentation notebook:** install **SAM2** (and PyTorch + CUDA if available)

Quickstart:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -U pip wheel
pip install numpy pandas matplotlib scipy opencv-python tifffile jupyter
# Follow SAM2 install instructions and verify its demo runs before using our notebook


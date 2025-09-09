# Convergent Flow–Mediated Mesenchymal Force in Foregut Splitting
Code & notebooks accompanying: **Yan et al.** “Convergent flow-mediated mesenchymal force drives embryonic foregut constriction and splitting.” *bioRxiv* (2025).

> This repository organizes the analysis used in the study into four task-focused folders with example inputs to help you validate your setup.

## Repository structure
- **Foregut_Epithelium_Segmentation/** — Jupyter notebook using **Segment Anything Model 2 (SAM2)** to segment the foregut epithelium from surrounding mesenchyme. Includes example inputs. Contributed by Deng Li.
- **PIV_strain_rate/** — Jupyter notebook that post-processes **PIVLab** velocity fields to compute strain rate and divergence maps. Includes example inputs. Contributed by Panos Oikonomou.
- **Single-cell-tracking-and-trajectory-analysis/** — MATLAB utilities for TrackMate-exported trajectories: directionality, persistence, MSD with power-law fitting, and **vbSPT** state inference. Includes an example input. Contributed by Rui Yan.
- **Golgi_Nuclei_Vectors/** — MATLAB utilities to compute nucleus→Golgi vectors and angle statistics (with a user-defined 0° reference) from dual-labeled, TrackMate-segmented data. Includes example inputs. Contributed by Rui Yan.

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

@article{Yan2025ForegutSplitting,
  author  = {Yan, Rui and Hoffmann, Ludwig A. and Oikonomou, Panagiotis and Li, Deng and Lee, ChangHee and Gill, Hasreet and Mongera, Alessandro and Nerurkar, Nandan L. and Mahadevan, L. and Tabin, Clifford J.},
  title   = {Convergent flow-mediated mesenchymal force drives embryonic foregut constriction and splitting},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.01.22.634318},
  url     = {https://www.biorxiv.org/content/10.1101/2025.01.22.634318v2}
}
MIT License

Copyright (c) 2025 Rui Yan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

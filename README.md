# Med-StyleGAN: Plastic Surgery Simulator with Anthropometric Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20GPU-20BEFF?logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-Research%20Only-lightgrey)

A deep learning pipeline that simulates plastic surgery outcomes using **GAN inversion** and quantifies facial changes with **16 clinical anthropometric metrics** — all within a live interactive dashboard.

---

## How It Works

```
Input Face Image
      │
      ▼
Face Detection (YOLOv8 → dlib HOG fallback)
      │
      ▼
FFHQ Alignment (dlib 68-pt landmarks → 1024×1024 canonical crop)
      │
      ▼
e4e Encoder → 18×512 W+ Latent Code   ←── GAN inversion
      │
      ▼
GANSpace PCA Directions + Surgical Sliders   ←── latent-space editing
      │
      ▼
StyleGAN2-FFHQ Generator → Simulated Post-Op Image
      │
      ▼
MediaPipe FaceMesh (468 landmarks) → 16 Clinical Metrics + Report
```

---

## Features

- **Interactive surgical controls** — 10 sliders (Nose Bridge, Jaw Width, Chin Projection, Eye Aperture, Lip Fullness, etc.) that manipulate GAN latent directions in real time
- **GAN inversion** via the [e4e encoder](https://github.com/omertov/encoder4editing) — encodes any real face into the StyleGAN2 W+ space
- **Clinical anthropometric report** — computes 16 normalized metrics including:
  - Nasolabial Angle, Alar Base Width, Jaw Width / Face Width ratio
  - Golden Ratio (W/L), Intercanthal Distance, Eye Aperture (L & R)
  - Lip Height & Width, Chin Height, Cheek Width, Face Length
- **Dataset-level analysis** — batch-processes 662 real pre/post-op pairs and aggregates per-metric change statistics
- **3-panel medical report** — side-by-side mesh overlays + Δ bar chart with percentage annotations, auto-saved as PNG
- **Robust fallbacks** — YOLOv8 → dlib HOG face detection, synthetic PCA basis if GANSpace download fails, stub modes when weights are missing

---

## Tech Stack

| Component | Technology |
|---|---|
| GAN Generator | [StyleGAN2-ADA PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch) (FFHQ, 1024×1024) |
| GAN Encoder | [encoder4editing (e4e)](https://github.com/omertov/encoder4editing) |
| Latent Directions | [GANSpace](https://github.com/harskish/ganspace) PCA directions |
| Face Detection | YOLOv8n-face (Ultralytics) + dlib HOG fallback |
| Face Alignment | dlib 68-point shape predictor (FFHQ protocol) |
| Landmark Analysis | MediaPipe FaceMesh (468 landmarks) |
| Framework | PyTorch 2.x, CUDA, NumPy, OpenCV, SciPy |
| UI | ipywidgets (interactive Jupyter dashboard) |
| Dataset | 662 pre/post-op facial image pairs |

---

## Quick Start (Kaggle)

1. Open the notebook on [Kaggle](https://www.kaggle.com/)
2. **Settings → Accelerator → GPU T4 x2** (required for StyleGAN2 synthesis)
3. **Settings → Internet → ON** (required for weight downloads)
4. Attach your own dataset of pre/post-op image pairs (see [Dataset](#-dataset) section below)
5. Update `DATASET_ROOT` in **Block 0** to match your dataset path
6. Run cells **top to bottom** — Block 1 installs dependencies (run once), Blocks 2–8 are the live pipeline

> Block 1 (environment setup) takes ~2–3 minutes on first run. After it completes, do **Run → Restart & Clear Output**, then re-run all cells.

---

## Dataset

> ** The dataset used in development (`yaswithanandu/hda-data-set`) is a private Kaggle dataset and is not publicly available. No images from this dataset are included in this repository.**

This project was developed with access to the **HDA Plastic Surgery dataset** — 662 matched pre/post-operative facial image pairs. To run the notebook with your own data, prepare a folder of paired images following this naming convention:

```
your-dataset/
└── flattened_images/
    ├── 001_a.jpg   ← pre-operative
    ├── 001_b.jpg   ← post-operative
    ├── 002_a.jpg
    ├── 002_b.jpg
    └── ...
```

Then update **Block 0** of the notebook:
```python
DATASET_ROOT = "/kaggle/input/your-dataset-slug/flattened_images"
PRE_SUFFIX   = "_a"   # change if your files use _before, _pre, etc.
POST_SUFFIX  = "_b"   # change if your files use _after, _post, etc.
```

- Dataset-level analysis is run on up to 200 pairs (configurable via `MAX_PAIRS`)

---

## Project Structure

```
design-plastic-preview.ipynb
│
├── Block 0  — User config: dataset paths, suffixes, output dirs
├── Block 1  — Environment setup (pip installs, git clones, weight downloads)
├── Block 2  — Core imports + GPU check
├── Block 3  — Identity encoding pipeline
│              ├── 3-A: FaceDetector (YOLOv8 / dlib HOG)
│              ├── 3-B: FaceAligner (FFHQ canonical alignment)
│              ├── 3-C: E4EEncoder (image → 18×512 W+ latent)
│              └── 3-D: StyleGAN2Generator (W+ → 1024×1024 image)
├── Block 4  — Surgical controls (GANSpace directions + apply_surgical_edits)
├── Block 5  — Anthropometric measurement (MediaPipe 468 landmarks)
│              ├── 5-A: AnthropometricAnalyzer (16 clinical metrics)
│              └── 5-B: render_medical_report (3-panel figure)
├── Block 6  — Live interactive surgical dashboard (ipywidgets)
├── Block 7  — Dataset-level before/after analysis (662 pairs)
└── Block 8  — Demo: single-pair end-to-end report
```

---

## Anthropometric Metrics

All distances are normalized by **Inter-Ocular Distance (IOD)** for scale invariance:

| Metric | Clinical Relevance |
|---|---|
| Nose Length | Rhinoplasty planning |
| Nose Tip Protrusion | Tip projection assessment |
| Alar Base Width | Alar base reduction |
| Nasolabial Angle° | Tip rotation (ideal: 90–115°) |
| Jaw Width | Mandibular contouring |
| Face Width | Overall facial proportion |
| Chin Height | Genioplasty planning |
| Chin Projection Δ | Sagittal chin position |
| L/R Eye Aperture | Blepharoplasty assessment |
| Intercanthal Distance | Orbital proportion |
| Lip Height & Width | Lip augmentation planning |
| Face Length | Vertical proportion |
| Cheek Width | Malar augmentation |
| Golden Ratio (W/L) | Aesthetic proportion index |

---

## Key Design Decisions

- **W+ latent space** (vs. Z or W): W+ gives per-layer control, enabling localized edits (e.g., only affecting mid-face layers for nose edits)
- **IOD normalization**: makes metrics comparable across different image scales and head sizes
- **Numpy version pinning**: Kaggle's scipy/jax/cupy are compiled against a specific numpy ABI — a constraints file prevents silent breakage during installs
- **Graceful degradation**: every component has a stub/fallback mode so partial results are always produced even if individual models fail to load

---

## Current Implementation vs. Full Research Design

This notebook is a **working prototype** built with transfer learning from pre-trained models. The full research design (described in the project presentation) envisions a supervised training pipeline on the HDA dataset from scratch. The table below documents what is implemented today vs. what the complete system would include.

| Feature | This Prototype  | Full Design |
|---|---|---|
| **Encoder** | Pre-trained e4e (pSp-based) | Custom Hybrid Encoder (pSp + PlasticGAN residuals) |
| **Latent editing** | GANSpace PCA directions (unsupervised) | Learned Latent Mapper trained on HDA pre/post pairs |
| **Surgical controls** | 10 sliders | 50 sliders (full anatomical coverage) |
| **Metric extraction** | MediaPipe post-generation (external) | Latent Regression Head (metric-supervised, baked into W+ space) |
| **Training losses** | None — uses pre-trained weights | Semantic Loss (L_semantic) + Disentanglement Loss (L_disent) |
| **Safety validation** | None | Bidirectional encoder-decoder check; rejects edits with >5% reconstruction error |
| **Metric units** | Pixel-normalized (IOD-relative) | Physical mm-level (calibrated to real anatomy) |

> **Why this approach?** Training the full pipeline requires paired supervision on the HDA dataset and significant GPU-days. The prototype demonstrates that the core architecture — GAN inversion → latent arithmetic → clinical measurement — is sound and produces clinically relevant outputs using transfer learning. The GANSpace directions, while unsupervised, effectively encode semantically meaningful facial attributes that map naturally to surgical parameters.

---

## Dataset Citation

This project uses the **HDA Plastic Surgery dataset**. If you use this work, please cite:

> Christian Rathgeb, Didem Dogan, Fabian Stockhardt, Maria De Marsico, Christoph Busch,
> *"Plastic Surgery: An Obstacle for Deep Face Recognition?"*,
> in **15th IEEE Computer Society Workshop on Biometrics (CVPRW)**, pp. 3510–3517, 2020.

```bibtex
@inproceedings{rathgeb2020plasticsurgery,
  author    = {Rathgeb, Christian and Dogan, Didem and Stockhardt, Fabian and De Marsico, Maria and Busch, Christoph},
  title     = {Plastic Surgery: An Obstacle for Deep Face Recognition?},
  booktitle = {15th IEEE Computer Society Workshop on Biometrics (CVPRW)},
  pages     = {3510--3517},
  year      = {2020}
}
```

---

## 📄 License

This project is for research and educational purposes only. StyleGAN2 weights are subject to [NVIDIA's research license](https://nvlabs.github.io/stylegan2-ada-pytorch/). The e4e encoder is under MIT license. The HDA dataset is used strictly for non-commercial academic research in accordance with the original authors' terms.

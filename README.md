# ImgSecure 🔍

**AI-Powered Image Forensics Tool — Phase 0**

Phase 0 of ImgSecure analyzes images for signs of tampering, manipulation, or AI generation using three forensic techniques: Error Level Analysis, Metadata Inspection, and Noise Pattern Analysis.

---

## Why This Exists

Non-consensual image manipulation is a growing threat — deepfakes, AI-generated faces, and edited photos cause real harm to real people. ImgSecure is a step toward building tools that can detect when an image has been tampered with, before that harm spreads.

This is Phase 0 of a larger digital safety system

---

## What It Does

### 1. Error Level Analysis (ELA)

Re-compresses the image at a known JPEG quality and computes the pixel-wise difference from the original. Tampered regions — which were compressed at a different error level — show up significantly brighter in the output.

### 2. Metadata Inspection

Extracts raw EXIF data and maps tag IDs to human-readable names. Red flags include editing software signatures on a photo claimed to be original, mismatched timestamps, or missing camera data entirely.

### 3. Noise Pattern Analysis

Applies Gaussian blur to isolate the high-frequency noise layer, then amplifies the difference. Authentic images have spatially consistent noise. Spliced, edited, or AI-generated regions break that consistency and appear as anomalies.

---

## Project Structure

```
IMGSECURE/
├── modules/
│   ├── ela.py          # Error Level Analysis
│   ├── metadata.py     # EXIF metadata extraction
│   └── noise.py        # Noise pattern analysis
├── output/
│   ├── ela_edited_result.jpeg
│   └── noise_result.jpeg
├── test_images/        # Sample images for testing
├── analyze.py          # CLI entry point
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Kavyaa156/imgsecure.git
cd imgsecure
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install pillow opencv-python numpy matplotlib
```

---

## Usage

```bash
python analyze.py test_images/your_image.jpeg
```

Outputs are saved to the `output/` folder. Metadata is printed to the terminal.

---

## Stack

- Python 3.11
- Pillow
- OpenCV
- NumPy
- Matplotlib

---

## Author

**Kavyaa** — final year B.E AI & Data Science  
Building at the intersection of computer vision and digital safety.  
[GitHub](https://github.com/Kavyaa156)

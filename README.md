Here is the fully updated, polished `README.md` file containing both Phase 0 (Forensics) and your newly completed Phase 1 (Adversarial Immunization & Benchmarking). You can copy and paste this directly into your repository's `README.md` file:

```markdown
# ImgSecure 🔍

**AI-Powered Image Forensics & Adversarial Immunization Toolkit**

ImgSecure is a dual-phase digital safety architecture. **Phase 0** provides forensic tools to detect image tampering, metadata manipulation, and generative AI artifacts. **Phase 1** implements active adversarial immunization, utilizing gradient-based attacks (FGSM and PGD) on multimodal vision encoders to protect digital media from unauthorized AI scrapers and deepfake manipulation.

---

## 🛡️ Why This Exists

Non-consensual image manipulation, unauthorized web scraping, and deepfakes pose critical threats to digital privacy and creative safety. ImgSecure bridges the gap between passive forensic detection and active defensive immunization—providing developers and creators the tools to audit and secure digital media.

---

## 🛠️ Phase 0: Image Forensics Suite

Phase 0 analyzes images for tampering and AI generation using three core computer vision techniques:

1. **Error Level Analysis (ELA):** Re-compresses images at known JPEG qualities to expose pixel-wise compression discrepancies. Tampered regions show up as high-error anomalies.
2. **Metadata Inspection:** Extracts raw EXIF data, surfacing hidden editing signatures, mismatched timestamps, or missing sensor tags.
3. **Noise Pattern Analysis:** Isolates high-frequency noise layers via Gaussian smoothing to reveal spatial inconsistencies caused by splicing or generative AI insertion.

---

## ⚡ Phase 1: Adversarial Immunization & Encoder Attacks

Phase 1 explores active defense by formulating **Latent Space Encoder Attacks** inspired by PhotoGuard. Instead of attacking a standard image classifier, ImgSecure targets the multimodal **CLIP (ViT-B/32)** vision-language encoder to induce semantic drift, breaking automated feature extraction while keeping perturbations visually imperceptible.

### Comparative Benchmark: FGSM vs. PGD on CLIP (ViT-B/32)
Both attacks were executed under an identical perturbation budget ($\epsilon = 0.03$) on the same test image:

| Attack Method | Final Cosine Similarity | PSNR (dB) | SSIM | Status / Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **FGSM** *(Single-Step)* | `0.8093` | `37.20` | `0.9925` | Moderate drift; single-step noise absorbed by transformer patch attention. |
| **PGD** *(Iterative, 30 Steps)* | `0.4980` | `38.14` | `0.9938` | **Success!** Robust semantic drift achieved below the $0.5$ threshold. |

* **Key Engineering Insight:** Iterative multi-step optimization (**PGD**) successfully warps Vision Transformer self-attention maps and collapses embedding similarity, whereas single-step gradient jumps (**FGSM**) lack the precision required for complex multimodal spaces. Both methods maintain an **SSIM > 0.99**, ensuring absolute visual stealth.

---

## 📂 Project Structure


```

IMGSECURE/
├── modules/
│   ├── ela.py              # Error Level Analysis (Phase 0)
│   ├── metadata.py         # EXIF metadata extraction (Phase 0)
│   └── noise.py            # Noise pattern analysis (Phase 0)
├── output/                 # Generated forensic and adversarial artifacts
├── test_images/            # Sample images for testing
├── analyze.py              # CLI entry point for forensics
├── encoder_attack.py       # PGD & FGSM CLIP encoder immunization pipeline
└── README.md

```

---

## 🚀 Setup & Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/Kavyaa156/imgsecure.git](https://github.com/Kavyaa156/imgsecure.git)
cd imgsecure
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install torch torchvision ftfy regex tqdm scikit-image pillow opencv-python numpy matplotlib

```

*(Note: Ensure you install OpenAI's CLIP package directly from GitHub: `pip install git+https://github.com/openai/CLIP.git`)*

---

## 💻 Usage

### Run Forensic Analysis (Phase 0)

```bash
python analyze.py test_images/your_image.jpeg

```

### Run Adversarial Encoder Immunization Pipeline (Phase 1)

```bash
python encoder_attack.py

```

*(Outputs protected adversarial images and amplified noise maps to your workspace).*

---

## 🧰 Tech Stack

* **Core & Deep Learning:** Python, PyTorch, torchvision, OpenAI CLIP (ViT-B/32)
* **Image Processing & Quality Metrics:** OpenCV, Pillow, NumPy, Scikit-Image (`skimage`)
* **Visualization:** Matplotlib

---

## Author

**Kavyaa** — Final-year B.E. Artificial Intelligence & Data Science

Building at the intersection of computer vision, deep learning security, and digital privacy.

[GitHub](https://github.com/Kavyaa156)

```

```
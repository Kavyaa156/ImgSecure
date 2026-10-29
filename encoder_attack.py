"""
ImgSecure - Phase 1: Adversarial Latent Space Immunization
Executes a Projected Gradient Descent (PGD) attack against the OpenAI CLIP (ViT-B/32)
vision encoder to induce semantic drift while preserving visual fidelity.
"""

import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import requests
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

def tensor_to_numpy(t):
    """Safely detach a PyTorch tensor, move to CPU, and convert to uint8 numpy array."""
    arr = t.detach().squeeze(0).cpu().permute(1, 2, 0).numpy()
    return np.clip(arr * 255, 0, 255).astype(np.uint8)

def main():
    # 1. Environment & Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing CLIP (ViT-B/32) on device: {device.upper()}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 2. Load Sample Image
    url = "https://images.cocodataset.org/test-stuff2017/000000028352.jpg"
    print(f"[*] Downloading test image from COCO dataset...")
    raw_image = Image.open(requests.get(url, stream=True, verify=False).raw)
    
    image = preprocess(raw_image).unsqueeze(0).to(device)

    # 3. Extract Baseline Latent Embeddings
    with torch.no_grad():
        base_features = model.encode_image(image)
        base_l2norm = torch.nn.functional.normalize(base_features, p=2, dim=1)

    # 4. Hyperparameters
    epsilon = 0.03
    num_steps = 30
    alpha = 2 * (epsilon / num_steps)

    print(f"[*] Running PGD Attack (epsilon={epsilon}, steps={num_steps}, alpha={alpha:.4f})...")
    pgd_copy = image.clone().to(device).requires_grad_(True)

    # 5. PGD Iterative Optimization Loop
    for step in range(num_steps):
        if pgd_copy.grad is not None:
            pgd_copy.grad.zero_()

        # Forward pass & L2 Normalization
        features = model.encode_image(pgd_copy)
        l2norm = torch.nn.functional.normalize(features, p=2, dim=1)

        # Minimize cosine similarity (maximize similarity loss)
        cos_sim = torch.nn.functional.cosine_similarity(l2norm, base_l2norm, dim=1)
        loss = cos_sim.mean()

        loss.backward()

        # Gradient descent update step with epsilon ball projection
        with torch.no_grad():
            pgd_copy.data.sub_(alpha * pgd_copy.grad.sign())
            
            # Project back into epsilon neighborhood
            displacement = pgd_copy - image
            displacement.clamp_(-epsilon, epsilon)
            
            pgd_copy.copy_(image + displacement)
            pgd_copy.clamp_(0, 1)

    # 6. Final Evaluation Metrics
    with torch.no_grad():
        final_features = model.encode_image(pgd_copy)
        final_l2norm = torch.nn.functional.normalize(final_features, p=2, dim=1)
        final_similarity = torch.nn.functional.cosine_similarity(base_l2norm, final_l2norm, dim=1).item()

    orig_np = tensor_to_numpy(image)
    adv_np = tensor_to_numpy(pgd_copy)
    
    psnr_val = compute_psnr(orig_np, adv_np)
    ssim_val = compute_ssim(orig_np, adv_np, channel_axis=2)

    # 7. Print Summary Report
    print("\n" + "="*45)
    print("      IMGSECURE PGD ENCODER BENCHMARK REPORT      ")
    print("="*45)
    print(f" Initial Cosine Similarity : 1.0000")
    print(f" Final Cosine Similarity   : {final_similarity:.4f}")
    print(f" PSNR (Distortion Power)   : {psnr_val:.2f} dB")
    print(f" SSIM (Visual Fidelity)    : {ssim_val:.4f}")
    print("-"*45)
    
    if final_similarity < 0.5:
        print(" [✓] Status: SUCCESS! Robust semantic drift achieved.")
    else:
        print(" [!] Status: Similarity remains high. Adjust epsilon/steps.")
    print("="*45 + "\n")

    # 8. Save Artifacts
    Image.fromarray(adv_np).save("adversarial_imgsecure.png")
    
    noise_np = (displacement.squeeze(0).cpu().permute(1, 2, 0).numpy() + epsilon) / (2 * epsilon)
    noise_np = np.clip(noise_np * 255, 0, 255).astype(np.uint8)
    Image.fromarray(noise_np).save("adversarial_noise_map.png")
    
    print("[*] Artifacts saved: 'adversarial_imgsecure.png' & 'adversarial_noise_map.png'")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import requests
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

url = "https://images.cocodataset.org/test-stuff2017/000000028352.jpg"
raw_image = Image.open(requests.get(url, stream=True, verify=False).raw)

# Keep clean original tensor for metrics
image = preprocess(raw_image).unsqueeze(0).to(device)

# Extract original latent embeddings
image_features = model.encode_image(image).detach()
image_l2norm = torch.nn.functional.normalize(image_features, p=2, dim=1)

epsilon = 0.03

print("\n--- Running FGSM Attack ---")
fgsm_copy = image.clone().to(device).requires_grad_(True)

if fgsm_copy.grad is not None:
    fgsm_copy.grad.zero_()

fgsm_features = model.encode_image(fgsm_copy)
fgsm_l2norm = torch.nn.functional.normalize(fgsm_features, p=2, dim=1)

fgsm_loss = torch.nn.functional.cosine_similarity(fgsm_l2norm, image_l2norm, dim=1).mean()
fgsm_loss.backward()

with torch.no_grad():
    fgsm_adv = fgsm_copy - epsilon * fgsm_copy.grad.sign()
    fgsm_adv = torch.clamp(fgsm_adv, 0, 1)
    
    fgsm_final_features = model.encode_image(fgsm_adv)
    fgsm_final_l2norm = torch.nn.functional.normalize(fgsm_final_features, p=2, dim=1)
    fgsm_sim = torch.nn.functional.cosine_similarity(image_l2norm, fgsm_final_l2norm, dim=1).item()

print(f"FGSM Final Cosine Similarity: {fgsm_sim:.4f}")

print("\n--- Running PGD Attack ---")
pgd_copy = image.clone().to(device).requires_grad_(True)
num_steps = 30
alpha = 2 * (epsilon / num_steps)

for step in range(num_steps):
    if pgd_copy.grad is not None:
        pgd_copy.grad.zero_()

    pgd_features = model.encode_image(pgd_copy)
    pgd_l2norm = torch.nn.functional.normalize(pgd_features, p=2, dim=1)

    pgd_loss = torch.nn.functional.cosine_similarity(pgd_l2norm, image_l2norm, dim=1).mean()
    pgd_loss.backward()

    with torch.no_grad():
        pgd_copy.data.sub_(alpha * pgd_copy.grad.sign())
        displacement = pgd_copy - image
        displacement.clamp_(-epsilon, epsilon)
        pgd_copy.copy_(image + displacement)
        pgd_copy.clamp_(0, 1)

with torch.no_grad():
    pgd_final_features = model.encode_image(pgd_copy)
    pgd_final_l2norm = torch.nn.functional.normalize(pgd_final_features, p=2, dim=1)
    pgd_sim = torch.nn.functional.cosine_similarity(image_l2norm, pgd_final_l2norm, dim=1).item()

print(f"PGD Final Cosine Similarity:  {pgd_sim:.4f}")

# Helper to convert tensor to numpy uint8 image
def tensor_to_numpy(t):
    # Detach the tensor from the computational graph first
    arr = t.detach().squeeze(0).cpu().permute(1, 2, 0).numpy()
    return np.clip(arr * 255, 0, 255).astype(np.uint8)

orig_np = tensor_to_numpy(image)
fgsm_np = tensor_to_numpy(fgsm_adv)
pgd_np = tensor_to_numpy(pgd_copy)

# Compute PSNR
fgsm_psnr = compute_psnr(orig_np, fgsm_np)
pgd_psnr = compute_psnr(orig_np, pgd_np)

# Compute SSIM (multichannel for RGB)
fgsm_ssim = compute_ssim(orig_np, fgsm_np, channel_axis=2)
pgd_ssim = compute_ssim(orig_np, pgd_np, channel_axis=2)

print("\n==============================================")
print("         COMPARATIVE PERFORMANCE BENCHMARK     ")
print("==============================================")
print(f"{'Attack':<10} | {'Cosine Sim':<12} | {'PSNR (dB)':<10} | {'SSIM':<8}")
print("-" * 50)
print(f"{'FGSM':<10} | {fgsm_sim:<12.4f} | {fgsm_psnr:<10.2f} | {fgsm_ssim:<8.4f}")
print(f"{'PGD':<10} | {pgd_sim:<12.4f} | {pgd_psnr:<10.2f} | {pgd_ssim:<8.4f}")
print("==============================================\n")

# Save output images
Image.fromarray(fgsm_np).save("fgsm_adversarial.png")
Image.fromarray(pgd_np).save("pgd_adversarial.png")
print("Saved comparison images successfully!")
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

url = "https://images.cocodataset.org/test-stuff2017/000000028352.jpg"
raw_image = Image.open(requests.get(url, stream=True, verify=False).raw)

image = preprocess(raw_image).unsqueeze(0).to(device)

# Forward pass - extract latent embeddings
image_features = model.encode_image(image).detach()

# L2 Normalization
image_l2norm = torch.nn.functional.normalize(image_features, p=2, dim=1)

copy = image.clone().to(device).requires_grad_(True)

#Define your attack hyperparameters: maximum perturbation budget ($\epsilon$), total iteration steps, and step size ($\alpha$).
epsilon = 0.03
num_steps = 30
alpha = 2 * (epsilon / num_steps)

#Iterative pgd loop
for step in range(num_steps):
    if copy.grad is not None:
        copy.grad.zero_()

#forward pass
    image_features = model.encode_image(copy)

#L2 normalization
    copy_l2norm = torch.nn.functional.normalize(image_features, p=2, dim=1)

# Cosine similarity loss
    cos = torch.nn.functional.cosine_similarity(copy_l2norm, image_l2norm, dim=1)
    loss = cos.mean()

    loss.backward()

    with torch.no_grad():
        copy.data.sub_(alpha * copy.grad.sign())

        displacement = copy - image
        displacement.clamp_(-epsilon, epsilon)

        copy.copy_(image + displacement)
        copy.clamp_(0, 1)

#Final evaluation report
with torch.no_grad():
    final_features = model.encode_image(copy)
    final_l2norm = torch.nn.functional.normalize(final_features, p=2, dim=1)
    final_similarity = torch.nn.functional.cosine_similarity(image_l2norm, final_l2norm, dim=1)

    print(f"\n--- PGD Optimization Complete ---")
    print(f"Initial Cosine Similarity: 1.0000")
    print(f"Final Cosine Similarity:   {final_similarity.item():.4f}")

    if final_similarity.item() < 0.5:
        print("Success! Semantic drift achieved")
    else:
        print("Similarity remains high. Consider increasing epsilon or num_steps")

    #save perturbed adversarial image
    adv_np = copy.squeeze(0).cpu().permute(1,2,0).numpy()
    adv_np = np.clip(adv_np * 255, 0, 255).astype(np.uint8)
    Image.fromarray(adv_np).save("adversarial_imgsecure.png")

    #save amplified noise map for visualization
    noise_np = (displacement.squeeze(0).cpu().permute(1, 2, 0).numpy() + epsilon) / (2 * epsilon)
    noise_np = np.clip(noise_np * 255, 0, 255).astype(np.uint8)
    Image.fromarray(noise_np).save("adversarial_noise_map.png")
    
    print("Saved 'adversarial_imgsecure.png' and 'adversarial_noise_map.png' to workspace!")

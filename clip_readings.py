import torch
import torch.nn as nn
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

image2 = image.clone().to(device).requires_grad_(True)

image2_features = model.encode_image(image2)

image2_l2norm = torch.nn.functional.normalize(image2_features, p=2, dim=1)

# Cosine similarity loss
cos = torch.nn.functional.cosine_similarity(image_l2norm, image2_l2norm, dim=1)
loss = cos.mean()

#backward pass - difference between the image & image2
loss.backward()


if image2.grad is not None:
    val = torch.abs(image2.grad).sum().item()
    if image2.grad.shape == image2.shape and val > 0:
        print("Successfull!")

# UNDERSTAND THE CODE FIRST AND THEN MOVE ON

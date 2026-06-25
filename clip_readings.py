import torch
import clip
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

url = "https://images.cocodataset.org/test-stuff2017/000000028352.jpg"
raw_image = Image.open(requests.get(url, stream=True).raw)

image = preprocess(raw_image).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)

print(image_features.shape)
print(image_features[0, :10])
print("Vector Magnitude (L2 Norm):", torch.norm(image_features, p=2).item())

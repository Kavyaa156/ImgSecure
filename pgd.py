import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import requests
from numpy import asarray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} for inference..')

resnet50 = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                       'nvidia_convnets_processing_utils')

resnet50.eval().to(device)

uri = 'http://images.cocodataset.org/test-stuff2017/000000028117.jpg'
ex_image = Image.open(requests.get(uri, stream=True).raw)

img_arr = asarray(ex_image)
img_tensor = torch.tensor(img_arr, dtype=torch.float).permute(2, 0, 1)
image = (torch.unsqueeze(img_tensor, 0)) / 255.0
image = image.to(device)

copy = image.clone().detach().to(device).requires_grad_(True)

# ImageNet normalization values
m = [0.485, 0.456, 0.406]
sd = [0.229, 0.224, 0.225]

mean = torch.tensor(m).reshape(1, 3, 1, 1).to(device)
std = torch.tensor(sd).reshape(1, 3, 1, 1).to(device)


epsilon = 0.05
num_steps = 10
alpha = 2 * (epsilon / num_steps)


for step in range(num_steps):
    if copy.grad is not None:
        copy.grad.zero_()

    norm_img = (copy - mean) / std

    img = resnet50(norm_img)
    true_label = torch.tensor([294]).long().to(device)

    loss = nn.CrossEntropyLoss()
    loss = loss(img, true_label)

    loss.backward()

    with torch.no_grad():
        copy.data.add_(alpha * copy.grad.sign())

        displacement = copy - image
        displacement.clamp_(-epsilon, epsilon)

        copy.copy_(image + displacement)
        copy.clamp_(0, 1)

with torch.no_grad():
    # 1. Get the model's new prediction on the perturbed image
    final_norm = (copy - mean) / std
    outputs = resnet50(final_norm)
    _, predicted_idx = torch.max(outputs, 1)

    print(f"\nTarget Class Index (True Label): {true_label.item()}")
    print(f"Model's Predicted Class Index After PGD: {predicted_idx.item()}")

    if true_label.item() == predicted_idx.item():
        print("❌ The attack failed to fool the model. Try increasing epsilon or num_steps.")
    else:
        print("🎉 Success! The model was fooled by the adversarial image.")

    # 2. Convert tensors back to PIL Images to save them
    # Remove batch dimension and move to CPU/NumPy
    adv_np = copy.squeeze(0).cpu().permute(1, 2, 0).numpy()
    adv_np = (adv_np * 255).astype(np.uint8)
    adv_img = Image.fromarray(adv_np)
    adv_img.save("adversarial_image.png")

    # 3. Amplify and save the noise pattern itself so we can see it
    noise_np = (displacement.squeeze(0).cpu().permute(
        1, 2, 0).numpy() + epsilon) / (2 * epsilon)
    noise_np = (noise_np * 255).astype(np.uint8)
    noise_img = Image.fromarray(noise_np)
    noise_img.save("adversarial_noise.png")

    print("\nSaved 'adversarial_image.png' and 'adversarial_noise.png' to your workspace!")

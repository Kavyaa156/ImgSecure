# -----SETUP ZONE-----#
import os
import requests
import matplotlib.pyplot as plt
from numpy import asarray
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import skimage as sk

output_dir = "result"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} for inference..')

resnet50 = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                       'nvidia_convnets_processing_utils')

resnet50.eval().to(device)

uri = 'http://images.cocodataset.org/test-stuff2017/000000028117.jpg'
ex_image = Image.open(requests.get(uri, stream=True).raw)

# images to NumPy array
img_arr = asarray(ex_image)

# NumPy array to tensors
# change {height, width, channel} to {channel, height, width}
img_tensor = torch.tensor(img_arr, dtype=torch.float).permute(2, 0, 1)
image = (torch.unsqueeze(img_tensor, 0)) / 255.0

# ImageNet normalization values
m = [0.485, 0.456, 0.406]
sd = [0.229, 0.224, 0.225]
mean, std = (torch.tensor(m)).reshape(
    1, 3, 1, 1), (torch.tensor(sd)).reshape(1, 3, 1, 1)

# Normalization
output_img = (image - mean) / std

# moving tensor from cpu to gpu
output_img = output_img.to(device)
output_img.requires_grad_(True)


epsilons = [0.01, 0.05, 0.1]
true_label = torch.tensor([294]).long().to(device)
loss = nn.CrossEntropyLoss()

# -----EXPERIMENT ZONE-----#

for epsilon in epsilons:
    print(f"\n--- Running Attack with Epsilon: {epsilon} ---")
    if output_img.grad is not None:
        output_img.grad.zero_()

    output = resnet50(output_img)
    calculated_loss = loss(output, true_label)
    calculated_loss.backward()

    # FGSM implementation
    dirty_img = (output_img + torch.sign(output_img.grad) * epsilon)

    # un-normalizing the dirty_img
    std, mean = std.to(device), mean.to(device)
    dirty_img = ((dirty_img * std) + mean).clamp(0, 1)

    # for visualizing
    new_output_img = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    new_dirty_img = dirty_img.squeeze(0).permute(
        1, 2, 0).detach().cpu().numpy()
    pert = new_dirty_img - new_output_img

    # normalizing the perturbation for visualization (scaling)
    scaled_pert = (pert - np.min(pert)) / (np.max(pert) - np.min(pert))

    # PSNR
    psnr = sk.metrics.peak_signal_noise_ratio(
        new_output_img, new_dirty_img, data_range=1.0)

    # SSIM
    ssim = sk.metrics.structural_similarity(
        new_output_img, new_dirty_img, data_range=1.0, channel_axis=-1)

    print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

    # visual comparison using subplot
    plt.subplot(1, 3, 1)
    plt.imshow(new_output_img)
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(scaled_pert)
    plt.axis('off')
    plt.title("Perturbation")

    plt.subplot(1, 3, 3)
    plt.imshow(new_dirty_img)
    plt.axis('off')
    plt.title("Perturbed Image")

    plt.suptitle(f"FGSM Attack (Epsilon = {epsilon})")

    # save results to folder
    file_path = os.path.join(output_dir, f"fgsm_eps_{epsilon}.png")
    plt.savefig(file_path, bbox_inches='tight', dpi=300)

    plt.close()

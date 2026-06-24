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

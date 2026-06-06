import requests  # to send http requests
import json
# image manipulation: convert to tensors, normailze, etc.
import torchvision.transforms as transforms
import warnings
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

warnings.filterwarnings('ignore')

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

resnet50 = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                       'nvidia_convnets_processing_utils')

resnet50.eval().to(device)

uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
]

batch = torch.cat(
    [utils.prepare_input_from_uri(uri) for uri in uris]
).to(device)

with torch.no_grad():
    output = torch.nn.functional.softmax(resnet50(batch), dim=1)

results = utils.pick_n_best(predictions=output, n=5)

for uri, result in zip(uris, results):
    img = Image.open(requests.get(uri, stream=True).raw)
    img.thumbnail((256, 256), Image.LANCZOS)
    plt.imshow(img)
    plt.savefig(f"prediction_{uri.split('/')[-1]}")
    plt.close()
    print(result)

import numpy as np
import gradio as gr

import torch
from torchvision import datasets,transforms
from torch.nn.functional import normalize

import torch.nn.functional as F

from PIL import Image

import torchvision.models as models

import requests
imagenet_classes = requests.get('https://files.fast.ai/models/imagenet_class_index.json').json()

device = "cuda" if torch.cuda.is_available() else \
    "mps" if torch.backends.mps.is_built() else "cpu"

IMAGE_SIZE = (224,224)

data_transforms = transforms.Compose([
    transforms.Resize(size=IMAGE_SIZE), # делаем все картинки квадратными
    transforms.ToTensor(), # преобразуем в тензор
])

model_full = models.resnet50(weights='DEFAULT').to(device)

def classify(input_img):
    

    img = data_transforms(Image.fromarray(input_img)).unsqueeze(0).to(device)
    
    model_full.eval()
    results = model_full(img)

    top = torch.sort(F.softmax(results, dim=1)[0] * 100, descending=True)
    predictions = [f"{imagenet_classes[str(ix.cpu().item())][1]} - {pct:.2f}%" \
               for pct, ix in zip(*top) ][:5]
    classes = ', '.join(predictions)
    
    return classes

# file_name = '1696609397_gas-kvas-com-p-kartinki-kota-malenkaya-7.jpg'
# img = Image.open(file_name)

# classify(img)

demo = gr.Interface(classify, gr.Image(), outputs="text")
demo.launch()
"""_summary_
    Pipeline to execute an inference pipeline with a ResNet-18

    Author: Ricard Santiago Raigada García
    Original code: Udacity Machine Learing DevOps Engineer
    Date: January, 2024
"""

import torch
from torchvision import transforms
from torch.nn import Sequential, Softmax
from PIL import Image
import numpy as np

# Setup an inference pipeline with a pre-trained model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
model.eval()

pipe = Sequential(
    transforms.Resize([256, 256]),
    transforms.CenterCrop([224, 224]),
    transforms.Normalize(
        mean=[
            0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]),
    model,
    Softmax(1)
)

# Save inference artifact
scripted = torch.jit.script(pipe)
scripted.save("inference_artifact.pt")

# Load inference artifact
pipe_reload = torch.jit.load("inference_artifact.pt")

# Load one example
img = Image.open("dog.jpg")
img.load()

# Make into a batch of 1 element
data = transforms.ToTensor()(np.asarray(img, dtype="uint8").copy()).unsqueeze(0)

# Perform inference
with torch.no_grad():
    logits = pipe_reload(data).detach()

proba = logits[0]

# Transform to class and print answer
with open("imagenet_classes.txt", "r") as f:
    classes = [s.strip() for s in f.readlines()]
print(f"Classification: {classes[proba.argmax()]}")

import subprocess

# Install the 'smdebug' library using 'pip' if not already installed
subprocess.call(['pip', 'install', 'smdebug'])

import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Import the 'smdebug' library after installing it
import smdebug

# Set up logging
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def model_fn(model_dir):
    """
    Function to load the pre-trained ResNet50 model and modify its last fully connected layer
    for a specific classification task.

    Parameters:
        model_dir (str): Directory path where the model file 'model.pth' is located.

    Returns:
        model (torch.nn.Module): The modified ResNet50 model ready for inference.
    """
    # Load the pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Freeze all parameters of the pre-trained model to avoid gradient updates during training
    for param in model.parameters():
        param.requires_grad = False

    # Modify the last fully connected layer for the new classification task
    num_features = model.fc.in_features
    num_classes = 133
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1)
    )

    # Load the fine-tuned model weights from the 'model.pth' file in the specified directory
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    # Set the model in evaluation mode (no gradient computation during inference)
    model.eval()
    return model

def input_fn(request_body, content_type):
    """
    Function to process the incoming request body and convert it to a format suitable for the model.

    Parameters:
        request_body (bytes): The incoming request body (image data in JPEG format).
        content_type (str): The content type of the incoming request.

    Returns:
        input_object (PIL.Image.Image): The processed input image in PIL Image format.
    """
    if content_type == 'image/jpeg':
        # Convert the request body bytes to a PIL Image
        return Image.open(io.BytesIO(request_body))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    """
    Function to perform inference on the input image using the provided model.

    Parameters:
        input_object (PIL.Image.Image): The input image in PIL Image format.
        model (torch.nn.Module): The model used for inference.

    Returns:
        prediction (torch.Tensor): The model's prediction for the input image.
    """
    # Apply image transformations to make it suitable for the model
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    input_object = transform(input_object)
    
    # Perform inference (no gradient computation needed for inference)
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction

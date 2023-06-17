import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import os
import sys

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import smdebug.pytorch as smd


def test(model, test_loader, criterion, hook):
    """
    Test the model on a given test dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to be tested.
        test_loader (torch.utils.data.DataLoader): The PyTorch data loader for the test dataset.
        criterion (torch.nn.Module): The loss function to be used for the test.

    Returns:
        None

    Logs:
        Starting test...
        Testing Accuracy: {average_accuracy:.4f}
        Testing Loss: {average_loss:.4f}
        Great job! if average_accuracy > 0.9 else Good job! Improve the accuracy. if average_accuracy > 0.8 else Keep working.
        Tested {num_images} images in {num_batches} batches of size {batch_size}.
    
    Writes:
        A new row in a CSV file named "results.csv" with the following format: {average_accuracy:.4f},{average_loss:.4f}
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting test...')
    model.eval()
    test_loss = 0
    running_corrects = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
    
    average_accuracy = running_corrects / len(test_loader.dataset)
    average_loss = test_loss / len(test_loader.dataset)
    logger.info(f'Test set: Average loss: {average_loss}, Accuracy: {average_accuracy * 100}%')
    
    # Logger info
    if average_accuracy > 0.9:
        logger.info('Great job!')
    elif average_accuracy > 0.8:
        logger.info('Good job! Improve the accuracy.')
    else:
        logger.warning('Keep working.')
    
    # Size of the dataset
    num_images = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    num_batches = len(test_loader)
    logger.info(f'Tested {num_images} images in {num_batches} batches of size {batch_size}.')
    
    # Record the results in a CSV file
    with open('results.csv', 'a') as f:
        f.write(f'{average_accuracy:.4f},{average_loss:.4f}\n')


def train(model, train_loader, validation_loader, epochs, criterion, optimizer, hook):
    """
    Trains a PyTorch model on a given dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (torch.utils.data.DataLoader): The DataLoader containing the training dataset.
        epochs (int): The number of epochs to train the model for.
        criterion (torch.nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to use.

    Returns:
        The trained PyTorch model.
    Logs:
        Starting training...: At the start of training.
        Training epoch...: For each epoch and batch, displaying epoch number, epoch count, batch number and batch count, and loss.
        Epoch...: At the end of each epoch, displaying epoch number, epoch count, train accuracy, and validation accuracy.
        Training complete.: At the end of training.
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting training...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(epochs):
        hook.set_mode(smd.modes.TRAIN)
        model.train()
        train_loss = 0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
        
        epoch_loss = train_loss / len(train_loader.dataset)
        epoch_accuracy = running_corrects / len(train_loader.dataset)
        
        logger.info(f'Epoch {epoch + 1}/{epochs}:')
        logger.info(f'Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%')
        
        hook.set_mode(smd.modes.EVAL)
        model.eval()
        validation_loss = 0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
        
        validation_loss = validation_loss / len(validation_loader.dataset)
        validation_accuracy = running_corrects / len(validation_loader.dataset)
        
        logger.info(f'Validation Loss: {validation_loss:.4f}, Accuracy: {validation_accuracy * 100:.2f}%')
    
    logger.info('Training complete.')
    
    return model
  
    
def net(train_data_loader):
    """
    Creates a ResNet50 model with additional layers for fine-tuning, and returns the model.
    
    Args:
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
    
    Returns:
        (torch.nn.Module): The created ResNet50 model with additional layers for fine-tuning.
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating the model...')

    model = models.resnet50(pretrained=True)
    
    # Freezing pretrained parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Adding new layers for fine-tuning
    num_features = model.fc.in_features
    num_classes = len(train_data_loader.dataset.classes)
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1)
    )
    # Counting number of layers and trainable parameters
    num_layers = len(list(model.parameters()))
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f'Model created: {num_layers} layers, {num_trainable_params} trainable parameters.')
    return model

def create_data_loaders(data, batch_size):
    """
    Creates the data loaders for training, testing, and validation datasets.
    
    Args:
        data: A string with the path to the directory containing the train, test and validation folders.
        batch_size: An integer representing the batch size for the data loaders.
    
    Returns:
        A tuple with three DataLoader objects for training, testing, and validation datasets respectively.
    
    Raises:
        ValueError: If one or more data directories not found.
    
    Logs:
        'Creating data loaders...': When the function is called.
        'Train dataset loaded with [train_dataset_length] images': After loading the train dataset.
        'Test dataset loaded with [test_dataset_length] images': After loading the test dataset.
        'Validation dataset loaded with [validation_dataset_length] images': After loading the validation dataset.
        'Train data loader created with batch size [batch_size] and [train_batches] batches': After creating the train data loader.
        'Test data loader created with batch size [batch_size] and [test_batches] batches': After creating the test data loader.
        'Validation data loader created with batch size [batch_size] and [validation_batches] batches': After creating the validation data loader.
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating data loaders...')
    
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    validation_path = os.path.join(data, 'valid')
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)
    validation_dataset = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f'Train dataset loaded with {len(train_dataset)} images')
    logger.info(f'Test dataset loaded with {len(test_dataset)} images')
    logger.info(f'Validation dataset loaded with {len(validation_dataset)} images')
    logger.info(f'Train data loader created with batch size {batch_size} and {len(train_data_loader)} batches')
    logger.info(f'Test data loader created with batch size {batch_size} and {len(test_data_loader)} batches')
    logger.info(f'Validation data loader created with batch size {batch_size} and {len(validation_data_loader)} batches')
    
    return train_data_loader, test_data_loader, validation_data_loader


def main(args):
    """
    The main function for training and saving a PyTorch model.

    Args:
        args (argparse.Namespace): command-line arguments.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")
    
    # Get the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create datasets
    train_data_loader, test_data_loader, validation_data_loader = create_data_loaders(data=args.data_dir, batch_size=args.batch_size)
    
    # Initialize a model by calling the net function
    model = net(train_data_loader)
    model = model.to(device)
    
    # Create the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    # Train the model
    logger.info("Training a new model")
    model = train(model, train_data_loader, validation_data_loader, args.epochs, criterion, optimizer, hook)
    
    # Test the model
    test(model, test_data_loader, criterion, hook)
    
    # Save the trained model
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    logger.info("Model saved")


if __name__ == '__main__':
    # Training args 
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=16, metavar="N", help="input batch size for training")
    parser.add_argument("--epochs", type=int, default=6, metavar="N", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate")
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="training data path in S3")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"], help="location to save the model to")
    
    args = parser.parse_args()
    
    main(args)

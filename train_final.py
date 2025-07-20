from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# You'll need to implement or import these classes
from dataset import AnimalDataset
from models import AdvancedCNN

def train():
    # Hyperparameters
    batch_size = 8
    lr = 1e-3
    momentum = 0.9
    num_epochs = 100
    
    # Data preprocessing
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset and DataLoader setup
    train_dataset = AnimalDataset(root="./data", train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    val_dataset = AnimalDataset(root="./data", train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    # Model setup
    model = AdvancedCNN(num_classes=len(train_dataset.categories))
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
    
    num_iters = len(train_dataloader)
    
    # TensorBoard setup
    tensorboard_path = "./tensorboard"
    if not os.path.isdir(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)
    
    # Training loop
    for epoch in range(num_epochs):
        # TRAINING PHASE
        model.train()
        train_losses = []
        progress_bar = tqdm(train_dataloader, colour="cyan")
        
        for iter, (images, targets) in enumerate(progress_bar):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, targets)
            train_losses.append(loss.item())
            
            # Update progress bar with more detailed info
            progress_bar.set_description(
                "Epoch {}/{}, Loss {:.6f}".format(epoch+1, num_epochs, loss.item())
            )
            
            # Log training loss to TensorBoard
            writer.add_scalar(
                tag="Train/Loss", 
                scalar_value=loss.item(), 
                global_step=epoch*num_iters+iter
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # VALIDATION PHASE
        model.eval()
        losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_dataloader:
                # Forward
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                losses.append(loss.item())
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.tolist())
                all_targets.extend(targets.tolist())
        
        loss = np.mean(losses)
        acc = accuracy_score(all_targets, all_predictions)
        print("Loss: {}. Accuracy: {}".format(loss, acc))
        writer.add_scalar(tag="Val/Loss", scalar_value=loss, global_step=epoch)
        writer.add_scalar(tag="Val/Accuracy", scalar_value=acc, global_step=epoch)
        
if __name__ == "__main__":
    train()
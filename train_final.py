from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
import gc
import torch

# Thu hồi các object không còn dùng nữa
gc.collect()

# Nếu bạn có dùng GPU thì nên thêm:
torch.cuda.empty_cache()
# You'll need to implement or import these classes
from dataset import AnimalDataset
from models import AdvancedCNN
import matplotlib.pyplot as plt

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

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
        num_workers=0,
    )
    
    val_dataset = AnimalDataset(root="./data", train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
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
                global_step=epoch*num_iters+iter # vi du epoch 1, iter 2 -> global_step = 1*100+2 = 102
            )
            
            # Backward pass and optimization
            optimizer.zero_grad() # set gradient ve 0 
            loss.backward() # tinh gradient
            optimizer.step() # cap nhat weights
        
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
                predictions = torch.argmax(logits, dim=1) # dim = 0 là chiều batch size, 1 là chó mèo gà..,
                all_predictions.extend(predictions.tolist())
                all_targets.extend(targets.tolist())
        
        loss = np.mean(losses)
        acc = accuracy_score(all_targets, all_predictions)
        print("Loss: {}. Accuracy: {}".format(loss, acc))
        writer.add_scalar(tag="Val/Loss", scalar_value=loss, global_step=epoch)
        writer.add_scalar(tag="Val/Accuracy", scalar_value=acc, global_step=epoch)
        plot_confusion_matrix(writer=writer, cm=confusion_matrix(all_targets, all_predictions),
                              class_names=train_dataset.categories, epoch=epoch)
        
        

        torch.save(model.state_dict(), "model/model_{}.pt".format(epoch+1))
        global best_acc  # nếu viết trong hàm train()

        # Lưu mô hình tốt nhất
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "model/best_model.pt")
            print(f"✅ Saved new best model with acc: {best_acc:.4f}")

                
if __name__ == "__main__":
    train()
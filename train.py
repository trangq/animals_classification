import torch
import torch.nn as nn
from models import AdvancedCNN
from dataset import AnimalDataset
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader

def train():
    batch_size = 8
    lr = 1e-3
    momentum = 0.9
    num_epochs = 100
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    train_dataset = AnimalDataset(root="./data", train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_dataset = AnimalDataset(root="./data", train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    model = AdvancedCNN(num_classes=len(train_dataset.categories))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(num_epochs):
        for images, targets in train_dataloader:
            # Forward
            logits = model(images)
            loss = criterion(logits, targets)
            print(loss.item())

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    train()
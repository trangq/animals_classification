import torch.nn as nn
import torch


class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.create_block(in_channels=3, out_channels=8)
        self.conv2 = self.create_block(in_channels=8, out_channels=16)
        self.conv3 = self.create_block(in_channels=16, out_channels=32)
        self.conv4 = self.create_block(in_channels=32, out_channels=64)
        self.conv5 = self.create_block(in_channels=64, out_channels=64)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=3136, out_features=512),
            nn.LeakyReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=128),
            nn.LeakyReLU()
        )

        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def create_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    sample_input = torch.rand(8, 3, 224, 224)
    model = AdvancedCNN()
    prediction = model(sample_input)  # [8, 8, 222, 222]
    print(prediction.shape)

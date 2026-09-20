"""CNN models built from scratch for animal classification."""

import torch
import torch.nn as nn


class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.create_conv_block(in_channels=3, out_channels=8, kernel_size=3)
        self.conv2 = self.create_conv_block(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = self.create_conv_block(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = self.create_conv_block(in_channels=32, out_channels=64, kernel_size=3)
        self.conv5 = self.create_conv_block(in_channels=64, out_channels=64, kernel_size=3)

        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=64 * 3 * 3, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    @staticmethod
    def create_conv_block(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)
        return self.classifier(x)
if __name__ == "__main__":
    random_data = torch.rand(8, 3, 96, 96)
    model = AdvancedCNN()
    output = model(random_data)
    print(output.shape)  # [8, 10]

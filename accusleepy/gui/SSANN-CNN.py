"""
Simple small artificial neural network from AccuSleePy
    
"""
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device, flatten
from torchviz import make_dot

IMAGE_HEIGHT = 184

class SSANN(nn.Module):
    """Small CNN for classifying images"""

    def __init__(self, n_classes: int):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding="same"
        )
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(int(32 * IMAGE_HEIGHT / 8), n_classes)

    def forward(self, x):
        x = x.float()
        x = self.pool(nn.functional.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.conv3_bn(self.conv3(x))))
        x = flatten(x, 1)  # flatten all dimensions except batch
        return self.fc1(x)
        
        
model = SSANN(n_classes=3)
input_data = torch.randn(1, 1, 13, 185) # Create a dummy input (Batch, Channels, Height, Width)
output = model(input_data)      # Perform forward pass

# Generate the visual for Netron

torch.onnx.export(model, input_data, 'SSANN.onnx')

# Generate and save the visualization
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("simple_nn_graph", format="png") # Saves as simple_nn_graph.png


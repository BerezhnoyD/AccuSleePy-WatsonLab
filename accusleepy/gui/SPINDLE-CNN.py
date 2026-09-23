    """
    Exact CNN architecture from Miladinović et al. (2019).
    Classifies sleep stages (WAKE, NREM, REM) from time-frequency EEG/EMG data.
    
    Expects the input image of the size 3 x 160 x 24
    3 = EEG1, EEG2 and EMG channel spectrograms
    160 = 5 epochs (central and two neighbours on both sides)
    24 = frequency bins of the spectrogram
    
    
    """
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class SPINDLE(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()

        # 1. First Max-Pooling Layer (Occurs BEFORE convolution!)
        # PyTorch expects (Height, Width) -> (Freq Bins, Time Steps)
        # Shrinks Freq: 24 -> 8. Shrinks Time: 160 -> 80.
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 2), stride=(3, 2))

        # 2. Convolutional Layer
        # 3 in_channels -> 50 out_channels. Kernel 3x3, no padding.
        # Shrinks Freq: 8 -> 6. Shrinks Time: 80 -> 78.
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=50, 
            kernel_size=(3, 3), 
            stride=(1, 1),
            padding=0 
        )

        # 3. Second Max-Pooling Layer
        # Shrinks Freq: 6 -> 3. Shrinks Time: 78 -> 39.
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # 4. Fully-Connected Layers with 50% Dropout
        # Flattened size is 50 channels * 3 height * 39 width = 5850
        self.fc1 = nn.Linear(5850, 1000)
        self.dropout = nn.Dropout(p=0.5)
        
        # Second FC layer maps the 1000 hidden units to the 3 classes (WAKE, NREM, REM)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        # Ensure input is a float tensor
        x = x.float()

        # Step 1: Initial Pooling (Output: Batch x 3 x 8 x 80)
        x = self.pool1(x)

        # Step 2: Convolution + ReLU Activation (Output: Batch x 50 x 6 x 78)
        x = F.relu(self.conv1(x))

        # Step 3: Second Pooling (Output: Batch x 50 x 3 x 39)
        x = self.pool2(x)

        # Step 4: Flatten all dimensions except batch (Output: Batch x 5850)
        x = torch.flatten(x, 1)

        # Step 5: Dense 1 + ReLU Activation + Dropout (Output: Batch x 1000)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Step 6: Final Output classification (Output: Batch x n_classes)
        x = self.fc2(x)

        return x
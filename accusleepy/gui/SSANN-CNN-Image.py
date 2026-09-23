
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
import matplotlib.pyplot as plt

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
        
        
def generate_class_image(model, target_class: int, image_shape=(1, 1, 13, 185), steps=1000):
    """
    Optimizes a random noise tensor to maximize the activation of a specific class.
    """
    # 1. Freeze the model's weights (we don't want to change the network)
    model.eval() 
    for param in model.parameters():
        param.requires_grad = False

    # 2. Create a random noise image AND tell PyTorch to track its gradients!
    # This is the magic step. `requires_grad=True` means this tensor is now a trainable parameter.
    generated_image = torch.randn(image_shape, requires_grad=True)

    # 3. Create an optimizer, but pass it the IMAGE, not the model parameters
    optimizer = optim.Adam([generated_image], lr=0.1)

    print(f"Optimizing image for Class {target_class}...")

    for i in range(steps):
        optimizer.zero_grad()

        # Forward pass: what does the network think this noise is right now?
        output_logits = model(generated_image)

        # We want to MAXIMIZE the logit of our target class. 
        # PyTorch optimizers only MINIMIZE. So, we make the loss the negative of the target logit.
        loss = -output_logits[0, target_class]

        # Backward pass: calculate the gradients of the image pixels with respect to the loss
        loss.backward()

        # Update the image pixels
        optimizer.step()

        if i % 100 == 0:
            print(f"Step {i}, Target Class Activation (Logit): {-loss.item():.4f}")

    return generated_image.detach()
    
    
def total_variation_loss(img_tensor):
    """
    Calculates the Total Variation (TV) loss.
    This penalizes sharp, jagged changes between adjacent pixels,
    forcing the generated image to be smooth and continuous.
    """
    # Calculate the absolute difference between adjacent pixels horizontally and vertically
    tv_h = torch.sum(torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]))
    return tv_h + tv_w

def generate_regularized_class_image(model, target_class: int, image_shape=(1, 1, 13, 185), steps=20000):
    # 1. Freeze the model's weights
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 2. Start with a gray canvas mixed with a tiny bit of noise
    # Starting near zero helps prevent the optimizer from getting stuck in extreme pixel values early on
    generated_image = torch.zeros(image_shape) + torch.randn(image_shape) * 0.05
    generated_image.requires_grad_(True)

    # 3. Setup the Optimizer
    optimizer = optim.Adam([generated_image], lr=0.05)

    # Hyperparameters for the regularizers (you may need to tweak these!)
    tv_weight = 0.0001  # How strongly to enforce smoothness
    l2_weight = 0.0001   # How strongly to penalize overly bright/dark pixels

    print(f"Optimizing regularized image for Class {target_class}...")

    for i in range(steps):
        optimizer.zero_grad()

        # Forward pass: what does the network see?
        output_logits = model(generated_image)
        class_score = output_logits[0, target_class]

        # Calculate Regularization Penalties
        tv_penalty = total_variation_loss(generated_image)
        l2_penalty = torch.norm(generated_image) # Keeps pixel values from exploding to infinity

        # The New Loss Function:
        # We want to MAXIMIZE the class score (so we make it negative)
        # We want to MINIMIZE the TV and L2 penalties (so we keep them positive)
        loss = -class_score + (tv_weight * tv_penalty) + (l2_weight * l2_penalty)

        # Backward pass & update
        loss.backward()
        optimizer.step()

        # Optional: Apply a slight Gaussian blur every few steps to kill stubborn high-frequency noise
        if i > 0 and i % 20 == 0:
            with torch.no_grad(): # Don't track gradients during the blur
                # Kernel size 3, sigma 0.5
                generated_image.copy_(TF.gaussian_blur(generated_image, kernel_size=[3, 3], sigma=[0.5, 0.5]))

        if i % 100 == 0:
            print(f"Step {i} | Total Loss: {loss.item():.4f} | Class Score: {class_score.item():.4f} ")

    return generated_image.detach()
    
    
    
        
# 1. Instantiate the EXACT same architecture and load the weights
model = SSANN(n_classes=3)
weights_path = 'F:/NFSS_cohort0/12_19_25-LD4/LD_12_21/ssann_1s.pth'
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

# 2. Clean state dictionary
clean_state_dict = {}

for key, value in state_dict.items():
    # We only care about keys that start with "model." (the actual weights)
    if key.startswith('model.'):
        # Strip away the "model." prefix (e.g., "model.conv1.weight" -> "conv1.weight")
        new_key = key.replace('model.', '', 1)
        
        # Add the fixed key and its weights to our clean dictionary
        clean_state_dict[new_key] = value


model.load_state_dict(clean_state_dict)
model.eval()

# 2. Run the reverse-optimization loop (using the function we wrote earlier)
# Let's say Class 1 is NREM Sleep
target_class_idx = 0
optimized_tensor = generate_regularized_class_image(model, target_class=target_class_idx)
# The tensor is shape (1, 3, 224, 224). Matplotlib strictly expects (Height, Width, Channels)
# So we drop the batch dimension (squeeze) and rearrange the axes (permute), normalize the pixel values
img_array = optimized_tensor.squeeze(0).permute(1, 2, 0).numpy()
img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
plt.imshow(img_array)
plt.axis('off')
plt.title(f"Network's internal representation of Class {target_class_idx}")
plt.show()


import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


# --- 1. Data Loading and Preprocessing ---

class FingerprintDataset(Dataset):
    """Custom PyTorch Dataset for SOCOFing fingerprints."""

    def __init__(self, real_paths, altered_paths, img_size=(96, 96)):
        self.real_paths = real_paths
        self.altered_paths = altered_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.real_paths)

    def __getitem__(self, idx):
        # Load altered image (input)
        img_altered = cv2.imread(self.altered_paths[idx], cv2.IMREAD_GRAYSCALE)
        img_altered = cv2.resize(img_altered, self.img_size) / 255.0

        # Load real image (target)
        img_real = cv2.imread(self.real_paths[idx], cv2.IMREAD_GRAYSCALE)
        img_real = cv2.resize(img_real, self.img_size) / 255.0

        # Convert to PyTorch tensors and add a channel dimension
        img_altered = torch.from_numpy(img_altered).float().unsqueeze(0)
        img_real = torch.from_numpy(img_real).float().unsqueeze(0)

        return img_altered, img_real


def prepare_filepaths(real_dir, altered_dirs):
    """
    Pairs real images with altered images from a LIST of altered directories.
    """
    real_paths, altered_paths = [], []

    print(f"Aggregating data from: {altered_dirs}")
    for altered_dir in altered_dirs:
        for altered_file in os.listdir(altered_dir):
            base_name = '_'.join(altered_file.split('_')[:-1]) + '.BMP'
            real_file_path = os.path.join(real_dir, base_name)

            if os.path.exists(real_file_path):
                altered_paths.append(os.path.join(altered_dir, altered_file))
                real_paths.append(real_file_path)

    print(f"Found {len(real_paths)} total image pairs.")
    return real_paths, altered_paths


# --- 2. The ResNet Autoencoder Model Definition ---

class ResNetAutoencoder(nn.Module):
    def __init__(self):
        super(ResNetAutoencoder, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Modify the first layer for 1-channel (grayscale) input
        original_weights = resnet.conv1.weight.clone()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data = original_weights.mean(dim=1, keepdim=True)

        # Encoder is the ResNet without the final layers
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# --- 3. Visualization Function ---
def visualize_results(model, data_loader, device, num_images=5):
    model.eval()
    altered_imgs, real_imgs = next(iter(data_loader))
    altered_imgs, real_imgs = altered_imgs.to(device), real_imgs.to(device)

    with torch.no_grad():
        reconstructed_imgs = model(altered_imgs).cpu()

    plt.figure(figsize=(15, 7))
    for i in range(num_images):
        # Altered Input
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(altered_imgs[i].cpu().squeeze(), cmap='gray')
        ax.set_title("Altered")
        ax.axis('off')

        # Reconstructed Output
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_imgs[i].squeeze(), cmap='gray')
        ax.set_title("Reconstructed")
        ax.axis('off')

        # Ground Truth
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(real_imgs[i].cpu().squeeze(), cmap='gray')
        ax.set_title("Ground Truth")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# --- 4. Main Execution Block ---
if __name__ == '__main__':

    BASE_DIR = '../SOCOFing'
    REAL_PATH = os.path.join(BASE_DIR, 'Real')
    ALTERED_PATHS = [
        os.path.join(BASE_DIR, 'Altered/Altered-Easy'),
        os.path.join(BASE_DIR, 'Altered/Altered-Medium'),
        os.path.join(BASE_DIR, 'Altered/Altered-Hard')
    ]

    # --- Configuration ---
    IMG_SIZE = (96, 96)
    BATCH_SIZE = 128
    LR = 1e-4
    EPOCHS = 100

    # --- Load and Split Data ---
    real_filepaths, altered_filepaths = prepare_filepaths(REAL_PATH, ALTERED_PATHS)

    # Split all aggregated data into training and testing sets
    alt_train, alt_test, real_train, real_test = train_test_split(
        altered_filepaths, real_filepaths, test_size=0.2, random_state=42
    )

    train_dataset = FingerprintDataset(real_train, alt_train, IMG_SIZE)
    test_dataset = FingerprintDataset(real_test, alt_test, IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Initialize Model, Loss, and Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Starting training on {device} with {len(train_dataset)} training images.")

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for altered_imgs, real_imgs in train_loader:
            altered_imgs, real_imgs = altered_imgs.to(device), real_imgs.to(device)

            outputs = model(altered_imgs)
            loss = criterion(outputs, real_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * altered_imgs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.6f}")

    print("\nTraining finished.")

    # --- Save the Final, Robust Model ---
    torch.save(model.state_dict(), 'fingerprint_autoencoder_robust.pth')
    print("Robust model saved to 'fingerprint_autoencoder_robust.pth'")

    # --- Visualize Results on the Test Set ---
    print("\nVisualizing test results...")
    visualize_results(model, test_loader, device)
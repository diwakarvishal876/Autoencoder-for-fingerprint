import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import matplotlib.pyplot as plt
import numpy as np


# --- 1. Re-define the Model Architecture ---

class ResNetAutoencoder(nn.Module):
    def __init__(self):
        super(ResNetAutoencoder, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Modify the first layer for 1-channel (grayscale) input
        original_weights = resnet.conv1.weight.clone()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data = original_weights.mean(dim=1, keepdim=True)

        # Encoder is the ResNet model without the final layers
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Decoder reconstructs the image
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


# --- 2. Function to Denoise a Single Image ---
def denoise_single_image(model, image_path, img_size=(96, 96)):
    """Loads an altered image, runs it through the model, and returns the denoised version."""
    device = next(model.parameters()).device

    # Load the altered image in grayscale
    img_altered = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_altered is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Preprocess the image (resize, normalize)
    img_resized = cv2.resize(img_altered, img_size)
    img_normalized = img_resized / 255.0

    # Convert to a PyTorch tensor and add batch & channel dimensions
    img_tensor = torch.from_numpy(img_normalized).float().unsqueeze(0).unsqueeze(0).to(device)

    # Set the model to evaluation mode and get the prediction
    model.eval()
    with torch.no_grad():
        denoised_tensor = model(img_tensor)

    # Convert the output tensor back to a displayable image
    denoised_img = denoised_tensor.cpu().squeeze().numpy()
    denoised_img = (denoised_img * 255).astype(np.uint8)

    return img_altered, denoised_img


# --- 3. Main Execution Block ---
if __name__ == '__main__':
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Create an instance of the model architecture
    model = ResNetAutoencoder().to(device)

    # 2. Load the saved weights into the model
    model_path = 'fingerprint_autoencoder_robust.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model weights loaded successfully from '{model_path}'")

    # --- Use the Model ---
    TEST_IMAGE_PATH = '../SOCOFing/Altered/Altered-Hard/10__M_Left_index_finger_CR.BMP'

    try:
        altered_image, denoised_image = denoise_single_image(model, TEST_IMAGE_PATH)

        # --- Visualize the Result ---
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Altered Input")
        plt.imshow(altered_image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Denoised Output")
        plt.imshow(denoised_image, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(e)
        print("Please update the TEST_IMAGE_PATH to a valid image file.")
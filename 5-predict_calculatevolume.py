import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r3d_18
from torchvision.transforms import Normalize

# 🔹 Define the ResNet3D Model for Classification
class ResNet3DForClassification(nn.Module):
    def __init__(self, num_classes=3):  # 3 Classes: 0 = Other, 1 = Myocardium, 2 = Trabeculae
        super(ResNet3DForClassification, self).__init__()
        self.input_adapter = nn.Conv3d(1, 3, kernel_size=1)  # Convert 1-channel to 3-channel
        self.model = r3d_18(weights=None)  # Load ResNet3D without pretrained weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust output layer

    def forward(self, x):
        x = self.input_adapter(x)
        return self.model(x)

# 🔹 Dataset for Loading `.pt` Files
class PTDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = torch.load(file_path)  # Load .pt file
        if self.transform:
            image = self.transform(image)
        return image, file_path  # Return file name for reference

# 🔹 Function to Calculate Myocardium & Trabeculae Volumes
def calculate_volumes(segmentation_tensor):
    """
    Calculates the volume of Myocardium (label=1) and Trabeculae (label=2)
    based on the number of voxels in the segmentation mask.
    """
    voxel_count = segmentation_tensor.numel()  # Total number of voxels
    myocardium_voxels = torch.sum(segmentation_tensor == 1).item()
    trabeculae_voxels = torch.sum(segmentation_tensor == 2).item()

    myocardium_volume = (myocardium_voxels / voxel_count) * 100  # Percentage volume
    trabeculae_volume = (trabeculae_voxels / voxel_count) * 100  # Percentage volume

    return myocardium_voxels, trabeculae_voxels, myocardium_volume, trabeculae_volume

# 🔹 Function to Get Predictions
def predict_tissue(patient_data, model, device):
    normalize = Normalize(mean=[0.5], std=[0.5])  # Normalize data
    patient_data = normalize(patient_data)
    patient_data = patient_data.unsqueeze(0).to(device, dtype=torch.float)

    model.eval()
    with torch.no_grad():
        output = model(patient_data)
        probs = torch.softmax(output, dim=1)  # Convert to probabilities
        predicted_class = torch.argmax(probs).item()

    return probs.cpu().numpy(), predicted_class

# 🔹 Main Function for Loading `.pt` Files & Running Predictions
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load `.pt` files
    pt_files = [os.path.join("pt_files", f) for f in os.listdir("pt_files") if f.endswith(".pt")]

    # Initialize dataset and dataloader
    dataset = PTDataset(pt_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load trained model
    model = ResNet3DForClassification(num_classes=3).to(device)
    model.load_state_dict(torch.load("resnet3d_myocardium_trabeculae.pth"))  # Load trained model

    # Class names
    class_names = ["Other Tissue", "Myocardium", "Trabeculae"]

    # Process each `.pt` file
    for patient_data, file_path in dataloader:
        patient_data = patient_data.to(device)

        # Get predictions
        mean_probs, predicted_class = predict_tissue(patient_data, model, device)

        # Load segmentation mask to calculate volume
        segmentation_tensor = torch.load(file_path[0])  # Reload for volume calculation
        myocardium_voxels, trabeculae_voxels, myocardium_volume, trabeculae_volume = calculate_volumes(segmentation_tensor)

        # Print results
        print(f"\nFile: {os.path.basename(file_path[0])}")
        print(f"Predicted Tissue: {class_names[predicted_class]}")
        print(f"Confidence Scores: Other: {mean_probs[0][0]:.4f}, Myocardium: {mean_probs[0][1]:.4f}, Trabeculae: {mean_probs[0][2]:.4f}")
        print(f"Myocardium Volume: {myocardium_voxels} voxels ({myocardium_volume:.2f}%)")
        print(f"Trabeculae Volume: {trabeculae_voxels} voxels ({trabeculae_volume:.2f}%)")

if __name__ == "__main__":
    main()

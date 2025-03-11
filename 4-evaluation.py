import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from torchvision.models.video import r3d_18
from torchvision.transforms import Normalize
import numpy as np

# 🔹 Dataset Class for Loading `.pt` Files
class PTDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        image = torch.load(file_path)  # Load .pt file

        if self.transform:
            image = self.transform(image)

        return image, label

# 🔹 ResNet3D Model for Classification
class ResNet3DForClassification(nn.Module):
    def __init__(self, num_classes=3):  # 3 classes: 0 = other, 1 = myocardium, 2 = trabeculae
        super(ResNet3DForClassification, self).__init__()
        self.input_adapter = nn.Conv3d(1, 3, kernel_size=1)  # Convert 1-channel to 3-channel
        self.model = r3d_18(weights=None)  # Load model without pretrained weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust output layer

    def forward(self, x):
        x = self.input_adapter(x)
        return self.model(x)

# 🔹 Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)  # Convert to probabilities
            preds = torch.argmax(outputs, dim=1)  # Get predicted class

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute evaluation metrics
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")  # ROC-AUC for multi-class
    cm = confusion_matrix(all_labels, all_preds)
    
    # Compute sensitivity & specificity
    sensitivity = {}
    specificity = {}
    for i in range(3):  # For each class: myocardium, trabeculae, other
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = cm.sum() - (tp + fn + fp)
        
        sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return roc_auc, sensitivity, specificity

# 🔹 Main Function to Load Model & Run Evaluation
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Normalize(mean=[0.5], std=[0.5])  # Normalize intensities

    # Load test set
    test_files = [os.path.join("pt_test", f) for f in os.listdir("pt_test") if f.endswith(".pt")]
    test_labels = [0] * len(test_files)  # Replace with actual labels if available

    # Create dataloader
    test_dataset = PTDataset(test_files, test_labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Load trained model
    model = ResNet3DForClassification(num_classes=3).to(device)
    model.load_state_dict(torch.load("resnet3d_myocardium_trabeculae.pth"))  # Load trained model

    # Evaluate model
    roc_auc, sensitivity, specificity = evaluate(model, test_loader, device)

    print(f"ROC-AUC: {roc_auc:.4f}")
    for cls in range(3):
        print(f"Class {cls} → Sensitivity: {sensitivity[cls]:.4f}, Specificity: {specificity[cls]:.4f}")

if __name__ == "__main__":
    main()

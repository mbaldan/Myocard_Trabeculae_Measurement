import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18  # 3D ResNet
from torchvision.transforms import Normalize

# 🔹 Dataset Class to Load .pt Files
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
    def __init__(self, num_classes=3):  # 3 classes (0 = other, 1 = myocardium, 2 = trabeculae)
        super(ResNet3DForClassification, self).__init__()
        self.input_adapter = nn.Conv3d(1, 3, kernel_size=1)  # Convert 1-channel to 3-channel
        self.model = r3d_18(weights="KINETICS400_V1")  # Pretrained ResNet3D
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust output layer

    def forward(self, x):
        x = self.input_adapter(x)
        return self.model(x)

# 🔹 Train the Model for One Epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# 🔹 Evaluate Model Performance
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# 🔹 Main Function to Load Data and Train
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load file paths
    myocardium_files = [os.path.join("pt_files", f) for f in os.listdir("pt_files") if f.endswith(".pt")]
    trabeculae_files = [os.path.join("pt_files", f) for f in os.listdir("pt_files") if f.endswith(".pt")]

    # Assign labels (0 = other, 1 = myocardium, 2 = trabeculae)
    myocardium_labels = [1] * len(myocardium_files)
    trabeculae_labels = [2] * len(trabeculae_files)

    all_files = myocardium_files + trabeculae_files
    all_labels = myocardium_labels + trabeculae_labels

    # Split data into train & validation (80% train, 20% val)
    split_idx = int(0.8 * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]

    # Create datasets and dataloaders
    transform = Normalize(mean=[0.5], std=[0.5])  # Normalize intensities
    train_dataset = PTDataset(train_files, train_labels, transform)
    val_dataset = PTDataset(val_files, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Initialize model, loss, and optimizer
    model = ResNet3DForClassification(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), "resnet3d_myocardium_trabeculae.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()

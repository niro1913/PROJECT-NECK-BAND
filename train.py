import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Dataset paths
train_path = r"C:\Users\niros\Desktop\currency\Indian currency dataset v1\training"
val_path = r"C:\Users\niros\Desktop\currency\Indian currency dataset v1\validation"

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# MobileNetV3 Small
model = models.mobilenet_v3_small(weights="DEFAULT")

# Change final layer for our 8 classes
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(train_dataset.classes)
)

# CPU
device = torch.device("cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )

# Save model
torch.save(model.state_dict(), "currency_model.pth")

# Save class names
with open("class_names.txt", "w") as f:
    for name in train_dataset.classes:
        f.write(name + "\n")

print("\nTraining completed!")
print("Model saved as currency_model.pth")
print("Classes saved as class_names.txt")
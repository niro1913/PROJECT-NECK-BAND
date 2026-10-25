import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Class names
class_names = [
    "10",
    "100",
    "20",
    "200",
    "2000",
    "50",
    "500",
    "Background"
]

# Load model
model = models.mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(class_names)
)

model.load_state_dict(torch.load("currency_model.pth", map_location="cpu"))

model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# CHANGE THIS IMAGE PATH
image_path = r"C:\Users\niros\Desktop\currency\Indian currency dataset v1\validation\500\500_val_10.jpg"

# Open image
image = Image.open(image_path).convert("RGB")

# Prepare image
image = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    confidence, prediction = torch.max(probabilities, 1)

# Result
predicted_class = class_names[prediction.item()]
confidence_value = confidence.item() * 100

print("Predicted:", predicted_class)
print(f"Confidence: {confidence_value:.2f}%")
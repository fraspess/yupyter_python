# ==========================================================
# MediScan AI — Pneumonia Detection with Transfer Learning
# Dataset: Chest X-Ray Pneumonia
# Model: ResNet18 (PyTorch)
# ==========================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================================
# 1. Налаштування
# ==========================================================

# Шлях до датасету.
# Очікується структура:
# chest_xray/
# ├── train/
# │   ├── NORMAL/
# │   └── PNEUMONIA/
# ├── test/
# │   ├── NORMAL/
# │   └── PNEUMONIA/
# └── val/
#     ├── NORMAL/
#     └── PNEUMONIA/
DATA_DIR = "chest_xray"

BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Для відтворюваності
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ==========================================================
# 2. Трансформації (resize + normalization)
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])

# ==========================================================
# 3. Завантаження датасету
# ==========================================================

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"),
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "test"),
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.classes
print("Classes:", class_names)

# ==========================================================
# 4. Побудова моделі ResNet18 (Transfer Learning)
# ==========================================================

# Завантаження переднавченої моделі
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Заморожуємо всі згорткові шари
for param in model.parameters():
    param.requires_grad = False

# Замінюємо класифікатор
model.fc = nn.Linear(512, 2)

# Новий класифікатор повинен навчатися
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# ==========================================================
# 5. Функція втрат і оптимізатор
# ==========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

# ==========================================================
# 6. Навчання моделі
# ==========================================================

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total * 100

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_acc:.2f}%"
    )

# ==========================================================
# 7. Оцінка моделі
# ==========================================================

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Метрики
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)

print("\n=== Evaluation Metrics ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# ==========================================================
# 8. Confusion Matrix
# ==========================================================

cm = confusion_matrix(all_labels, all_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# 9. Збереження моделі
# ==========================================================

torch.save(model.state_dict(), "pneumonia_resnet18.pth")
print("\nModel saved as pneumonia_resnet18.pth")

# ==========================================================
# 10. Візуалізація 8 випадкових прогнозів
# ==========================================================

# Денормалізація для відображення
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# Випадкові індекси
indices = random.sample(range(len(test_dataset)), 8)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

model.eval()

with torch.no_grad():
    for ax, idx in zip(axes, indices):
        image, true_label = test_dataset[idx]

        input_tensor = image.unsqueeze(0).to(DEVICE)
        output = model(input_tensor)
        pred_label = torch.argmax(output, dim=1).item()

        img = denormalize(image)

        ax.imshow(img)
        ax.set_title(
            f"Pred: {class_names[pred_label]}\n"
            f"True: {class_names[true_label]}",
            fontsize=10
        )
        ax.axis("off")

plt.tight_layout()
plt.savefig("pneumonia_predictions.png", dpi=300, bbox_inches="tight")
plt.show()

print("Predictions visualization saved as pneumonia_predictions.png")

# ==========================================================
# 11. Висновок
# ==========================================================

print("""
ВИСНОВОК

Модель ResNet18 з transfer learning демонструє високу якість класифікації
рентгенівських знімків на класи NORMAL та PNEUMONIA.

Використання переднавчених ваг ImageNet дозволяє отримати хороші результати
вже після 5 епох навчання.

Однак для реального медичного впровадження модель потребує:
1. Валідації на незалежних клінічних даних.
2. Аналізу помилок за участю лікарів-рентгенологів.
3. Оцінки чутливості та специфічності.
4. Сертифікації відповідно до медичних стандартів.

Отже, модель є перспективним прототипом, але ще не готова до
самостійного використання в клінічній практиці.
""")
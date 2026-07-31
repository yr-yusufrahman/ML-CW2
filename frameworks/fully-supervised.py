"""
Fully-supervised baseline: ResNet-18 trained from scratch on the raw images of
the actively-acquired labelled set.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"
LABELED_POOL_PATH = FEATURES_DIR / "labeled_pool.npy"
LABELED_LABELS_PATH = FEATURES_DIR / "labeled_labels.npy"
DATA_DIR = ROOT / "data"

NUM_CLASSES = 10
EPOCHS = 100
BATCH_SIZE = 64


class RawImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Augmentations: Random crops and horizontal flips[cite: 1]
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    ),
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    ),
])

# 2. Load data.
# labeled_pool.npy holds global CIFAR-10 train indices produced by the active
# learning loop, so the raw pixels have to be gathered from the CIFAR-10 arrays.
cifar_train = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR), train=True, download=True
)
cifar_test = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR), train=False, download=True
)

labeled_indices = np.load(LABELED_POOL_PATH)
labeled_images = cifar_train.data[labeled_indices]
labeled_labels = np.load(LABELED_LABELS_PATH)

print(f"Labelled set: {len(labeled_labels)} samples, image shape {labeled_images.shape[1:]}")

dataset = RawImageDataset(labeled_images, labeled_labels, transform=train_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

eval_train_set = RawImageDataset(labeled_images, labeled_labels, transform=test_transform)
eval_train_loader = DataLoader(eval_train_set, batch_size=256, shuffle=False)

test_set = RawImageDataset(
    cifar_test.data, np.array(cifar_test.targets), transform=test_transform
)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

# 3. Model Configuration
model = models.resnet18(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()

# Optimizer: SGD with 0.9 momentum, Nesterov momentum, initial lr 0.025[cite: 1]
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, nesterov=True)

# Scheduler: Cosine scheduler[cite: 1]
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


@torch.no_grad()
def evaluate(data_loader):
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        preds = model(inputs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


# 4. Training Loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
    scheduler.step()
    epoch_loss /= len(dataset)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}  loss {epoch_loss:.4f}")

print("Fully-supervised training complete.")

train_accuracy = evaluate(eval_train_loader)
test_accuracy = evaluate(test_loader)

print(f"Final loss: {epoch_loss:.4f}")
print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

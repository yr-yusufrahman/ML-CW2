"""
Linear evaluation of the SimCLR representation on a random CIFAR-10 subset
(same size as the TypiClust labelled pool): a single linear layer trained on
frozen 512-D embeddings. Baseline for comparing random sampling vs TypiClust.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"
TRAIN_FEATURES_PATH = FEATURES_DIR / "cifar10_simclr_features.npy"
TEST_FEATURES_PATH = FEATURES_DIR / "cifar10_simclr_features_test.npy"
CHECKPOINT_PATH = ROOT / "pretrained-models" / "simclr_cifar-10.pth.tar"
DATA_DIR = ROOT / "data"

NUM_CLASSES = 10
NUM_SAMPLES = 100
EPOCHS = 10
SEED = 42


def load_representation_learning():
    path = ROOT / "representation-learning.py"
    spec = importlib.util.spec_from_file_location("representation_learning", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_test_features():
    """Test-set embeddings, extracted once and cached alongside the train ones."""
    if TEST_FEATURES_PATH.exists():
        return np.load(TEST_FEATURES_PATH)

    representation_learning = load_representation_learning()
    features = representation_learning.extract_features(
        checkpoint_path=str(CHECKPOINT_PATH),
        data_dir=str(DATA_DIR),
        train=False,
    )
    np.save(TEST_FEATURES_PATH, features)
    return features


# 1. Load data: random CIFAR-10 train subset (no TypiClust labelled pool).
all_train_features = np.load(TRAIN_FEATURES_PATH)
cifar_train = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR), train=True, download=True
)
all_train_labels = np.array(cifar_train.targets)

rng = np.random.default_rng(SEED)
random_indices = rng.choice(len(all_train_labels), size=NUM_SAMPLES, replace=False)

X_train = torch.tensor(all_train_features[random_indices], dtype=torch.float32)
y_train = torch.tensor(all_train_labels[random_indices], dtype=torch.long)

print(
    f"Random CIFAR-10 subset: {X_train.shape[0]} samples, "
    f"{X_train.shape[1]}-D embeddings (seed={SEED})"
)

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# 2. Model configuration (exact paper specs)
# Single linear layer of size d x C (512 x 10 for CIFAR-10)
model = nn.Linear(X_train.shape[1], NUM_CLASSES)
criterion = nn.CrossEntropyLoss()

# Optimizer: SGD with increased learning rate by a factor of 100 to 2.5
optimizer = optim.SGD(model.parameters(), lr=2.5, momentum=0.9)

# 3. Training loop
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
    epoch_loss /= len(dataset)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}  loss {epoch_loss:.4f}")

print("Linear embedding evaluation complete.")

# 4. Evaluate on the held-out CIFAR-10 test set
X_test = torch.tensor(load_test_features(), dtype=torch.float32)
test_targets = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR), train=False, download=True
).targets
y_test = torch.tensor(test_targets, dtype=torch.long)

model.eval()
with torch.no_grad():
    train_accuracy = (model(X_train).argmax(dim=1) == y_train).float().mean().item()
    test_accuracy = (model(X_test).argmax(dim=1) == y_test).float().mean().item()

print(f"Final loss: {epoch_loss:.4f}")
print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

class ResNet18Penultimate(nn.Module):
    """
    Standard ResNet-18 modified to output the 512-dimensional 
    penultimate layer instead of classification logits.
    """
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        # Strip the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1) # Shape: (Batch_Size, 512)
        # TypiClust/SCAN step 1 requires L2 normalized embeddings
        x = F.normalize(x, p=2, dim=1)
        return x

def load_scan_pretrained_weights(model, checkpoint_path):
    """
    Loads weights from the Van Gansbeke et al. SimCLR checkpoint.
    Filters out the projection head and handles state_dict prefixes.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Depending on how it was saved, get the state dict
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # The SCAN repo wraps models in custom classes, creating prefixes
        new_key = key.replace('backbone.', '').replace('resnet.', '').replace('module.', '')
        
        # We only want the ResNet weights, NOT the SimCLR MLP projection head ('head' or 'fc')
        if 'head' not in new_key and 'fc' not in new_key:
            cleaned_state_dict[new_key] = value
            
    # Load with strict=False because our model doesn't have the FC layer
    model.backbone.load_state_dict(cleaned_state_dict, strict=False)
    return model

def extract_features(checkpoint_path, data_dir='./data', batch_size=512, train=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize Model and Load Weights
    model = ResNet18Penultimate()
    model = load_scan_pretrained_weights(model, checkpoint_path)
    model = model.to(device)
    model.eval()

    # 2. Setup CIFAR-10 Dataset
    # Use standard evaluation transforms (no random crops/flips during extraction)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    # TypiClust requires the unlabeled pool U0. We extract for the whole train set.
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )
    # num_workers=0 avoids Windows multiprocessing issues when this
    # module is loaded via importlib from active-learning.py.
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # 3. Extract Representations
    all_features = []
    
    print("Extracting 512-dim L2-normalized features...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            features = model(images)
            all_features.append(features.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
                
    all_features = np.concatenate(all_features, axis=0)
    
    print(f"Extraction complete! Final feature matrix shape: {all_features.shape}")
    
    return all_features

if __name__ == "__main__":
    # Replace with the path to the file you downloaded from the SCAN repo
    PATH_TO_PRETRAINED_WEIGHTS = "pretrained-models/simclr_cifar-10.pth.tar" 
    
    extract_features(checkpoint_path=PATH_TO_PRETRAINED_WEIGHTS)
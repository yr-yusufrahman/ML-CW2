import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class SCANClusteringHead(nn.Module):
    """
    A lightweight network to map the 512-dimensional embeddings 
    to the dynamically required number of clusters.
    """
    def __init__(self, input_dim=512, num_clusters=10):
        super().__init__()
        self.cluster_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_clusters)
        )
        
    def forward(self, x):
        logits = self.cluster_head(x)
        return F.softmax(logits, dim=1)

def get_knn(features, k=5, chunk_size=1000, device='cuda'):
    """
    Computes k-nearest neighbors using chunked matrix multiplication
    to avoid Out-Of-Memory (OOM) errors on large datasets like CIFAR-10.
    """
    N = features.size(0)
    knn_indices = torch.zeros(N, k, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            chunk = features[i:end]
            
            # Since features are L2 normalized, dot product = cosine similarity
            sim = torch.mm(chunk, features.t()) 
            
            # k+1 because the closest neighbor is the image itself (we want to exclude it)
            _, topk_idx = sim.topk(k=k+1, dim=1) 
            knn_indices[i:end] = topk_idx[:, 1:] 
            
    return knn_indices

def scan_loss_fn(p_anchor, p_neighbor, entropy_weight=2.0):
    """
    The official SCAN loss formulation:
    1. Consistency Loss: Forces neighbors to have the same cluster distribution.
    2. Entropy Penalty: Prevents trivial solutions (assigning everything to one cluster).
    """
    # 1. Consistency Loss (Dot product of probabilities)
    consistency_loss = -torch.log((p_anchor * p_neighbor).sum(dim=1) + 1e-8).mean()
    
    # 2. Entropy Regularization
    # Average cluster probabilities across the batch
    p_avg = p_anchor.mean(dim=0) 
    # Maximize entropy by minimizing p * log(p)
    entropy_loss = (p_avg * torch.log(p_avg + 1e-8)).sum() 
    
    return consistency_loss + entropy_weight * entropy_loss

def run_step_2_clustering(embeddings_path, labeled_indices, budget, k_neighbors=5, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Embeddings & Calculate target number of clusters
    embeddings = torch.tensor(np.load(embeddings_path), device=device)
    num_labeled = len(labeled_indices)
    num_clusters = num_labeled + budget
    
    print(f"Labeled pool size: {num_labeled} | Budget: {budget}")
    print(f"Targeting {num_clusters} clusters...")

    # 2. Mine Nearest Neighbors
    print("Mining nearest neighbors...")
    knn_indices = get_knn(embeddings, k=k_neighbors, device=device)

    # 3. Initialize dynamic SCAN head
    model = SCANClusteringHead(input_dim=512, num_clusters=num_clusters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Train the Clustering Head
    print("Training SCAN clustering head...")
    model.train()
    
    batch_size = 1024
    N = embeddings.size(0)
    
    for epoch in range(epochs):
        permutation = torch.randperm(N)
        epoch_loss = 0.0
        
        for i in range(0, N, batch_size):
            indices = permutation[i:i+batch_size]
            
            anchor_features = embeddings[indices]
            
            # Randomly sample one neighbor per anchor for the batch
            neighbor_idx = knn_indices[indices, torch.randint(0, k_neighbors, (len(indices),))]
            neighbor_features = embeddings[neighbor_idx]
            
            optimizer.zero_grad()
            
            p_anchor = model(anchor_features)
            p_neighbor = model(neighbor_features)
            
            loss = scan_loss_fn(p_anchor, p_neighbor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / (N/batch_size):.4f}")

    # 5. Extract Final Cluster Assignments
    model.eval()
    with torch.no_grad():
        all_logits = model(embeddings)
        cluster_assignments = torch.argmax(all_logits, dim=1).cpu().numpy()
        
    print("Clustering complete!")

    # Save the clustering head to a file so it can be loaded later without retraining
    torch.save(model.state_dict(), 'pretrained-models/scan_clustering_head.pth')
    print("Saved clustering head to pretrained-models/scan_clustering_head.pth")

    # Save cluster assignments for Step 3 (querying) to load without re-clustering
    assignments_path = 'features/cluster_assignments.npy'
    np.save(assignments_path, cluster_assignments)
    print(f"Saved cluster assignments to {assignments_path}")
    
    # 6. Identify Uncovered Clusters
    # Uncovered = Clusters that do NOT contain any indices from the labeled pool
    labeled_clusters = set(cluster_assignments[labeled_indices])
    all_clusters = set(range(num_clusters))
    
    uncovered_clusters = list(all_clusters - labeled_clusters)

    uncovered_path = 'features/uncovered_clusters.npy'
    np.save(uncovered_path, np.array(uncovered_clusters, dtype=np.int64))
    print(f"Saved uncovered clusters to {uncovered_path}")
    
    print(f"Total Clusters: {num_clusters}")
    print(f"Uncovered Clusters Found: {len(uncovered_clusters)} (Expected at least {budget})")
    
    return cluster_assignments, uncovered_clusters

if __name__ == "__main__":
    # Example usage for the first Active Learning iteration
    # At iteration 0, L_i-1 is usually empty or contains a tiny seed set.
    
    MOCK_LABELED_INDICES = [] # Empty for the very first step
    BUDGET = 20 # B
    
    clusters, uncovered = run_step_2_clustering(
        embeddings_path='features/cifar10_simclr_features.npy',
        labeled_indices=MOCK_LABELED_INDICES,
        budget=BUDGET
    )
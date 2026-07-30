import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

def get_typical_queries(
    embeddings_path: str, 
    cluster_assignments: np.ndarray, 
    labeled_indices: list, 
    budget: int,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> list:
    """
    Executes Step 3 of TypiClust: Iteratively queries the most typical examples 
    from the most uncovered, largest clusters.
    """
    print(f"Executing Step 3 on {device}...")
    
    # 1. Load Embeddings
    embeddings = torch.tensor(np.load(embeddings_path), device=device)
    
    # 2. Map Cluster Statistics
    num_clusters = len(np.unique(cluster_assignments))
    
    cluster_sizes = {c: np.sum(cluster_assignments == c) for c in range(num_clusters)}
    
    # Track how many labeled points fall into each cluster
    labeled_counts = {c: 0 for c in range(num_clusters)}
    for idx in labeled_indices:
        labeled_counts[cluster_assignments[idx]] += 1
        
    # Drop clusters with less than 5 samples to avoid inaccurate typicality estimation
    valid_clusters = [c for c, size in cluster_sizes.items() if size >= 5]
    
    queries = []
    
    # 3. Iteratively select points until budget is exhausted
    for step in range(budget):
        if not valid_clusters:
            print("Warning: Ran out of valid clusters before budget was exhausted.")
            break
            
        # --- SELECTION CRITERIA 1: Fewest Labeled Points ---
        min_labels = min([labeled_counts[c] for c in valid_clusters])
        candidate_clusters = [c for c in valid_clusters if labeled_counts[c] == min_labels]
        
        # --- SELECTION CRITERIA 2: Largest Cluster Size ---
        selected_cluster = max(candidate_clusters, key=lambda c: cluster_sizes[c])
        
        # --- SELECTION CRITERIA 3: Highest Typicality ---
        # Get all global indices for this cluster
        cluster_idx_global = np.where(cluster_assignments == selected_cluster)[0]
        
        # Filter out points that are already labeled or have been queried in this loop
        unlabeled_idx_global = [
            idx for idx in cluster_idx_global 
            if idx not in labeled_indices and idx not in queries
        ]
        
        if len(unlabeled_idx_global) == 0:
            # Failsafe: if a cluster has no unlabeled points left, remove it and retry
            valid_clusters.remove(selected_cluster)
            continue
            
        # Extract embeddings for distance calculations
        cluster_emb = embeddings[cluster_idx_global]
        unlabeled_emb = embeddings[unlabeled_idx_global]
        
        # k = min{20, cluster_size}. 
        # (Note: cluster_sizes[selected_cluster] is guaranteed >= 5 due to our filter)
        k_neighbors = min(20, cluster_sizes[selected_cluster])
        
        # Calculate pairwise Euclidean distances between the unlabeled points 
        # and ALL points in the cluster to measure true typicality/density.
        dist_matrix = torch.cdist(unlabeled_emb, cluster_emb, p=2.0)
        
        # Sort distances ascending
        sorted_dist, _ = torch.sort(dist_matrix, dim=1)
        
        # The closest point is the image itself (distance 0). 
        # We take the distances to the 1st through k-th neighbors.
        knn_dist = sorted_dist[:, 1:k_neighbors+1]
        
        # Typicality is the inverse of the average distance to the K nearest neighbors
        avg_dist = knn_dist.mean(dim=1)
        typicality = 1.0 / (avg_dist + 1e-8) # Add epsilon to prevent division by zero
        
        # Find the local index with the maximum typicality
        best_local_idx = torch.argmax(typicality).item()
        
        # Map back to the global index
        best_global_idx = unlabeled_idx_global[best_local_idx]
        
        # 4. Update state for the next iteration
        queries.append(best_global_idx)
        labeled_counts[selected_cluster] += 1
        
        print(f"Query {step+1}/{budget}: Selected Image ID {best_global_idx} "
              f"from Cluster {selected_cluster} (Typicality: {typicality[best_local_idx]:.4f})")
              
    return queries

if __name__ == "__main__":
    # Example execution continuing from Step 2
    MOCK_LABELED_INDICES = []
    BUDGET = 20

    assignments = np.load('features/cluster_assignments.npy')

    new_queries = get_typical_queries(
        embeddings_path='features/cifar10_simclr_features.npy',
        cluster_assignments=assignments,
        labeled_indices=MOCK_LABELED_INDICES,
        budget=BUDGET,
    )
    print(f"Selected {len(new_queries)} queries: {new_queries}")

    # Display up to 3 of the selected query images
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    class_names = dataset.classes
    display_indices = new_queries[:3]
    n = len(display_indices)

    if n == 0:
        print("No queries to display.")
    else:
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]
        for ax, idx in zip(axes, display_indices):
            image, label = dataset[idx]
            ax.imshow(image)
            ax.set_title(f"ID {idx}\n{class_names[label]}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()


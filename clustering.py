"""
- SCAN's pseudo-labelling/self-labeling stage is deliberately omitted,
  as required by the TPC-DC description.
- The backbone is NOT trained here. The input embeddings are fixed.
- The clustering head is a linear layer, as in official SCAN.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

EMBEDDINGS_PATH = "features/cifar10_simclr_features.npy"

# Output paths
CLUSTER_ASSIGNMENTS_PATH = "features/cluster_assignments.npy"
UNCOVERED_CLUSTERS_PATH = "features/uncovered_clusters.npy"
SCAN_HEAD_PATH = "pretrained-models/scan_clustering_head.pth"

# Embedding dimensionality
EMBEDDING_DIM = 512

# TPC-DC
BUDGET = 20
MAX_CLUSTERS = 500

# SCAN
K_NEIGHBORS = 20

# Training
EPOCHS = 50
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
# Confirmed against the official CIFAR-10 SCAN config: 5.0 (not 2.0).
ENTROPY_WEIGHT = 5.0
# Number of DataLoader workers. Since the kNN indices are now plain CPU
# data (see fix #1), this can safely be > 0 to parallelize batch
# assembly. Set to 0 if you hit multiprocessing issues on your platform.
NUM_WORKERS = 2
SEED = 42


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# SCAN clustering head & loss
# ============================================================

class SCANClusteringHead(nn.Module):
    """
    Backbone was already trained and its embeddings
    are supplied as input, so this linear layer operates directly
    on the 512-dimensional embeddings.
    """

    def __init__(self, input_dim, num_clusters):
        super().__init__()

        self.cluster_head = nn.Linear(input_dim, num_clusters)

    def forward(self, x):
        return self.cluster_head(x)


class SCANLoss(nn.Module):
    """
    L = consistency_loss - lambda * entropy
    """

    def __init__(self, entropy_weight=5.0):
        super().__init__()

        self.entropy_weight = entropy_weight
        self.bce = nn.BCELoss()

    def forward(self, anchors, neighbours):
        anchors_prob = F.softmax(anchors, dim=1)
        neighbours_prob = F.softmax(neighbours, dim=1)

        batch_size, num_clusters = anchors_prob.shape

        similarity = torch.bmm(
            anchors_prob.view(batch_size, 1, num_clusters),
            neighbours_prob.view(batch_size, num_clusters, 1)
        ).squeeze()

        consistency_loss = self.bce(similarity, torch.ones_like(similarity))

        avg_prob = torch.mean(anchors_prob, dim=0)
        avg_prob = torch.clamp(avg_prob, min=1e-8)

        entropy = -torch.sum(avg_prob * torch.log(avg_prob))
        total_loss = consistency_loss - self.entropy_weight * entropy

        return total_loss, consistency_loss, entropy


# ============================================================
# kNN construction
# ============================================================

def get_knn(features, k=20, chunk_size=1024, device=None):
    """
    Construct a cosine-similarity kNN graph.
    Returns a CPU LongTensor (see fix #1 in the module docstring for why
    this deliberately does NOT stay on GPU).
    """

    if device is None:
        device = features.device

    N = features.shape[0]
    knn_indices = torch.empty((N, k), dtype=torch.long, device=device)
    features = features.to(device)

    print(f"Building kNN graph for {N:,} samples with k={k}...")

    with torch.no_grad():

        # Store transposed features once
        features_t = features.t()
        for start in range(0, N, chunk_size):

            end = min(start + chunk_size, N)

            chunk = features[start:end]
            similarity = torch.mm(chunk, features_t)

            # The sample itself is always the highest similarity, therefore retrieve k+1.
            topk_values, indices = torch.topk(
                similarity, k=k + 1, dim=1, largest=True, sorted=True
            )

            # Remove the sample itself.
            knn_indices[start:end] = indices[:, 1:]

            del similarity

            if (start // chunk_size) % 10 == 0:
                print(f"  Processed {end:,}/{N:,}")

    print("kNN graph complete.")
    return knn_indices.cpu()


# ============================================================
# SCAN neighbour dataset
# ============================================================

class SCANNeighbourDataset(Dataset):
    """
    Dataset for SCAN training.
    A random neighbour is selected from the precomputed kNN graph.
    """

    def __init__(self, knn_indices):
        if isinstance(knn_indices, torch.Tensor):
            knn_indices = knn_indices.cpu().numpy()

        self.knn_indices = knn_indices

    def __len__(self):
        return self.knn_indices.shape[0]

    def __getitem__(self, index):

        neighbours = self.knn_indices[index]

        # Randomly choose a neighbour uniformly from the full k-NN pool.
        neighbour_position = np.random.randint(0, neighbours.shape[0])
        neighbour_index = int(neighbours[neighbour_position])

        return index, neighbour_index


# ============================================================
# Train one SCAN head
# ============================================================

def train_scan_head(
    embeddings, knn_indices, num_clusters, epochs, batch_size,
    learning_rate, entropy_weight, device, num_workers=0
):
    """
    Train one SCAN clustering head.
    """

    print()
    print("=" * 70)
    print("Training SCAN head")
    print("=" * 70)

    model = SCANClusteringHead(
        input_dim=embeddings.shape[1], num_clusters=num_clusters
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = SCANLoss(entropy_weight=entropy_weight)
    dataset = SCANNeighbourDataset(knn_indices)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    model.train()
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):

        epoch_total = 0.0
        epoch_consistency = 0.0
        epoch_entropy = 0.0
        num_batches = 0

        for anchor_indices, neighbour_indices in dataloader:

            anchor_indices = anchor_indices.to(device, non_blocking=True)
            neighbour_indices = neighbour_indices.to(device, non_blocking=True)

            anchor_features = embeddings[anchor_indices]
            neighbour_features = embeddings[neighbour_indices]

            anchor_logits = model(anchor_features)
            neighbour_logits = model(neighbour_features)

            loss, consistency_loss, entropy = criterion(anchor_logits, neighbour_logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_consistency += consistency_loss.item()
            epoch_entropy += entropy.item()
            num_batches += 1

        avg_loss = epoch_total / num_batches
        avg_consistency = epoch_consistency / num_batches
        avg_entropy = epoch_entropy / num_batches

        # Save best model according to SCAN loss
        if avg_loss < best_loss:
            best_loss = avg_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:

            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Consistency: {avg_consistency:.4f} | "
                f"Entropy: {avg_entropy:.4f}"
            )

    # Restore best state
    model.load_state_dict(best_state)
    print(f"Best SCAN loss: {best_loss:.6f}")

    return model, best_loss


# ============================================================
# Predict cluster assignments
# ============================================================

@torch.no_grad()
def get_cluster_assignments(model, embeddings, batch_size, device):
    """
    Obtain the final SCAN cluster assignment for every sample.
    """
    model.eval()
    N = embeddings.shape[0]

    assignments = []
    for start in range(0, N, batch_size):

        end = min(start + batch_size, N)
        batch = embeddings[start:end]
        logits = model(batch)

        predictions = torch.argmax(logits, dim=1)
        assignments.append(predictions.cpu())

    assignments = torch.cat(assignments)

    return assignments.numpy()


# ============================================================
# Cluster statistics
# ============================================================

def print_cluster_statistics(assignments, num_clusters):

    counts = np.bincount(assignments, minlength=num_clusters)

    non_empty = np.sum(counts > 0)
    empty = np.sum(counts == 0)

    print()
    print("Cluster statistics")
    print("-" * 40)
    print(f"Requested clusters : {num_clusters}")
    print(f"Non-empty clusters  : {non_empty}")
    print(f"Empty clusters      : {empty}")
    print(f"Smallest cluster    : {counts[counts > 0].min()}")
    print(f"Largest cluster     : {counts.max()}")
    print(f"Mean cluster size   : {counts.mean():.2f}")


# ============================================================
# Find uncovered clusters
# ============================================================

def find_uncovered_clusters(assignments, labeled_indices, num_clusters):
    """
    An uncovered cluster is a cluster containing no sample
    from the existing labelled set L_{i-1}.
    """

    if len(labeled_indices) == 0:

        # If there are no labelled examples, every cluster
        # is uncovered.
        return np.arange(num_clusters, dtype=np.int64)

    labeled_indices = np.asarray(labeled_indices, dtype=np.int64)

    labeled_clusters = np.unique(assignments[labeled_indices])

    all_clusters = np.arange(num_clusters, dtype=np.int64)

    uncovered_clusters = np.setdiff1d(all_clusters, labeled_clusters)

    return uncovered_clusters


# ============================================================
# Main function
# ============================================================

def run_scan(
    embeddings_path, labeled_indices, budget, max_clusters=500,
    k_neighbors=20, epochs=50, batch_size=1024,
    learning_rate=0.001, entropy_weight=5.0, num_workers=0,
    output_assignments_path=CLUSTER_ASSIGNMENTS_PATH,
    output_uncovered_path=UNCOVERED_CLUSTERS_PATH,
    output_head_path=SCAN_HEAD_PATH
):
    """
    Run SCAN clustering on the embeddings.
    """

    device = get_device()
    print()
    print(f"Loading embeddings from:\n  {embeddings_path}")

    embeddings_np = np.load(embeddings_path)
    print(f"Raw embedding shape: {embeddings_np.shape}")

    if embeddings_np.ndim != 2:
        raise ValueError("Embeddings must have shape [N, D].")

    N, embedding_dim = embeddings_np.shape

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if embedding_dim != EMBEDDING_DIM:
        print(
            f"Warning: expected embedding dimension "
            f"{EMBEDDING_DIM}, but received {embedding_dim}."
        )

    embeddings = torch.from_numpy(embeddings_np).float().to(device)
    del embeddings_np
    print("L2-normalising embeddings...")
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # --------------------------------------------------------
    # Validate labelled indices
    # --------------------------------------------------------

    labeled_indices = np.asarray(labeled_indices, dtype=np.int64)
    if len(labeled_indices) > 0:

        if labeled_indices.min() < 0:
            raise ValueError("labeled_indices contains a negative index.")

        if labeled_indices.max() >= N:
            raise ValueError(
                "labeled_indices contains an index outside the embedding dataset."
            )

    # --------------------------------------------------------
    # Calculate number of clusters
    # --------------------------------------------------------

    num_labeled = len(labeled_indices)
    num_clusters = min(num_labeled + budget, max_clusters)

    print()
    print("=" * 70)
    print("CLUSTERING CONFIGURATION FOR TPC-DC")
    print("=" * 70)

    print(f"Number of samples : {N:,}")
    print(f"Embedding dim     : {embedding_dim}")
    print(f"Existing labelled : {num_labeled:,}")
    print(f"Budget B          : {budget:,}")
    print(f"Max clusters      : {max_clusters:,}")
    print(
        f"Actual K          : min("
        f"{num_labeled} + {budget}, {max_clusters}) = {num_clusters}"
    )

    # --------------------------------------------------------
    # Build kNN graph
    # --------------------------------------------------------
    knn_indices = get_knn(features=embeddings, k=k_neighbors, chunk_size=1024, device=device)

    model, best_loss = train_scan_head(
        embeddings=embeddings, knn_indices=knn_indices,
        num_clusters=num_clusters, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, entropy_weight=entropy_weight,
        device=device, num_workers=num_workers
    )

    os.makedirs(os.path.dirname(output_head_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": embedding_dim,
            "num_clusters": num_clusters,
            "max_clusters": max_clusters,
            "k_neighbors": k_neighbors,
            "entropy_weight": entropy_weight,
            "best_loss": best_loss
        },
        output_head_path
    )

    print(f"Saved SCAN head to:\n  {output_head_path}")
    print()
    print("Computing final cluster assignments...")

    cluster_assignments = get_cluster_assignments(
        model=model, embeddings=embeddings, batch_size=batch_size, device=device
    )

    # --------------------------------------------------------
    # Cluster statistics
    # --------------------------------------------------------

    print_cluster_statistics(assignments=cluster_assignments, num_clusters=num_clusters)
    uncovered_clusters = find_uncovered_clusters(
        assignments=cluster_assignments, labeled_indices=labeled_indices,
        num_clusters=num_clusters
    )

    os.makedirs(os.path.dirname(output_assignments_path), exist_ok=True)
    np.save(output_assignments_path, cluster_assignments)
    print()
    print(f"Saved cluster assignments to:\n  {output_assignments_path}")

    os.makedirs(os.path.dirname(output_uncovered_path), exist_ok=True)
    np.save(output_uncovered_path, uncovered_clusters)
    print(f"Saved uncovered clusters to:\n  {output_uncovered_path}")


    # Final summary
    print()
    print("=" * 70)
    print("TPC-DC STEP 2 COMPLETE")
    print("=" * 70)

    print(f"Total clusters       : {num_clusters}")
    print(f"Uncovered clusters   : {len(uncovered_clusters)}")
    print(f"Required budget B    : {budget}")

    if len(uncovered_clusters) >= budget:
        print(f"Requirement satisfied: {len(uncovered_clusters)} >= {budget}")

    else:
        print(
            f"WARNING: only {len(uncovered_clusters)} uncovered "
            f"clusters were found, fewer than the requested budget of {budget}."
        )

    return (cluster_assignments, uncovered_clusters)


# ============================================================
# Example execution
# ============================================================

if __name__ == "__main__":

    set_seed(SEED)

    # --------------------------------------------------------
    # Existing labelled pool
    # --------------------------------------------------------
    #
    # Replace this with your actual L_{i-1}.
    #
    # At the very first AL iteration, if there are no labelled
    # samples, this can be empty.
    #
    # Example:
    #
    # LABELED_INDICES = np.array(
    #     [10, 42, 91, 103, ...],
    #     dtype=np.int64
    # )
    #
    # --------------------------------------------------------

    LABELED_INDICES = np.array([], dtype=np.int64)
    BUDGET = 20

    clusters, uncovered = run_scan(
        embeddings_path=EMBEDDINGS_PATH,
        labeled_indices=LABELED_INDICES,
        budget=BUDGET,
        max_clusters=MAX_CLUSTERS,
        k_neighbors=K_NEIGHBORS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        entropy_weight=ENTROPY_WEIGHT,
        num_workers=NUM_WORKERS,
        output_assignments_path=CLUSTER_ASSIGNMENTS_PATH,
        output_uncovered_path=UNCOVERED_CLUSTERS_PATH,
        output_head_path=SCAN_HEAD_PATH
    )

    print()
    print("First 20 cluster assignments:")
    print(clusters[:20])

    print()
    print("Uncovered cluster IDs:")
    print(uncovered)
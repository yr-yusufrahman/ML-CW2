"""
Step 3 of TypiClust/TPC-DC: Querying & labelling.

Algorithm (as specified):
  Because max_clusters caps K, we are not guaranteed B uncovered clusters.
  To handle this, points are added iteratively until the budget is
  exhausted:
    (1) Among clusters with the fewest labelled points, and of size > 5,
        select the largest one.
    (2) Compute the typicality of every (unlabelled) point in that
        cluster, using min(20, cluster_size) nearest neighbours.
    (3) Add the point with the highest typicality to the query set.
  Each time a point is added, that cluster's labelled-point count is
  bumped, so step (1) naturally spreads queries across clusters first,
  and only revisits a cluster once every valid cluster has been touched
  at least as many times.

Labelling is a separate function: given the query set, reveal the
ground-truth CIFAR-10 label for each queried index (the oracle step).
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


# ============================================================
# Step 3a: Querying
# ============================================================

def _most_typical_unlabelled(
    cluster_id: int,
    cluster_assignments: np.ndarray,
    embeddings: torch.Tensor,
    excluded: set,
    k_neighbors: int,
) -> Tuple[Optional[int], Optional[float]]:
    """
    Within a single cluster, find the point (not in `excluded`) with the
    highest typicality: the inverse of its mean Euclidean distance to
    its min(k_neighbors, cluster_size) nearest neighbours *within the
    cluster*.

    Returns (global_index, typicality), or (None, None) if every point
    in the cluster is already labelled/queried.
    """
    cluster_idx_global = np.where(cluster_assignments == cluster_id)[0]

    unlabeled_idx_global = np.array(
        [idx for idx in cluster_idx_global if idx not in excluded],
        dtype=np.int64,
    )

    if len(unlabeled_idx_global) == 0:
        return None, None

    cluster_emb = embeddings[cluster_idx_global]
    unlabeled_emb = embeddings[unlabeled_idx_global]

    cluster_size = len(cluster_idx_global)
    k = min(k_neighbors, cluster_size)

    # Distance from each unlabelled candidate to every point in the
    # cluster (including itself -> distance 0, dropped below).
    dist_matrix = torch.cdist(unlabeled_emb, cluster_emb, p=2.0)
    sorted_dist, _ = torch.sort(dist_matrix, dim=1)

    knn_dist = sorted_dist[:, 1:k + 1]
    avg_dist = knn_dist.mean(dim=1)
    typicality = 1.0 / (avg_dist + 1e-8)

    best_local_idx = torch.argmax(typicality).item()
    best_global_idx = int(unlabeled_idx_global[best_local_idx])

    return best_global_idx, float(typicality[best_local_idx].item())


def get_typical_queries(
    embeddings_path: str,
    cluster_assignments: np.ndarray,
    labeled_indices: List[int],
    budget: int,
    min_cluster_size: int = 5,
    k_neighbors: int = 20,
    uncovered_clusters: Optional[np.ndarray] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> List[int]:
    """
    Select `budget` query points by iteratively picking, at each step,
    the most typical unlabelled point from the largest cluster among
    those with the fewest labelled points (restricted to clusters with
    more than `min_cluster_size` samples).

    `uncovered_clusters` is optional and only used for a sanity-check
    print comparing against Step 2's output; it is not required by the
    selection algorithm itself, since labelled-point counts per cluster
    are recomputed here from `cluster_assignments` + `labeled_indices`.
    """
    print(f"Executing Step 3 on {device}...")

    # --------------------------------------------------------
    # Load & normalise embeddings (same space Step 2 clustered in)
    # --------------------------------------------------------
    embeddings = torch.from_numpy(np.load(embeddings_path)).float().to(device)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    cluster_assignments = np.asarray(cluster_assignments)
    num_clusters = int(cluster_assignments.max()) + 1
    cluster_sizes = np.bincount(cluster_assignments, minlength=num_clusters)

    # --------------------------------------------------------
    # Labelled-point count per cluster (this is what step (1) ranks on)
    # --------------------------------------------------------
    labeled_counts = np.zeros(num_clusters, dtype=np.int64)
    for idx in labeled_indices:
        labeled_counts[cluster_assignments[idx]] += 1

    if uncovered_clusters is not None:
        actually_uncovered = np.sum(labeled_counts == 0)
        print(
            f"Sanity check: Step 2 reported {len(uncovered_clusters)} uncovered "
            f"clusters; {actually_uncovered} clusters currently have 0 labelled points."
        )

    # Clusters with a reliable-enough typicality estimate.
    valid_clusters = set(
        int(c) for c in range(num_clusters) if cluster_sizes[c] > min_cluster_size
    )
    dropped = num_clusters - len(valid_clusters)
    if dropped > 0:
        print(f"Dropped {dropped} cluster(s) with <= {min_cluster_size} samples.")

    excluded = set(int(i) for i in labeled_indices)  # labelled + queried-this-round
    queries: List[int] = []

    while len(queries) < budget:

        if not valid_clusters:
            print(
                f"Warning: ran out of valid clusters before budget was "
                f"exhausted. Selected {len(queries)}/{budget}."
            )
            break

        # (1) Among clusters with the fewest labelled points, select the largest.
        min_labels = min(labeled_counts[c] for c in valid_clusters)
        tied_clusters = [c for c in valid_clusters if labeled_counts[c] == min_labels]
        selected_cluster = max(tied_clusters, key=lambda c: cluster_sizes[c])

        # (2) + (3) Typicality of every point in the cluster (k = min(20, size)),
        # take the highest.
        best_global_idx, typicality = _most_typical_unlabelled(
            selected_cluster, cluster_assignments, embeddings, excluded, k_neighbors
        )

        if best_global_idx is None:
            # Every point in this cluster is already labelled/queried; drop it.
            valid_clusters.discard(selected_cluster)
            continue

        queries.append(best_global_idx)
        excluded.add(best_global_idx)
        labeled_counts[selected_cluster] += 1

        print(
            f"Query {len(queries)}/{budget}: Selected Image ID {best_global_idx} "
            f"from Cluster {selected_cluster} "
            f"(size {cluster_sizes[selected_cluster]}, "
            f"labelled-in-cluster now {labeled_counts[selected_cluster]}, "
            f"Typicality: {typicality:.4f})"
        )

    return queries


# ============================================================
# Step 3b: Labelling (oracle step)
# ============================================================

def label_queries(query_indices: List[int], dataset: torchvision.datasets.CIFAR10) -> Dict[int, int]:
    """
    Reveal the ground-truth label for each queried index.

    In a real AL loop this is where a human annotator would provide the
    label; here we simulate the oracle by reading the true CIFAR-10
    label already attached to the dataset. Uses `dataset.targets`
    directly (a plain list of ints) rather than `dataset[idx]`, avoiding
    decoding every queried image just to read its label.
    """
    return {int(idx): int(dataset.targets[idx]) for idx in query_indices}


if __name__ == "__main__":
    # Example execution continuing from Step 2
    MOCK_LABELED_INDICES: List[int] = []
    BUDGET = 20

    assignments = np.load('features/cluster_assignments.npy')
    uncovered = np.load('features/uncovered_clusters.npy')  # optional, for the sanity check only

    new_queries = get_typical_queries(
        embeddings_path='features/cifar10_simclr_features.npy',
        cluster_assignments=assignments,
        labeled_indices=MOCK_LABELED_INDICES,
        budget=BUDGET,
        uncovered_clusters=uncovered,
    )
    print(f"\nSelected {len(new_queries)} queries: {new_queries}")

    # --------------------------------------------------------
    # Labelling: reveal the oracle labels for the query set
    # --------------------------------------------------------
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    class_names = dataset.classes

    revealed_labels = label_queries(new_queries, dataset)
    print("\nRevealed labels:")
    for idx, label in revealed_labels.items():
        print(f"  ID {idx}: {class_names[label]} (class {label})")

    # Update the labelled pool for the next AL iteration: L_i = L_{i-1} + queries
    updated_labeled_indices = sorted(set(MOCK_LABELED_INDICES) | set(new_queries))
    np.save('features/labeled_indices.npy', np.array(updated_labeled_indices, dtype=np.int64))
    print(f"\nLabelled pool size after this round: {len(updated_labeled_indices)}")
    print("Saved updated labelled indices to: features/labeled_indices.npy")

    # --------------------------------------------------------
    # Display up to 3 of the selected query images with revealed labels
    # --------------------------------------------------------
    display_indices = new_queries[:3]
    n = len(display_indices)

    if n == 0:
        print("No queries to display.")
    else:
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]
        for ax, idx in zip(axes, display_indices):
            image, _ = dataset[idx]
            label = revealed_labels[idx]
            ax.imshow(image)
            ax.set_title(f"ID {idx}\n{class_names[label]}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()
'''
Executes 5 iterations of active learning, where each iteration involves:
1. Representation learning - producing embeddings for the unlabeled data
2. Clustering - clustering the embeddings into clusters
3. Querying & labelling - query the oracle for labels for the most typical examples
   from the most uncovered, largest clusters
4. Update datasets - remove the labelled examples from the unlabeled dataset
   and add them to the labeled dataset
'''
import importlib.util
from pathlib import Path

import numpy as np
import torchvision


def _load_module(module_name: str, filename: str):
    """Load a local .py file that may have a hyphenated name."""
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


representation_learning = _load_module(
    "representation_learning", "representation-learning.py"
)
clustering = _load_module("clustering", "clustering.py")
querying = _load_module("querying", "querying.py")

AL_ITERATIONS = 5
BUDGET = 20
DATA_DIR = "./data"
CHECKPOINT_PATH = "pretrained-models/simclr_cifar-10.pth.tar"
FEATURES_DIR = Path("features")
UNLABELED_POOL_PATH = FEATURES_DIR / "unlabeled_pool.npy"
LABELED_POOL_PATH = FEATURES_DIR / "labeled_pool.npy"
LABELED_LABELS_PATH = FEATURES_DIR / "labeled_labels.npy"


def main():
    clustering.set_seed(clustering.SEED)

    # Oracle dataset (CIFAR-10 train). Indices are global into this set.
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True
    )
    n_samples = len(dataset)

    # --------------------------------------------------------
    # Active-learning pools (owned here, updated each iteration)
    # --------------------------------------------------------
    # U_0 = full train set; L_0 = empty
    unlabeled_indices = np.arange(n_samples, dtype=np.int64)
    labeled_indices = np.array([], dtype=np.int64)
    labeled_labels = {}  # global index -> CIFAR-10 class

    print("=" * 70)
    print("TPC-DC ACTIVE LEARNING")
    print("=" * 70)
    print(f"Pool size |U0| = {len(unlabeled_indices):,}")
    print(f"Initial  |L0| = {len(labeled_indices):,}")
    print(f"Iterations    = {AL_ITERATIONS}")
    print(f"Budget B      = {BUDGET}")

    for iteration in range(1, AL_ITERATIONS + 1):
        print()
        print("=" * 70)
        print(f"ACTIVE LEARNING ITERATION {iteration}/{AL_ITERATIONS}")
        print("=" * 70)
        print(f"|L| = {len(labeled_indices):,}   |U| = {len(unlabeled_indices):,}")

        # ----------------------------------------------------
        # 1. Representation learning (in-memory embeddings)
        # ----------------------------------------------------
        print()
        print("-" * 70)
        print("Step 1: Representation learning")
        print("-" * 70)
        embeddings = representation_learning.extract_features(
            checkpoint_path=CHECKPOINT_PATH,
            data_dir=DATA_DIR,
        )

        # ----------------------------------------------------
        # 2. Clustering (K = min(|L| + B, max_clusters))
        # ----------------------------------------------------
        print()
        print("-" * 70)
        print("Step 2: Clustering")
        print("-" * 70)
        cluster_assignments, uncovered_clusters = clustering.run_scan(
            embeddings=embeddings,
            labeled_indices=labeled_indices,
            budget=BUDGET,
            max_clusters=clustering.MAX_CLUSTERS,
            k_neighbors=clustering.K_NEIGHBORS,
            epochs=clustering.EPOCHS,
            batch_size=clustering.BATCH_SIZE,
            learning_rate=clustering.LEARNING_RATE,
            entropy_weight=clustering.ENTROPY_WEIGHT,
            num_workers=0,  # required on Windows with importlib-loaded modules
        )

        # ----------------------------------------------------
        # 3. Querying & labelling
        # ----------------------------------------------------
        print()
        print("-" * 70)
        print("Step 3: Querying & labelling")
        print("-" * 70)
        queries = querying.get_typical_queries(
            embeddings=embeddings,
            cluster_assignments=cluster_assignments,
            labeled_indices=labeled_indices.tolist(),
            budget=BUDGET,
            uncovered_clusters=uncovered_clusters,
        )

        if len(queries) == 0:
            print("No queries selected; stopping early.")
            break

        queries_arr = np.asarray(queries, dtype=np.int64)
        not_in_pool = np.setdiff1d(queries_arr, unlabeled_indices)
        if len(not_in_pool) > 0:
            raise RuntimeError(
                f"Query indices not in the unlabelled pool: {not_in_pool.tolist()}"
            )

        revealed_labels = querying.label_queries(queries, dataset)
        labeled_labels.update(revealed_labels)

        print()
        print(f"Oracle labelled {len(revealed_labels)} queries:")
        for idx in queries:
            class_id = revealed_labels[idx]
            print(f"  ID {idx}: {dataset.classes[class_id]} (class {class_id})")

        # ----------------------------------------------------
        # 4. Update labelled / unlabelled pools
        # ----------------------------------------------------
        labeled_indices = np.concatenate([labeled_indices, queries_arr])
        unlabeled_indices = np.setdiff1d(unlabeled_indices, queries_arr)

        print()
        print(
            f"After iteration {iteration}: "
            f"|L| = {len(labeled_indices):,}   |U| = {len(unlabeled_indices):,}"
        )

    print()
    print("=" * 70)
    print("ACTIVE LEARNING COMPLETE")
    print("=" * 70)
    print(f"Final |L| = {len(labeled_indices):,}")
    print(f"Final |U| = {len(unlabeled_indices):,}")
    label_array = np.array(
        [labeled_labels[int(idx)] for idx in labeled_indices],
        dtype=np.int64,
    )
    if len(labeled_indices) > 0:
        print(f"Label distribution: {np.bincount(label_array, minlength=10).tolist()}")

    # Persist final pools for downstream use
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(UNLABELED_POOL_PATH, unlabeled_indices)
    np.save(LABELED_POOL_PATH, labeled_indices)
    np.save(LABELED_LABELS_PATH, label_array)
    print()
    print(f"Saved unlabelled pool → {UNLABELED_POOL_PATH}  ({len(unlabeled_indices):,} indices)")
    print(f"Saved labelled pool   → {LABELED_POOL_PATH}  ({len(labeled_indices):,} indices)")
    print(f"Saved labelled labels → {LABELED_LABELS_PATH}  ({len(label_array):,} labels)")


if __name__ == "__main__":
    main()

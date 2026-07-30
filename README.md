# ML-CW2
Recreation of a variant of the TypiClust from a research paper. Then a modification made to it.

# Overview of the TPC-DC 

CIFAR-10
   │
   ▼
Unlabelled pool U
   │
   ▼
1. SCAN representation learning
   │
   ▼
Semantic embeddings
   │
   ▼
2. SCAN clustering
   │
   ▼
B clusters
   │
   ▼
3. Calculate Typicality
   │
   ▼
4. Find B largest uncovered clusters
   │
   ▼
5. Select most typical example from each
   │
   ▼
Queries


SCAN RepLearn:
You used a pretrained SCAN/SimCLR from the repo for the CIFAR-10. Final output of this step were embeddings, feature space.

SCAN Clustering:
Train a new SCAN to perform clustering on the embeddings, with your given budget. Params were saved for this clustering model (but it was quick to train!)

- To load your pretrained clustering model:
def load_model_for_demo(embeddings_path, model_path, num_clusters):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize the empty architecture 
    # (num_clusters must match what you used during training)
    demo_model = SCANClusteringHead(input_dim=512, num_clusters=num_clusters).to(device)
    
    # 2. Load the saved weights
    demo_model.load_state_dict(torch.load(model_path, map_location=device))
    demo_model.eval()
    print(f"Successfully loaded model from {model_path}")
    
    # 3. Run inference (demo)
    embeddings = torch.tensor(np.load(embeddings_path), device=device)
    
    with torch.no_grad():
        all_logits = demo_model(embeddings)
        cluster_assignments = torch.argmax(all_logits, dim=1).cpu().numpy()
        
    return cluster_assignments

Example usage in your demo script:
assignments = load_model_for_demo('features/cifar10_simclr_features.npy', 'scan_clustering_head.pth', num_clusters=20)



Querying: TO-DO
Iterate through each cluster and add the max typicality example to your query list/set

---

THAT COMPLETES THE GENERAL TYPICLUST ALGORITHM - You now have a set of typical points in your queries set, ready to be labeled.

# NEXT STAGE: 3 FRAMEWORKS

What do you do with your queries?

What happened next in the research paper?

Supervised, Unsupervised, Semi-supervised?

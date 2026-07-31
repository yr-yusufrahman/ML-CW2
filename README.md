# ML-CW2
Recreation of a variant of the TypiClust from a research paper. Then a modification made to it.

# Overview of the TPC-DC 

CIFAR-10 --> Representation learning --> Embeddings --> Clustering --> Clusters --> Querying --> Oracle Labelling --> Labelled set of typical examples

A supervised learning model is then train on the labelled set

## SCAN RepLearn:
You used a pretrained SCAN/SimCLR from the exact repo cited in the paper, for the CIFAR-10. Final output of this step were embeddings, feature space.


## SCAN Clustering:
Train a new SCAN to perform clustering on the embeddings, with your given budget. Params were saved for this clustering model (but it was quick to train!)

## Querying & labelling:
Iterate through each cluster and add the max typicality example to your query list/set

Reveal labels to act as an oracle


---

THAT COMPLETES THE GENERAL TYPICLUST ALGORITHM - You now have a set of typical points in your queries set, ready to be labeled.

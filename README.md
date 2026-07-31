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

# How to run

The main entry-point for running the TypiClust algorithm is the `active-learning.py` script. This script implements the full active learning workflow, including clustering the embeddings and selecting the most typical examples from each cluster.

Running `active-learning.py` will build up a labelled pool of typical examples from the data. You'll interact with the oracle to reveal the true labels of these selected samples, populating your labelled set.

Once you have generated the labelled pool, you can then train a downstream supervised model using this data. To do this, use the relevant training script found in the `framework/` directory, which will let you train your model of choice on only the labelled typical examples.

If you wish to use the slightly modified version of the pipeline—where the representation learning step is *decoupled* from the active learning loop—you should run the `active-learning-mode.py` script instead. This script implements a version where representation learning is performed separately before the active selection and labelling.

For further details, check the comments and docstrings in each script for required arguments and configuration steps.


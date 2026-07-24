# Copilot Adaptation Plan: Step 1 (Heterogeneous Graph Extension)

## Project Context
This repository is a fork of the original `C:/git/mamintoosi/Overlapping-Cluster-GCN` project. 
- **Original Work**: Efficient GCN training on **Homogeneous** graphs using overlapping clustering.
- **Current Objective (Step 1)**: Generalize the existing codebase to support **Heterogeneous** Graphs (graphs with multiple node types and edge types) while **preserving the core overlapping clustering mechanism**.
- I have to borrow some idea from this paper: https://arxiv.org/html/2607.03097v1 and enhance my previous paper:https://jac.ut.ac.ir/article_85195.html
The following instructions are the initial suggestion for this matter, but if you have a better idea, inform me.

## Target Datasets (for testing)
We will validate the changes on standard Heterogeneous datasets:
- ACM (Paper, Author, Subject)
- DBLP (Paper, Author, Conference, Term)
- IMDB (Movie, Actor, Director)

## Required Modifications (Please execute sequentially)

### Phase 1: Data Loading & Representation
1.  **Update Dependencies**: Ensure `torch-geometric` or `dgl` is updated. We will use **PyG's `HeteroData`** object as the standard input format.
2.  **Refactor `data_loader.py`**:
    - Replace the homogeneous adjacency matrix loading with a function that reads `HeteroData`.
    - The function should extract:
      - `node_types`: A list/dictionary of node types.
      - `edge_index_dict`: A dictionary of edge types (e.g., `{('author', 'writes', 'paper'): edge_index}`).
      - `node_features`: A dictionary where keys are node types and values are feature matrices.
3.  **Target Node Selection**: Add a parameter `target_node_type` (e.g., "paper") to specify which node type is used for the classification task.

### Phase 2: Feature Unification (Crucial for Clustering)
The original overlapping clustering algorithm relies on a unified feature space to compute similarities.
1.  **Create `models/projector.py`**:
    - Since different node types have different feature dimensions, implement a simple linear projection (MLP) that projects *all* node features into a unified latent space (e.g., `hidden_dim=64`).
    - This ensures we can run the overlapping clustering across all nodes in the graph, regardless of their type.

### Phase 3: Modifying the Overlapping Clustering Algorithm
1.  Locate the `overlapping_cluster()` function.
2.  Modify the input so it accepts the **unified embedding matrix** (the output of the Projector).
3.  **Crucial**: The clustering logic itself (the mathematical formulation of overlapping memberships) **must remain unchanged**. We are only changing *what* is being fed into it (from raw homogeneous features to projected heterogeneous features).
4.  Ensure the output stores the `cluster_assignments` for ALL nodes (including non-target nodes like authors/conferences).

### Phase 4: Heterogeneous Graph Construction for Training
1.  **Create `utils/graph_construction.py`**:
    - Based on the overlapping cluster assignments, build a coarsened or partitioned sub-graph for training.
    - Unlike the homogeneous version where we just sample nodes, here we must ensure that for a cluster containing target nodes, we **also include the connected non-target nodes** (authors, conferences) as context.
2.  Modify the training sampler to pass the `edge_index_dict` (hetero edges) to the GCN layers, rather than a single `edge_index`.

### Phase 5: Model Architecture Update
1.  Replace the standard `GCNConv` with **`RGCNConv`** or **`HeteroConv`** (from PyG) in the `GCN` model class.
2.  Update the forward pass to accept a dictionary of tensors (`x_dict`) and an `edge_index_dict`.
3.  Ensure the output projection only computes the loss for the `target_node_type`.

### Phase 6: Evaluation & Metrics
1.  Update the evaluation loop to filter predictions and labels strictly for the `target_node_type`.
2.  Run a sanity check: Compare the training time and accuracy against a standard full-batch Heterogeneous GCN (baseline) to ensure the overlapping clustering accelerates the training.

---

## Success Criteria for Step 1
- The code runs without errors on DBLP/ACM.
- The model utilizes overlapping clusters to partition the heterogeneous graph.
- The training time is significantly lower than a full-batch RGCN.
- Accuracy does not drop drastically compared to the full-batch RGCN.

---

## Waiting for Step 2
Once Step 1 is fully functional, I will provide additional instructions to inject the "Role-Aware" logic (inspired by HGC-RC) into this new heterogeneous pipeline.

<!-- Start of Step 1 Implementation -->
<!-- @Copilot: Please start modifying the code according to Phase 1 to Phase 6. -->

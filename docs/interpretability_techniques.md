# Additional GNN Interpretability Techniques

Reference document for interpretability methods applicable to HybridGCN (3 GCNConv layers + attention pooling) for hallucination detection on bipartite AMR alignment graphs.

## What We've Already Done

| Phase | Technique | Key Finding |
|-------|-----------|-------------|
| Phase 1 | Statistical baselines (LR, RF) | LR F1=0.676, RF F1=0.666; GNN adds ~0.06 F1 |
| Phase 2 | Structural pattern analysis | 15/27 features significant; hallu graphs sparser, more modular |
| Phase 3 | Ablation studies (A1-A6) | SBERT most important (-0.035 F1); alignment edges slightly harmful (+0.015 F1 when removed) |
| Phase 4 | Gradient saliency, Integrated Gradients | component_type > node_type > SBERT per-dim; edge attrs unused by GCNConv |
| Phase 5 | Linear probing, CKA, cosine similarity | Separation emerges at layer 3 (F1 0.60 -> 0.68); embeddings in narrow cone |

---

## 1. Neuron Activation Analysis ("What Lights Up")

### 1A. Per-Neuron Concept Alignment

**What:** Correlate each of the 256 neurons per GCN layer with domain concepts (alignment degree, flow values, node type, bipartite membership). Identifies "concept detector" neurons.

**Question answered:** "Does neuron 47 in layer 2 fire specifically for unaligned summary nodes?"

**Applicable to GCNConv?** Yes. Uses existing `LayerEmbeddingExtractor` hook infrastructure.

**Library:** Custom code on existing hooks. Methodology from [Xuanyuan et al., AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26267).

**Difficulty:** LOW-MEDIUM

**Sketch:**
```python
from scipy.stats import pointbiserialr

# For each graph, record activation of neuron j at layer k
# Correlate with concept c (e.g., "high alignment degree")
for neuron_idx in range(hidden_dim):
    activations = [act[layer_name][:, neuron_idx].mean().item() for act in all_activations]
    r, p = pointbiserialr(concept_labels, activations)
```

### 1B. Activation Maximization for Graphs

**What:** Find/generate graph inputs that maximally activate a specific neuron or class output. Two approaches: (a) search dataset for max-activating graphs, (b) gradient-optimize node features with fixed topology.

**Question answered:** "What graph pattern does the model consider the ideal hallucination?"

**Applicable to GCNConv?** Yes.

**Library:** Custom. Approach (a) trivially uses hook infrastructure.

**Difficulty:** LOW (dataset search) / MEDIUM (optimization)

### 1C. Activation Atlas / Co-activation Graphs

**What:** Build a co-activation graph where neurons are nodes, edges represent activation correlations. Network analysis reveals functional neuron groups.

**Question answered:** "Which neurons work together as circuits?"

**Library:** Custom + networkx. Inspired by [Sedighin et al.](https://www.sciencedirect.com/science/article/pii/S0167739X21000613).

**Difficulty:** MEDIUM

---

## 2. Subgraph-Level Explanation Methods

### 2A. PGExplainer (Parameterized Explainer)

**What:** Trains a neural network to predict edge importance masks. Unlike GNNExplainer (per-instance), PGExplainer is amortized -- trained once, explains new instances in a single forward pass.

**Question answered:** "Which edges are globally important for the model's decisions?"

**Applicable to GCNConv?** Yes, fully supported in PyG.

**Library:** `torch_geometric.explain.algorithm.PGExplainer`

**Difficulty:** LOW

```python
from torch_geometric.explain import Explainer, PGExplainer

explainer = Explainer(
    model=model,
    algorithm=PGExplainer(epochs=30, lr=0.003),
    explanation_type='phenomenon',
    edge_mask_type='object',
    model_config=dict(mode='binary_classification', task_level='graph', return_type='raw'),
)
# Training phase
for epoch in range(30):
    for data in train_loader:
        loss = explainer.algorithm.train(epoch, model, data.x, data.edge_index,
                                          target=data.y, batch=data.batch)
# Inference
explanation = explainer(data.x, data.edge_index, target=data.y, batch=data.batch)
```

### 2B. SubgraphX (Monte Carlo Tree Search)

**What:** Uses MCTS to find the most explanatory connected subgraph for a prediction. Scores subgraphs with Shapley values.

**Question answered:** "What is the minimal connected subgraph that explains this prediction?"

**Library:** [DIG library](https://diveintographs.readthedocs.io/en/latest/tutorials/subgraphx.html) (`dig.xgraph.method.SubgraphX`)

**Difficulty:** MEDIUM (DIG integration, computationally expensive per graph)

```python
from dig.xgraph.method import SubgraphX
explainer = SubgraphX(model, num_classes=2, device=device, explain_graph=True)
explanation = explainer(data.x, data.edge_index, edge_attr=data.edge_attr)
```

### 2C. GraphMaskExplainer

**What:** Learns layer-wise differentiable edge masks at each GCN layer. Key: importance is layer-specific -- an edge might matter in layer 1 but not layer 3.

**Question answered:** "Which edges matter at which layer of message passing?"

**Library:** `torch_geometric.explain.algorithm.GraphMaskExplainer` (PyG contrib)

**Difficulty:** LOW-MEDIUM

```python
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GraphMaskExplainer

explainer = Explainer(
    model=model,
    algorithm=GraphMaskExplainer(num_layers=3, epochs=10),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(mode='binary_classification', task_level='graph', return_type='raw'),
)
```

### 2D. AttentionExplainer

**What:** Extracts attention weights from the attention pooling component. Shows which nodes receive high attention during readout.

**Question answered:** "Which nodes does the attention pooling focus on?"

**Library:** `torch_geometric.explain.algorithm.AttentionExplainer`

**Difficulty:** LOW

---

## 3. Concept-Based Explanations

### 3A. GRAPHTRAIL (NeurIPS 2024)

**What:** Translates GNN predictions into human-interpretable boolean formulas over subgraph-level concepts. Automatically mines discriminative subgraph concepts via Shapley values.

**Question answered:** "What logical rule does the model implement?" e.g., `hallucination = (low_alignment_coverage AND high_unmatched_summary_nodes)`

**Library:** [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/df2d51e1d3e899241c5c4c779c1d509f-Paper-Conference.pdf). Custom integration.

**Difficulty:** HIGH

### 3B. Concept Bottleneck / Concept Distillation

**What:** Define human-interpretable concepts (alignment fraction, mean flow, density) and train a concept bottleneck that predicts concepts first, then classifies. Post-hoc variant: train embeddings -> concepts -> prediction without retraining.

**Question answered:** "Can the model's decision be explained entirely through human-defined concepts?"

**Library:** Custom. Phase 2 structural features serve as the concept vocabulary.

**Difficulty:** MEDIUM

```python
# Post-hoc approach:
embeddings = extractor.extract_graph_embeddings(dataset)  # existing
concepts = structural_analyzer.extract_batch(dataset)       # existing
# Train: embeddings -> concepts -> prediction
# Analyze: which concepts are sufficient?
```

### 3C. GNN Neuron-Level Concept Detection (AAAI 2023)

**What:** Measures alignment between neuron activations and logical compositions of graph properties (node degree, neighborhood structure).

**Question answered:** "Do neurons learn interpretable graph properties?"

**Library:** Custom. [Xuanyuan et al.](https://ojs.aaai.org/index.php/AAAI/article/view/26267).

**Difficulty:** MEDIUM

---

## 4. Counterfactual Explanations

### 4A. CF-GNNExplainer

**What:** Finds the minimal set of edge deletions that flips the prediction. For hallucination detection: "What edges would need to change to make this hallucination look truthful?"

**Applicable to GCNConv?** Yes, model-agnostic perturbation method.

**Library:** [GitHub](https://github.com/a-lucic/cf-gnnexplainer). AISTATS 2022.

**Difficulty:** MEDIUM (reference impl targets node classification; needs adaptation for graph classification)

### 4B. GCFExplainer (Global Counterfactual)

**What:** Finds global counterfactual patterns that apply across many instances. Identifies common minimal changes that flip predictions.

**Question answered:** "What general structural change turns hallucination graphs into truthful ones?"

**Library:** [Kosan et al., ACM TIST 2024](https://www.mertkosan.com/docs/kosan2023gcfexplainer.pdf).

**Difficulty:** HIGH

### 4C. COMBINEX (2025)

**What:** Jointly considers node feature perturbations AND structural perturbations (edge add/delete).

**Library:** [ArXiv 2025](https://arxiv.org/html/2502.10111v1).

**Difficulty:** HIGH

---

## 5. Layer-wise Relevance Propagation (LRP)

### 5A. GNN-LRP (Walk-Based Relevance)

**What:** Decomposes the model's output into contributions from individual "walks" (length-L paths through the graph during L layers of message passing). With 3 GCN layers, walks are length-3 paths like: `source_concept -> alignment_edge -> summary_concept -> internal_edge -> summary_modifier`.

**Question answered:** "Which specific message-passing paths drove this prediction?"

**Applicable to GCNConv?** Yes. GCNConv is standard message-passing where LRP decomposition is well-defined.

**Library:** [GitHub: liwenke1/GNN-LRP](https://github.com/liwenke1/GNN-LRP). Also DIG library. [Zennit](https://github.com/chr5tphr/zennit) for LRP primitives.

**Difficulty:** MEDIUM-HIGH

**Why especially valuable:** Walk-level explanations directly map to information flow paths through the bipartite alignment graph. A relevant walk might be: "summary concept 'treatment-01' received information from source concept 'disease-01' via the alignment edge with flow=0.12."

### 5B. Relevant Walk Search (ICML 2023)

**What:** Efficient search for top-k most relevant walks (addresses exponential blowup in GNN-LRP).

**Library:** [Xiong et al., ICML 2023](https://proceedings.mlr.press/v202/xiong23b/xiong23b.pdf).

**Difficulty:** HIGH

---

## 6. Training Data Attribution

### 6A. TracIn (Gradient-based)

**What:** Tracks gradient-dot-product contribution of each training example across checkpoints. Simpler than full influence functions.

**Question answered:** "Which training examples had the most influence on this prediction?"

**Library:** `captum.influence.TracInCPFast`. Requires saving checkpoints during training.

**Difficulty:** MEDIUM

### 6B. Classical Influence Functions

**What:** Estimates which training examples most influenced a prediction via Hessian approximation.

**Library:** Captum, or [LogIX](https://github.com/logix-project/logix) for automation.

**Difficulty:** HIGH

### 6C. TRAK (2023)

**What:** Uses random projections for efficient training data attribution.

**Library:** [TRAK library](https://github.com/MadryLab/trak).

**Difficulty:** MEDIUM

---

## 7. Captum Unified Attribution Suite (via PyG)

PyG's `CaptumExplainer` wraps all Captum methods for graph inputs with one-line changes:

| Method | Code | What it adds over existing work |
|--------|------|---------------------------------|
| DeepLIFT | `CaptumExplainer('DeepLift')` | Reference-based attribution, more stable than gradients |
| DeepLIFT-SHAP | `CaptumExplainer('DeepLiftShap')` | SHAP-value interpretation of DeepLIFT |
| GradCAM | `CaptumExplainer('LayerGradCam')` | Node-level "heatmaps" of feature activation |
| GuidedBackprop | `CaptumExplainer('GuidedBackprop')` | Highlights positive-influence features only |
| Kernel SHAP | `CaptumExplainer('KernelShap')` | Model-agnostic SHAP values |
| Feature Ablation | `CaptumExplainer('FeatureAblation')` | Systematic feature zeroing (more principled than our A1-A6) |
| Shapley Value Sampling | `CaptumExplainer('ShapleyValueSampling')` | Approximated Shapley values per feature |

```python
from torch_geometric.explain import Explainer, CaptumExplainer

explainer = Explainer(
    model=model,
    algorithm=CaptumExplainer('DeepLiftShap'),
    explanation_type='model',
    node_mask_type='attributes',
    model_config=dict(mode='binary_classification', task_level='graph', return_type='raw'),
)
```

**Difficulty:** LOW

---

## 8. Mechanistic Interpretability

### 8A. Sparse Autoencoders on GCN Activations

**What:** Train SAEs on hidden representations to discover interpretable features learned in superposition. The GNN analog of Anthropic's SAE work on LLMs.

**Question answered:** "What interpretable features has the GCN learned beyond what individual neurons encode?"

**Library:** Custom.

**Difficulty:** MEDIUM-HIGH

```python
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, n_features, sparsity_coeff=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_features)
        self.decoder = nn.Linear(n_features, input_dim)
        self.sparsity_coeff = sparsity_coeff

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        recon_loss = (x - x_hat).pow(2).mean()
        sparsity_loss = z.abs().mean()
        return x_hat, z, recon_loss + self.sparsity_coeff * sparsity_loss

# Train on node embeddings from layer k across dataset
# Analyze: which SAE features correlate with hallucination?
```

### 8B. Graph Lottery Tickets (Pruning)

**What:** Find sparse subnetworks that maintain full performance. Reveals which parts of the model are truly necessary.

**Question answered:** "What minimal subnetwork suffices for hallucination detection?"

**Library:** [UGS framework](https://arxiv.org/abs/2102.06790). [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/83c230118e9f6688ba8f20bfef99e6da-Paper-Conference.pdf).

**Difficulty:** HIGH

---

## 9. Model-Level / Generative Explanations

### 9A. XGNN

**What:** RL-based graph generator that produces graphs maximizing target class prediction. Shows what the model thinks is the "ideal" hallucination/truthful graph.

**Library:** [PyG PR #8618](https://github.com/pyg-team/pytorch_geometric/pull/8618). [MAGE extension, NeurIPS 2024](https://arxiv.org/html/2405.12519).

**Difficulty:** HIGH

### 9B. GNNInterpreter (NeurIPS 2024)

**What:** Probabilistic generative model that learns a distribution over graph structures maximizing a target class.

**Library:** [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/99329).

**Difficulty:** HIGH

### 9C. GNAN (Graph Neural Additive Network, NeurIPS 2024)

**What:** Interpretable-by-design GNN where each feature's contribution is modeled by a separate sub-network. A replacement for HybridGCN that is inherently transparent.

**Library:** [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/95111).

**Difficulty:** HIGH (requires training new model)

---

## 10. Other Methods

### 10A. PGMExplainer (Probabilistic Graphical Model)

**What:** Fits a Bayesian network to explain GNN predictions. Identifies conditional dependencies between node perturbations and prediction changes.

**Library:** `torch_geometric.explain.algorithm.PGMExplainer`

**Difficulty:** LOW-MEDIUM

### 10B. TopInG (ICML 2025)

**What:** Uses persistent homology (topological data analysis) to identify discriminative topological features (loops, connected components, voids).

**Library:** Research code. Requires TDA libraries (`giotto-tda`, `gudhi`).

**Difficulty:** HIGH

### 10C. Attention Rollout for GCN + Attention Pooling

**What:** Combines GCNConv message-passing influence with attention pooling weights to get compound node importance across the full network.

**Library:** Custom. Inspired by [Mechanistic Interpretability of Graph Transformers](https://arxiv.org/html/2502.12352v1).

**Difficulty:** MEDIUM

---

## Recommended Priority Order

| Priority | Technique | Effort | Value | Why |
|----------|-----------|--------|-------|-----|
| 1 | Per-neuron concept alignment | LOW-MED | HIGH | Directly answers "what lights up"; uses existing hooks |
| 2 | PGExplainer | LOW | HIGH | Native PyG; global edge importance patterns |
| 3 | GraphMaskExplainer | LOW-MED | HIGH | Layer-wise edge importance at each GCN depth |
| 4 | CaptumExplainer suite | LOW | MED | One-line integration; DeepLIFT, GradCAM, SHAP |
| 5 | CF-GNNExplainer | MED | HIGH | "What edges to change to flip verdict" |
| 6 | GNN-LRP walk-based | MED-HIGH | HIGH | Walk-level explanations match bipartite structure |
| 7 | Concept bottleneck | MED | HIGH | Phase 2 features as concept vocabulary |
| 8 | Sparse autoencoders | MED-HIGH | MED | Novel mechanistic interp for GNNs |
| 9 | TracIn | MED | MED | Training data attribution for debugging |
| 10 | GRAPHTRAIL | HIGH | HIGH | Boolean rules -- holy grail of interpretability |

---

## Key References

- [PyG Explainability Tutorial](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/explain.html)
- [PyG Explain Module Docs](https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html)
- [DIG SubgraphX Tutorial](https://diveintographs.readthedocs.io/en/latest/tutorials/subgraphx.html)
- [GraphXAI Survey 2025](https://link.springer.com/article/10.1007/s00521-025-11054-3)
- [GNN Explainability Taxonomic Survey (IEEE TPAMI)](https://dl.acm.org/doi/10.1109/TPAMI.2022.3204236)
- [GNN-LRP: Relevant Walks](https://arxiv.org/abs/2006.03589)
- [Neuron Concept Detection (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26267)
- [GRAPHTRAIL (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/df2d51e1d3e899241c5c4c779c1d509f-Paper-Conference.pdf)
- [CF-GNNExplainer](https://github.com/a-lucic/cf-gnnexplainer)
- [Captum](https://captum.ai/)
- [Zennit LRP](https://github.com/chr5tphr/zennit)
- [Awesome Graph Explainability Papers](https://github.com/flyingdoog/awesome-graph-explainability-papers)
- [GNNX-BENCH (ICLR 2024)](https://par.nsf.gov/servlets/purl/10524810)

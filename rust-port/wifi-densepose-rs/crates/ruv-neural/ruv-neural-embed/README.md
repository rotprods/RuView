# rUv Neural Embed

Graph embedding generation for brain connectivity states using RuVector format.

## Overview

`ruv-neural-embed` converts brain connectivity graphs into fixed-dimensional vector
representations suitable for downstream classification, clustering, and temporal analysis.
Multiple embedding strategies are provided, each capturing different aspects of graph structure.

## Embedding Methods

| Method | Module | Description | Output Dimension |
|--------|--------|-------------|-----------------|
| **Spectral** | `spectral_embed` | Laplacian eigenvector positional encoding | `k * 4` (mean/std/min/max per eigenvector) |
| **Topology** | `topology_embed` | Hand-crafted topological feature vector | 13 (with all features enabled) |
| **Node2Vec** | `node2vec` | Random-walk co-occurrence SVD embedding | `dim * 2` (mean/std per component) |
| **Combined** | `combined` | Weighted concatenation of multiple methods | Sum of sub-embedder dimensions |
| **Temporal** | `temporal` | Sliding-window context-enriched embedding | `base_dim * 2` (current + context) |

## Distance Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| Cosine Similarity | `cosine_similarity` | Direction similarity in [-1, 1] |
| Euclidean Distance | `euclidean_distance` | L2 norm of difference |
| Manhattan Distance | `manhattan_distance` | L1 norm of difference |
| k-Nearest Neighbors | `k_nearest` | Find k closest embeddings |
| Trajectory Distance | `trajectory_distance` | DTW alignment cost for sequences |

## Usage

```rust
use ruv_neural_embed::spectral_embed::SpectralEmbedder;
use ruv_neural_embed::topology_embed::TopologyEmbedder;
use ruv_neural_embed::combined::CombinedEmbedder;
use ruv_neural_embed::distance::{cosine_similarity, k_nearest};
use ruv_neural_core::traits::EmbeddingGenerator;

// Single-method embedding
let spectral = SpectralEmbedder::new(4);
let embedding = spectral.embed(&brain_graph).unwrap();

// Combined multi-method embedding
let combined = CombinedEmbedder::new()
    .add(Box::new(SpectralEmbedder::new(4)), 1.0)
    .add(Box::new(TopologyEmbedder::new()), 0.5);
let combined_emb = combined.embed(&brain_graph).unwrap();

// Compare embeddings
let sim = cosine_similarity(&emb_a, &emb_b);
let neighbors = k_nearest(&query, &candidates, 5);
```

## RVF Export

```rust
use ruv_neural_embed::rvf_export::{export_rvf, import_rvf};

// Save embeddings
export_rvf(&embeddings, "brain_states.rvf").unwrap();

// Load embeddings
let restored = import_rvf("brain_states.rvf").unwrap();
```

## Features

- `std` (default) -- Standard library support
- `wasm` -- WebAssembly compatibility
- `rvf` -- Extended RVF format support

## License

MIT OR Apache-2.0

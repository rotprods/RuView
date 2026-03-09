# ruv-neural-graph

**rUv Neural** -- Brain connectivity graph construction from neural signals.

Part of the [rUv Neural](https://github.com/ruvnet/RuView) workspace for brain topology analysis.

## Overview

`ruv-neural-graph` transforms multi-channel neural time series data into brain connectivity graphs and computes graph-theoretic metrics used in network neuroscience. It supports built-in brain atlases, sliding-window graph construction, spectral analysis, and temporal dynamics tracking.

## Dependency Diagram

```
ruv-neural-core
    |
    v
ruv-neural-signal
    |
    v
ruv-neural-graph  <-- petgraph
    |
    v
ruv-neural-mincut / ruv-neural-embed / ruv-neural-decoder
```

## Modules

| Module            | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `atlas`           | Brain atlas definitions (Desikan-Killiany 68 regions)        |
| `constructor`     | Graph construction from connectivity matrices and time series|
| `petgraph_bridge` | Convert between `BrainGraph` and petgraph types              |
| `metrics`         | Graph-theoretic metrics (efficiency, clustering, centrality) |
| `spectral`        | Spectral graph properties (Laplacian, Fiedler value)         |
| `dynamics`        | Temporal graph dynamics and topology tracking                |

## Graph Metrics

| Metric                  | Function                   | Description                                      |
|-------------------------|----------------------------|--------------------------------------------------|
| Global efficiency       | `global_efficiency`        | Average inverse shortest path length              |
| Local efficiency        | `local_efficiency`         | Average node-level subgraph efficiency            |
| Clustering coefficient  | `clustering_coefficient`   | Weighted triangle ratio                           |
| Node degree             | `node_degree`              | Weighted degree of a single node                  |
| Degree distribution     | `degree_distribution`      | All node degrees                                  |
| Betweenness centrality  | `betweenness_centrality`   | Fraction of shortest paths through each node      |
| Graph density           | `graph_density`            | Fraction of possible edges present                |
| Small-world index       | `small_world_index`        | sigma = (C/C_rand) / (L/L_rand)                  |
| Modularity              | `modularity`               | Newman modularity Q for a given partition         |
| Graph Laplacian         | `graph_laplacian`          | L = D - A                                         |
| Normalized Laplacian    | `normalized_laplacian`     | L_norm = D^{-1/2} L D^{-1/2}                     |
| Fiedler value           | `fiedler_value`            | Algebraic connectivity (second smallest eigenvalue)|
| Spectral gap            | `spectral_gap`             | lambda_2 - lambda_1                               |

## Usage

```rust
use ruv_neural_graph::{
    AtlasType, BrainGraphConstructor, load_atlas,
    global_efficiency, clustering_coefficient, fiedler_value,
    to_petgraph, TopologyTracker,
};
use ruv_neural_core::graph::ConnectivityMetric;
use ruv_neural_core::signal::FrequencyBand;

// Load the Desikan-Killiany atlas (68 cortical regions)
let parcellation = load_atlas(AtlasType::DesikanKilliany);
assert_eq!(parcellation.num_regions(), 68);

// Build a graph constructor
let constructor = BrainGraphConstructor::new(
    AtlasType::DesikanKilliany,
    ConnectivityMetric::PhaseLockingValue,
    FrequencyBand::Alpha,
)
.with_threshold(0.1);

// Construct a graph from a connectivity matrix
let connectivity = vec![vec![1.0; 68]; 68]; // example: fully connected
let graph = constructor.construct_from_matrix(&connectivity, 0.0);

// Compute metrics
let eff = global_efficiency(&graph);
let cc = clustering_coefficient(&graph);
let fv = fiedler_value(&graph);

// Convert to petgraph for advanced algorithms
let pg = to_petgraph(&graph);

// Track topology over time
let mut tracker = TopologyTracker::new();
tracker.track(&graph);
let transitions = tracker.detect_transitions(0.1);
```

## License

MIT OR Apache-2.0

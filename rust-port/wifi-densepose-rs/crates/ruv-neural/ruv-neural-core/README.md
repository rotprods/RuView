# rUv Neural Core

**Core types, traits, and error types for the ruv-neural brain topology analysis system.**

`ruv-neural-core` is the foundational crate of the ruv-neural workspace. It defines all shared types, error variants, and trait interfaces that downstream crates implement. It has **zero internal dependencies** -- every other ruv-neural crate depends on this one.

## Feature Flags

| Feature  | Default | Description                              |
|----------|---------|------------------------------------------|
| `std`    | Yes     | Standard library support                 |
| `no_std` | No      | Embedded/ESP32 target compatibility      |
| `wasm`   | No      | WebAssembly target support               |
| `rvf`    | No      | RuVector RVF file format extensions      |

## Type Overview

| Module      | Key Types                                                        |
|-------------|------------------------------------------------------------------|
| `error`     | `RuvNeuralError`, `Result<T>`                                    |
| `sensor`    | `SensorType`, `SensorChannel`, `SensorArray`                     |
| `signal`    | `MultiChannelTimeSeries`, `FrequencyBand`, `SpectralFeatures`    |
| `brain`     | `Atlas`, `BrainRegion`, `Hemisphere`, `Lobe`, `Parcellation`     |
| `graph`     | `BrainGraph`, `BrainEdge`, `ConnectivityMetric`                  |
| `topology`  | `MincutResult`, `MultiPartition`, `CognitiveState`, `TopologyMetrics` |
| `embedding` | `NeuralEmbedding`, `EmbeddingMetadata`, `EmbeddingTrajectory`    |
| `rvf`       | `RvfFile`, `RvfHeader`, `RvfDataType`                           |

## Trait Overview

| Trait                | Purpose                                        |
|----------------------|------------------------------------------------|
| `SensorSource`       | Read chunks from hardware or simulated sensors |
| `SignalProcessor`    | Transform time series (filter, artifact removal)|
| `GraphConstructor`   | Build connectivity graphs from signals         |
| `TopologyAnalyzer`   | Compute mincut, modularity, efficiency         |
| `EmbeddingGenerator` | Map brain graphs to vector space               |
| `StateDecoder`       | Classify cognitive state from embeddings       |
| `NeuralMemory`       | Store and query embedding history              |
| `RvfSerializable`    | Serialize/deserialize to RVF file format       |

## Usage

```rust
use ruv_neural_core::{
    Atlas, BrainGraph, BrainEdge, ConnectivityMetric, FrequencyBand,
    NeuralEmbedding, EmbeddingMetadata, CognitiveState,
    RvfFile, RvfDataType,
    Result,
};

// Build a connectivity graph
let graph = BrainGraph {
    num_nodes: 68,
    edges: vec![BrainEdge {
        source: 0,
        target: 1,
        weight: 0.85,
        metric: ConnectivityMetric::PhaseLockingValue,
        frequency_band: FrequencyBand::Alpha,
    }],
    timestamp: 1000.0,
    window_duration_s: 2.0,
    atlas: Atlas::DesikanKilliany68,
};

let adj = graph.adjacency_matrix();
let density = graph.density();
```

## License

MIT OR Apache-2.0

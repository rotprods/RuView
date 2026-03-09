# rUv Neural — Brain Topology Analysis System

> Quantum sensor integration x RuVector graph memory x Dynamic mincut coherence detection

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]()
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)]()

## Overview

**rUv Neural** is a modular Rust crate ecosystem for real-time brain network topology
analysis. It transforms neural magnetic field measurements from quantum sensors (NV diamond
magnetometers, optically pumped magnetometers) into dynamic connectivity graphs, then uses
minimum cut algorithms to detect cognitive state transitions.

This is not mind reading -- it measures **how cognition organizes itself** by tracking the
topology of brain networks in real time.

## Architecture

```
                         rUv Neural Pipeline
    ================================================================

    +------------------+     +-------------------+     +------------------+
    |                  |     |                   |     |                  |
    |  SENSOR LAYER    |---->|  SIGNAL LAYER     |---->|  GRAPH LAYER     |
    |                  |     |                   |     |                  |
    |  NV Diamond      |     |  Bandpass Filter  |     |  PLV / Coherence |
    |  OPM             |     |  Artifact Reject  |     |  Brain Regions   |
    |  EEG             |     |  Hilbert Phase    |     |  Connectivity    |
    |  Simulated       |     |  Spectral (PSD)   |     |  Matrix          |
    |                  |     |                   |     |                  |
    +------------------+     +-------------------+     +--------+---------+
                                                                |
                                                                v
    +------------------+     +-------------------+     +------------------+
    |                  |     |                   |     |                  |
    |  DECODE LAYER    |<----|  MEMORY LAYER     |<----|  MINCUT LAYER    |
    |                  |     |                   |     |                  |
    |  Cognitive State |     |  HNSW Index       |     |  Stoer-Wagner    |
    |  Classification  |     |  Pattern Store    |     |  Normalized Cut  |
    |  BCI Output      |     |  Drift Detection  |     |  Spectral Cut    |
    |  Transition Log  |     |  Temporal Window  |     |  Coherence Detect|
    |                  |     |                   |     |                  |
    +------------------+     +-------------------+     +------------------+
                                      ^
                                      |
                              +-------+--------+
                              |                |
                              |  EMBED LAYER   |
                              |                |
                              |  Spectral Pos. |
                              |  Topology Vec  |
                              |  Node2Vec      |
                              |  RVF Export     |
                              |                |
                              +----------------+

    Peripheral Crates:
    +----------+   +----------+   +----------+
    | ESP32    |   | WASM     |   | VIZ      |
    | Edge     |   | Browser  |   | ASCII    |
    | Preproc  |   | Bindings |   | Render   |
    +----------+   +----------+   +----------+
```

## Crate Map

| Crate | Description | Dependencies |
|-------|-------------|--------------|
| `ruv-neural-core` | Core types, traits, errors, RVF format | None |
| `ruv-neural-sensor` | NV diamond, OPM, EEG sensor interfaces | core |
| `ruv-neural-signal` | DSP: filtering, spectral, connectivity | core |
| `ruv-neural-graph` | Brain connectivity graph construction | core, signal |
| `ruv-neural-mincut` | Dynamic minimum cut topology analysis | core |
| `ruv-neural-embed` | RuVector graph embeddings | core |
| `ruv-neural-memory` | Persistent neural state memory + HNSW | core, embed |
| `ruv-neural-decoder` | Cognitive state classification + BCI | core, embed |
| `ruv-neural-esp32` | ESP32 edge sensor integration | core |
| `ruv-neural-wasm` | WebAssembly browser bindings | core |
| `ruv-neural-viz` | Visualization and ASCII rendering | core, graph |
| `ruv-neural-cli` | CLI tool (`ruv-neural` binary) | all |

## Dependency Graph

```
                    ruv-neural-core
                    (types, traits, errors)
                   /    |    |    \     \
                  /     |    |     \     \
                 v      v    v      v     v
           sensor  signal  embed  esp32  (wasm)
                          |
                          v
                  graph --|------> viz
                  |
                  v
               mincut
                  |
                  v
         decoder <--- memory <--- embed
                  |
                  v
                 cli (depends on all)
```

## Quick Start

### Build

```bash
cd rust-port/wifi-densepose-rs/crates/ruv-neural
cargo build --workspace
cargo test --workspace
```

### Run CLI

```bash
cargo run -p ruv-neural-cli -- simulate --channels 64 --duration 10
cargo run -p ruv-neural-cli -- pipeline --channels 32 --duration 5 --dashboard
cargo run -p ruv-neural-cli -- mincut --input brain_graph.json
```

### Use as Library

```rust
use ruv_neural_core::*;
use ruv_neural_sensor::simulator::SimulatedSensorArray;
use ruv_neural_signal::PreprocessingPipeline;
use ruv_neural_mincut::DynamicMincutTracker;
use ruv_neural_embed::NeuralEmbedding;

// Create simulated sensor array (64 channels, 1000 Hz)
let mut sensor = SimulatedSensorArray::new(64, 1000.0);
let data = sensor.acquire(1000)?;

// Preprocess: bandpass filter + artifact rejection
let pipeline = PreprocessingPipeline::default();
let clean = pipeline.process(&data)?;

// Compute connectivity and build graph
let connectivity = ruv_neural_signal::compute_all_pairs(
    &clean,
    ruv_neural_signal::ConnectivityMetric::PhaseLockingValue,
);

// Track topology changes via dynamic mincut
let mut tracker = DynamicMincutTracker::new();
let result = tracker.update(&graph)?;
println!(
    "Mincut: {:.3}, Partitions: {} | {}",
    result.cut_value,
    result.partition_a.len(),
    result.partition_b.len()
);

// Generate embedding for downstream classification
let embedding = NeuralEmbedding::new(
    result.to_feature_vector(),
    data.timestamp,
    "spectral",
)?;
println!("Embedding dim: {}", embedding.dimension);
```

## Mix and Match

Each crate is independently usable. Common combinations:

- **Sensor + Signal** -- Data acquisition and preprocessing only
- **Graph + Mincut** -- Graph analysis without sensor dependency
- **Embed + Memory** -- Embedding storage without real-time pipeline
- **Core + WASM** -- Browser-based graph visualization
- **ESP32 alone** -- Edge preprocessing on embedded hardware
- **Signal + Embed** -- Feature extraction pipeline without graph construction
- **Mincut + Viz** -- Topology analysis with ASCII dashboard output

## Platform Support

| Platform | Status | Crates Available |
|----------|--------|-----------------|
| Linux x86_64 | Full | All 12 |
| macOS ARM64 | Full | All 12 |
| Windows x86_64 | Full | All 12 |
| WASM (browser) | Partial | core, wasm, viz |
| ESP32 (no_std) | Partial | core, esp32 |

**Note:** The `ruv-neural-wasm` crate is excluded from the default workspace members.
Build it separately with:

```bash
cargo build -p ruv-neural-wasm --target wasm32-unknown-unknown --release
```

## Key Algorithms

### Signal Processing (`ruv-neural-signal`)

- **Butterworth IIR filters** in second-order sections (SOS) form
- **Welch PSD** estimation with configurable window and overlap
- **Hilbert transform** for instantaneous phase extraction
- **Artifact detection** -- eye blink, muscle, cardiac artifact rejection
- **Connectivity metrics** -- PLV, coherence, imaginary coherence, AEC

### Minimum Cut Analysis (`ruv-neural-mincut`)

- **Stoer-Wagner** -- Global minimum cut in O(V^3)
- **Normalized cut** (Shi-Malik) -- Spectral bisection via the Fiedler vector
- **Multiway cut** -- Recursive normalized cut for k-module detection
- **Spectral cut** -- Cheeger constant and spectral bisection bounds
- **Dynamic tracking** -- Temporal topology transition detection
- **Coherence events** -- Network formation, dissolution, merger, split

### Embeddings (`ruv-neural-embed`)

- **Spectral** -- Laplacian eigenvector positional encoding
- **Topology** -- Hand-crafted topological feature vectors
- **Node2Vec** -- Random-walk co-occurrence embeddings
- **Combined** -- Weighted concatenation of multiple methods
- **Temporal** -- Sliding-window context-enriched embeddings
- **RVF export** -- Serialization to RuVector `.rvf` format

## RVF Format

RuVector File (RVF) is a binary format for neural data interchange:

```
+--------+--------+---------+----------+----------+
| Magic  | Version| Type    | Payload  | Checksum |
| RVF\x01| u8     | u8      | [u8; N]  | u32      |
+--------+--------+---------+----------+----------+
```

- **Magic bytes**: `RVF\x01`
- **Supported types**: brain graphs, embeddings, topology metrics, time series
- **Binary format** for efficient storage and streaming
- **Compatible** with the broader RuVector ecosystem

## RuVector Integration

rUv Neural integrates with five RuVector crates from the `2.0.4` release:

| RuVector Crate | Used By | Purpose |
|----------------|---------|---------|
| `ruvector-mincut` | mincut | Spectral mincut algorithms |
| `ruvector-attn-mincut` | mincut | Attention-weighted cut |
| `ruvector-temporal-tensor` | signal | Compressed temporal buffers |
| `ruvector-solver` | graph | Sparse interpolation solver |
| `ruvector-attention` | embed | Spatial attention mechanisms |

## Testing

```bash
# Run all workspace tests
cargo test --workspace

# Run a specific crate's tests
cargo test -p ruv-neural-mincut

# Run with logging enabled
RUST_LOG=debug cargo test --workspace -- --nocapture

# Run benchmarks (requires nightly or criterion)
cargo bench -p ruv-neural-mincut
```

## Crate Publishing Order

Crates must be published in dependency order:

1. `ruv-neural-core` (no internal deps)
2. `ruv-neural-sensor` (depends on core)
3. `ruv-neural-signal` (depends on core)
4. `ruv-neural-esp32` (depends on core)
5. `ruv-neural-graph` (depends on core, signal)
6. `ruv-neural-embed` (depends on core)
7. `ruv-neural-mincut` (depends on core)
8. `ruv-neural-viz` (depends on core, graph)
9. `ruv-neural-memory` (depends on core, embed)
10. `ruv-neural-decoder` (depends on core, embed)
11. `ruv-neural-wasm` (depends on core)
12. `ruv-neural-cli` (depends on all)

## License

MIT OR Apache-2.0

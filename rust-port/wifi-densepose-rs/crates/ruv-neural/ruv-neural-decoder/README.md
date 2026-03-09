# rUv Neural Decoder

Cognitive state classification and BCI decoding from neural topology embeddings.

Part of the **rUv Neural** brain-computer interface platform.

## Decoders

| Decoder | Description |
|---------|-------------|
| **KnnDecoder** | K-nearest neighbor classification using stored labeled embeddings with inverse-distance weighting |
| **ThresholdDecoder** | Rule-based classification from topology metric ranges (mincut, modularity, efficiency, entropy) |
| **TransitionDecoder** | Detects cognitive state transitions by matching topology delta patterns against a sliding window |
| **ClinicalScorer** | Biomarker detection via z-score deviation from a learned healthy baseline population |
| **DecoderPipeline** | End-to-end ensemble combining all decoders with configurable weights and clinical scoring |

## Pipeline Architecture

```
NeuralEmbedding ──> KnnDecoder ─────────┐
                                        │
TopologyMetrics ──> ThresholdDecoder ────┤── Weighted Vote ──> DecoderOutput
                │                       │       state, confidence
                ├─> TransitionDecoder ──┘       transition
                │                               brain_health_index
                └─> ClinicalScorer ─────────>   clinical_flags
```

## Usage

```rust
use ruv_neural_decoder::{DecoderPipeline, TopologyThreshold};
use ruv_neural_core::topology::{CognitiveState, TopologyMetrics};

// Build a pipeline with all decoders
let mut pipeline = DecoderPipeline::new()
    .with_knn(5)
    .with_thresholds()
    .with_transitions(10)
    .with_clinical(baseline_metrics, baseline_std);

// Train the KNN decoder
pipeline.knn_mut().unwrap().train(labeled_embeddings);

// Configure threshold ranges
pipeline.threshold_mut().unwrap().set_threshold(
    CognitiveState::Focused,
    TopologyThreshold {
        mincut_range: (7.0, 9.0),
        modularity_range: (0.5, 0.7),
        efficiency_range: (0.4, 0.6),
        entropy_range: (2.5, 3.5),
    },
);

// Decode
let output = pipeline.decode(&embedding, &metrics);
println!("State: {:?} (confidence: {:.2})", output.state, output.confidence);

if let Some(health) = output.brain_health_index {
    println!("Brain health: {:.2}", health);
}
for flag in &output.clinical_flags {
    println!("WARNING: {}", flag);
}
```

## Clinical Applications

The `ClinicalScorer` provides research-grade biomarker detection for:

- **Alzheimer's disease**: Detects network fragmentation (reduced efficiency, increased modularity, reduced mincut)
- **Epilepsy**: Detects hypersynchrony (increased mincut, decreased modularity, increased local efficiency)
- **Depression**: Detects connectivity weakening (reduced efficiency, reduced Fiedler value, altered entropy)
- **Brain Health Index**: Composite score from 0 (severe abnormality) to 1 (healthy baseline)

**Note**: These scores are intended for research use only. Clinical diagnosis requires professional medical evaluation.

## Features

- `std` (default) — Standard library support
- `wasm` — WebAssembly target support

## License

MIT OR Apache-2.0

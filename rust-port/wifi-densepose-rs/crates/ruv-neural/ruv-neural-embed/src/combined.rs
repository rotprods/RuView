//! Combined multi-method embedding.
//!
//! Concatenates weighted embeddings from multiple embedding generators
//! into a single vector representation.

use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::graph::BrainGraph;

use crate::{EmbeddingGenerator, NeuralEmbedding};

/// Combines multiple embedding methods into a single embedding vector.
pub struct CombinedEmbedder {
    embedders: Vec<Box<dyn EmbeddingGenerator>>,
    weights: Vec<f64>,
}

impl CombinedEmbedder {
    /// Create a new empty combined embedder.
    pub fn new() -> Self {
        Self {
            embedders: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Add an embedding generator with a weight.
    ///
    /// The weight scales each element of the generator's output.
    pub fn add(mut self, embedder: Box<dyn EmbeddingGenerator>, weight: f64) -> Self {
        self.embedders.push(embedder);
        self.weights.push(weight);
        self
    }

    /// Number of sub-embedders.
    pub fn num_embedders(&self) -> usize {
        self.embedders.len()
    }

    /// Total embedding dimension (sum of all sub-embedder dimensions).
    pub fn total_dimension(&self) -> usize {
        self.embedders.iter().map(|e| e.dimension()).sum()
    }

    /// Generate a combined embedding by concatenating weighted sub-embeddings.
    pub fn embed(&self, graph: &BrainGraph) -> Result<NeuralEmbedding> {
        if self.embedders.is_empty() {
            return Err(RuvNeuralError::Embedding(
                "CombinedEmbedder has no sub-embedders".into(),
            ));
        }

        let mut values = Vec::with_capacity(self.total_dimension());

        for (embedder, &weight) in self.embedders.iter().zip(self.weights.iter()) {
            let sub_emb = embedder.embed(graph)?;
            for v in &sub_emb.values {
                values.push(v * weight);
            }
        }

        NeuralEmbedding::new(values, graph.timestamp, "combined")
    }
}

impl Default for CombinedEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingGenerator for CombinedEmbedder {
    fn dimension(&self) -> usize {
        self.total_dimension()
    }

    fn method_name(&self) -> &str {
        "combined"
    }

    fn embed(&self, graph: &BrainGraph) -> Result<NeuralEmbedding> {
        CombinedEmbedder::embed(self, graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_embed::SpectralEmbedder;
    use crate::topology_embed::TopologyEmbedder;
    use ruv_neural_core::brain::Atlas;
    use ruv_neural_core::graph::{BrainEdge, ConnectivityMetric};
    use ruv_neural_core::signal::FrequencyBand;

    fn make_test_graph() -> BrainGraph {
        BrainGraph {
            num_nodes: 4,
            edges: vec![
                BrainEdge {
                    source: 0,
                    target: 1,
                    weight: 1.0,
                    metric: ConnectivityMetric::Coherence,
                    frequency_band: FrequencyBand::Alpha,
                },
                BrainEdge {
                    source: 1,
                    target: 2,
                    weight: 0.8,
                    metric: ConnectivityMetric::Coherence,
                    frequency_band: FrequencyBand::Alpha,
                },
                BrainEdge {
                    source: 2,
                    target: 3,
                    weight: 0.6,
                    metric: ConnectivityMetric::Coherence,
                    frequency_band: FrequencyBand::Alpha,
                },
                BrainEdge {
                    source: 0,
                    target: 3,
                    weight: 0.5,
                    metric: ConnectivityMetric::Coherence,
                    frequency_band: FrequencyBand::Alpha,
                },
            ],
            timestamp: 1.0,
            window_duration_s: 1.0,
            atlas: Atlas::Custom(4),
        }
    }

    #[test]
    fn test_combined_concatenates_correctly() {
        let graph = make_test_graph();
        let spectral = SpectralEmbedder::new(2);
        let topo = TopologyEmbedder::new();

        let spectral_dim = spectral.dimension();
        let topo_dim = topo.dimension();

        let combined = CombinedEmbedder::new()
            .add(Box::new(spectral), 1.0)
            .add(Box::new(topo), 1.0);

        assert_eq!(combined.total_dimension(), spectral_dim + topo_dim);

        let emb = combined.embed(&graph).unwrap();
        assert_eq!(emb.dimension, spectral_dim + topo_dim);
        assert_eq!(emb.method, "combined");
    }

    #[test]
    fn test_combined_weights_scale() {
        let graph = make_test_graph();
        let topo = TopologyEmbedder::new();

        // Weight = 2.0
        let combined = CombinedEmbedder::new().add(Box::new(topo), 2.0);
        let emb = combined.embed(&graph).unwrap();

        // Compare with direct topology embedding
        let topo2 = TopologyEmbedder::new();
        let direct = topo2.embed(&graph).unwrap();

        for (c, d) in emb.values.iter().zip(direct.values.iter()) {
            assert!(
                (c - 2.0 * d).abs() < 1e-10,
                "Weight should scale values: {} vs 2*{}",
                c,
                d
            );
        }
    }

    #[test]
    fn test_combined_empty_fails() {
        let graph = make_test_graph();
        let combined = CombinedEmbedder::new();
        assert!(combined.embed(&graph).is_err());
    }
}

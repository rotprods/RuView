//! rUv Neural Embed -- Graph embedding generation for brain connectivity states.
//!
//! This crate provides multiple embedding methods to convert brain connectivity
//! graphs (`BrainGraph`) into fixed-dimensional vector representations suitable
//! for downstream classification, clustering, and temporal analysis.
//!
//! # Embedding Methods
//!
//! - **Spectral**: Laplacian eigenvector-based positional encoding
//! - **Topology**: Hand-crafted topological feature vectors
//! - **Node2Vec**: Random-walk co-occurrence embeddings
//! - **Combined**: Weighted concatenation of multiple methods
//! - **Temporal**: Sliding-window context-enriched embeddings
//!
//! # RVF Export
//!
//! Embeddings can be serialized to the RuVector `.rvf` format for interoperability
//! with the broader RuVector ecosystem.

pub mod combined;
pub mod distance;
pub mod node2vec;
pub mod rvf_export;
pub mod spectral_embed;
pub mod temporal;
pub mod topology_embed;

use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::graph::{BrainGraph, BrainGraphSequence};
use serde::{Deserialize, Serialize};

/// A fixed-dimensional embedding of a brain graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralEmbedding {
    /// The embedding vector.
    pub values: Vec<f64>,
    /// Dimensionality of the embedding.
    pub dimension: usize,
    /// Timestamp of the source graph (Unix seconds).
    pub timestamp: f64,
    /// Name of the method that produced this embedding.
    pub method: String,
    /// Optional metadata (e.g., parameters used).
    pub metadata: Option<String>,
}

impl NeuralEmbedding {
    /// Create a new embedding, validating dimension consistency.
    pub fn new(values: Vec<f64>, timestamp: f64, method: &str) -> Result<Self> {
        let dimension = values.len();
        if dimension == 0 {
            return Err(RuvNeuralError::Embedding(
                "Embedding must have at least one dimension".into(),
            ));
        }
        Ok(Self {
            values,
            dimension,
            timestamp,
            method: method.to_string(),
            metadata: None,
        })
    }

    /// Create a zero embedding of a given dimension.
    pub fn zeros(dimension: usize, timestamp: f64, method: &str) -> Self {
        Self {
            values: vec![0.0; dimension],
            dimension,
            timestamp,
            method: method.to_string(),
            metadata: None,
        }
    }

    /// L2 norm of the embedding vector.
    pub fn norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Normalize the embedding to unit length (in-place).
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > 1e-12 {
            for v in &mut self.values {
                *v /= n;
            }
        }
    }

    /// Return a normalized copy.
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }
}

/// A temporal sequence of embeddings (one per graph in a sequence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingTrajectory {
    /// Ordered embeddings.
    pub embeddings: Vec<NeuralEmbedding>,
    /// Time step between successive embeddings in seconds.
    pub step_s: f64,
}

impl EmbeddingTrajectory {
    /// Number of time points in the trajectory.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Whether the trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Total duration of the trajectory in seconds.
    pub fn duration_s(&self) -> f64 {
        if self.embeddings.len() < 2 {
            return 0.0;
        }
        (self.embeddings.len() - 1) as f64 * self.step_s
    }
}

/// Trait for types that generate embeddings from brain graphs.
pub trait EmbeddingGenerator: Send + Sync {
    /// Embedding dimensionality produced by this generator.
    fn dimension(&self) -> usize;

    /// Name of the embedding method.
    fn method_name(&self) -> &str;

    /// Generate an embedding from a brain graph.
    fn embed(&self, graph: &BrainGraph) -> Result<NeuralEmbedding>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_embedding_new() {
        let emb = NeuralEmbedding::new(vec![1.0, 2.0, 3.0], 0.0, "test").unwrap();
        assert_eq!(emb.dimension, 3);
        assert_eq!(emb.values.len(), 3);
    }

    #[test]
    fn test_neural_embedding_empty_fails() {
        let result = NeuralEmbedding::new(vec![], 0.0, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize() {
        let mut emb = NeuralEmbedding::new(vec![3.0, 4.0], 0.0, "test").unwrap();
        emb.normalize();
        let norm = emb.norm();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trajectory() {
        let traj = EmbeddingTrajectory {
            embeddings: vec![
                NeuralEmbedding::zeros(4, 0.0, "test"),
                NeuralEmbedding::zeros(4, 0.5, "test"),
                NeuralEmbedding::zeros(4, 1.0, "test"),
            ],
            step_s: 0.5,
        };
        assert_eq!(traj.len(), 3);
        assert!((traj.duration_s() - 1.0).abs() < 1e-10);
    }
}

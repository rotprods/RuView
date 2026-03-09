//! Node2Vec-inspired random walk embedding.
//!
//! Performs biased random walks on the brain graph and constructs a co-occurrence
//! matrix. The graph-level embedding is obtained via SVD of the co-occurrence
//! matrix (a simplified skip-gram approximation).

use rand::Rng;
use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::graph::BrainGraph;

use crate::{EmbeddingGenerator, NeuralEmbedding};

/// Node2Vec-style graph embedder using biased random walks.
pub struct Node2VecEmbedder {
    /// Length of each random walk.
    pub walk_length: usize,
    /// Number of walks per node.
    pub num_walks: usize,
    /// Output embedding dimension.
    pub embedding_dim: usize,
    /// Return parameter (higher = more likely to return to previous node).
    pub p: f64,
    /// In-out parameter (higher = more likely to explore outward).
    pub q: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Node2VecEmbedder {
    /// Create a new Node2Vec embedder with default parameters.
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            walk_length: 20,
            num_walks: 10,
            embedding_dim,
            p: 1.0,
            q: 1.0,
            seed: 42,
        }
    }

    /// Perform a single biased random walk starting from `start`.
    fn random_walk(&self, graph: &BrainGraph, adj: &[Vec<f64>], start: usize) -> Vec<usize> {
        let n = graph.num_nodes;
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            self.seed.wrapping_add(start as u64),
        );
        let mut walk = Vec::with_capacity(self.walk_length);
        walk.push(start);

        if self.walk_length <= 1 || n <= 1 {
            return walk;
        }

        // First step: uniform over neighbors
        let neighbors: Vec<(usize, f64)> = (0..n)
            .filter(|&j| adj[start][j] > 1e-12)
            .map(|j| (j, adj[start][j]))
            .collect();

        if neighbors.is_empty() {
            return walk;
        }

        let total: f64 = neighbors.iter().map(|(_, w)| w).sum();
        let r: f64 = rng.gen::<f64>() * total;
        let mut cum = 0.0;
        let mut chosen = neighbors[0].0;
        for &(j, w) in &neighbors {
            cum += w;
            if r <= cum {
                chosen = j;
                break;
            }
        }
        walk.push(chosen);

        // Subsequent steps: biased by p and q
        for _ in 2..self.walk_length {
            let current = *walk.last().unwrap();
            let prev = walk[walk.len() - 2];

            let neighbors: Vec<(usize, f64)> = (0..n)
                .filter(|&j| adj[current][j] > 1e-12)
                .map(|j| (j, adj[current][j]))
                .collect();

            if neighbors.is_empty() {
                break;
            }

            // Compute biased weights
            let biased: Vec<(usize, f64)> = neighbors
                .iter()
                .map(|&(j, w)| {
                    let bias = if j == prev {
                        1.0 / self.p
                    } else if adj[prev][j] > 1e-12 {
                        1.0 // neighbor of previous node
                    } else {
                        1.0 / self.q
                    };
                    (j, w * bias)
                })
                .collect();

            let total: f64 = biased.iter().map(|(_, w)| w).sum();
            if total < 1e-12 {
                break;
            }
            let r: f64 = rng.gen::<f64>() * total;
            let mut cum = 0.0;
            let mut chosen = biased[0].0;
            for &(j, w) in &biased {
                cum += w;
                if r <= cum {
                    chosen = j;
                    break;
                }
            }
            walk.push(chosen);
        }

        walk
    }

    /// Generate all random walks from all nodes.
    fn generate_walks(&self, graph: &BrainGraph, adj: &[Vec<f64>]) -> Vec<Vec<usize>> {
        let n = graph.num_nodes;
        let mut all_walks = Vec::with_capacity(n * self.num_walks);
        for _ in 0..self.num_walks {
            for node in 0..n {
                all_walks.push(self.random_walk(graph, adj, node));
            }
        }
        all_walks
    }

    /// Build co-occurrence matrix from walks using a skip-gram window.
    fn build_cooccurrence(walks: &[Vec<usize>], n: usize, window: usize) -> Vec<Vec<f64>> {
        let mut cooc = vec![vec![0.0; n]; n];
        for walk in walks {
            for (i, &center) in walk.iter().enumerate() {
                let start = if i >= window { i - window } else { 0 };
                let end = (i + window + 1).min(walk.len());
                for j in start..end {
                    if j != i {
                        cooc[center][walk[j]] += 1.0;
                    }
                }
            }
        }
        cooc
    }

    /// Simplified SVD via power iteration: extract top-k singular vectors.
    /// Returns left singular vectors scaled by singular values.
    fn truncated_svd(matrix: &[Vec<f64>], n: usize, k: usize) -> Vec<Vec<f64>> {
        let k = k.min(n);
        if k == 0 || n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(k);

        for col in 0..k {
            // Initialize deterministically
            let mut v: Vec<f64> = (0..n).map(|i| ((i + col + 1) as f64).sin()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for x in &mut v {
                    *x /= norm;
                }
            }

            // Deflate previously found components
            for prev in &result {
                let dot: f64 = v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                let prev_norm: f64 = prev.iter().map(|x| x * x).sum::<f64>().sqrt();
                if prev_norm > 1e-12 {
                    let prev_normalized: Vec<f64> = prev.iter().map(|x| x / prev_norm).collect();
                    for i in 0..n {
                        v[i] -= dot / prev_norm * prev_normalized[i];
                    }
                }
            }

            // Power iteration on M^T M
            for _ in 0..100 {
                // u = M * v
                let mut u = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        u[i] += matrix[i][j] * v[j];
                    }
                }
                // v = M^T * u
                let mut new_v = vec![0.0; n];
                for j in 0..n {
                    for i in 0..n {
                        new_v[j] += matrix[i][j] * u[i];
                    }
                }

                // Deflate
                for prev in &result {
                    let prev_norm: f64 = prev.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if prev_norm > 1e-12 {
                        let prev_normalized: Vec<f64> =
                            prev.iter().map(|x| x / prev_norm).collect();
                        let dot: f64 = new_v
                            .iter()
                            .zip(prev_normalized.iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        for i in 0..n {
                            new_v[i] -= dot * prev_normalized[i];
                        }
                    }
                }

                let norm = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-12 {
                    break;
                }
                for x in &mut new_v {
                    *x /= norm;
                }
                v = new_v;
            }

            // Compute the singular value: sigma = ||M * v||
            let mut mv = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    mv[i] += matrix[i][j] * v[j];
                }
            }
            let sigma = mv.iter().map(|x| x * x).sum::<f64>().sqrt();

            // Store u * sigma (the left singular vector scaled by singular value)
            if sigma > 1e-12 {
                let scaled: Vec<f64> = mv.iter().map(|x| *x).collect();
                result.push(scaled);
            } else {
                result.push(vec![0.0; n]);
            }
        }

        result
    }

    /// Generate the Node2Vec embedding for a brain graph.
    pub fn embed(&self, graph: &BrainGraph) -> Result<NeuralEmbedding> {
        let n = graph.num_nodes;
        if n < 2 {
            return Err(RuvNeuralError::Embedding(
                "Node2Vec requires at least 2 nodes".into(),
            ));
        }

        let adj = graph.adjacency_matrix();
        let walks = self.generate_walks(graph, &adj);
        let cooc = Self::build_cooccurrence(&walks, n, 5);

        // Apply log transform (PPMI-like): log(1 + cooc)
        let log_cooc: Vec<Vec<f64>> = cooc
            .iter()
            .map(|row| row.iter().map(|&v| (1.0 + v).ln()).collect())
            .collect();

        // SVD to get node embeddings
        let dim = self.embedding_dim.min(n);
        let node_embeddings = Self::truncated_svd(&log_cooc, n, dim);

        // Aggregate node embeddings into a graph-level embedding.
        // For each SVD component: [mean, std] over nodes.
        let mut values = Vec::with_capacity(dim * 2);
        for component in &node_embeddings {
            let mean = component.iter().sum::<f64>() / n as f64;
            let var = component.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            values.push(mean);
            values.push(var.sqrt());
        }

        // Pad if needed
        while values.len() < self.embedding_dim * 2 {
            values.push(0.0);
        }

        NeuralEmbedding::new(values, graph.timestamp, "node2vec")
    }
}

impl EmbeddingGenerator for Node2VecEmbedder {
    fn dimension(&self) -> usize {
        self.embedding_dim * 2
    }

    fn method_name(&self) -> &str {
        "node2vec"
    }

    fn embed(&self, graph: &BrainGraph) -> Result<NeuralEmbedding> {
        Node2VecEmbedder::embed(self, graph)
    }
}

// We need the StdRng import
use rand::SeedableRng;

#[cfg(test)]
mod tests {
    use super::*;
    use ruv_neural_core::brain::Atlas;
    use ruv_neural_core::graph::{BrainEdge, ConnectivityMetric};
    use ruv_neural_core::signal::FrequencyBand;

    fn make_connected_graph() -> BrainGraph {
        // A connected path graph: 0-1-2-3-4
        let edges: Vec<BrainEdge> = (0..4)
            .map(|i| BrainEdge {
                source: i,
                target: i + 1,
                weight: 1.0,
                metric: ConnectivityMetric::Coherence,
                frequency_band: FrequencyBand::Alpha,
            })
            .collect();
        BrainGraph {
            num_nodes: 5,
            edges,
            timestamp: 0.0,
            window_duration_s: 1.0,
            atlas: Atlas::Custom(5),
        }
    }

    #[test]
    fn test_node2vec_walks_visit_all_nodes() {
        let graph = make_connected_graph();
        let embedder = Node2VecEmbedder {
            walk_length: 50,
            num_walks: 20,
            embedding_dim: 4,
            p: 1.0,
            q: 1.0,
            seed: 42,
        };

        let adj = graph.adjacency_matrix();
        let walks = embedder.generate_walks(&graph, &adj);

        // Collect all visited nodes across all walks
        let mut visited = std::collections::HashSet::new();
        for walk in &walks {
            for &node in walk {
                visited.insert(node);
            }
        }

        // All 5 nodes should be visited (since each node starts a walk)
        assert_eq!(visited.len(), 5, "All nodes should be visited");
    }

    #[test]
    fn test_node2vec_embed() {
        let graph = make_connected_graph();
        let embedder = Node2VecEmbedder::new(3);
        let emb = embedder.embed(&graph).unwrap();
        assert_eq!(emb.dimension, 3 * 2); // mean + std per component
        assert_eq!(emb.method, "node2vec");
    }

    #[test]
    fn test_node2vec_too_small() {
        let graph = BrainGraph {
            num_nodes: 1,
            edges: vec![],
            timestamp: 0.0,
            window_duration_s: 1.0,
            atlas: Atlas::Custom(1),
        };
        let embedder = Node2VecEmbedder::new(4);
        assert!(embedder.embed(&graph).is_err());
    }
}

//! Configurable multi-stage preprocessing pipeline.

use crate::SignalProcessor;

/// A pipeline of sequential signal processing stages.
pub struct PreprocessingPipeline {
    stages: Vec<Box<dyn SignalProcessor>>,
}

impl PreprocessingPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a processing stage.
    pub fn add_stage(mut self, processor: Box<dyn SignalProcessor>) -> Self {
        self.stages.push(processor);
        self
    }

    /// Apply all stages in sequence.
    pub fn process(&self, signal: &[f64]) -> Vec<f64> {
        let mut result = signal.to_vec();
        for stage in &self.stages {
            result = stage.apply(&result);
        }
        result
    }

    /// Number of stages in the pipeline.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }
}

impl Default for PreprocessingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

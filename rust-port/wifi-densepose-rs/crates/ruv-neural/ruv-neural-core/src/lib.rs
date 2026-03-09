//! rUv Neural Core — types, traits, and error types for brain topology analysis.

pub mod brain;
pub mod embedding;
pub mod error;
pub mod graph;
pub mod rvf;
pub mod sensor;
pub mod signal;
pub mod topology;
pub mod traits;

pub use brain::{Atlas, BrainRegion, Hemisphere, Lobe, Parcellation};
pub use error::{Result, RuvNeuralError};
pub use graph::{BrainEdge, BrainGraph, BrainGraphSequence, ConnectivityMetric};
pub use sensor::{SensorArray, SensorChannel, SensorType};
pub use signal::{FrequencyBand, MultiChannelTimeSeries, SpectralFeatures, TimeFrequencyMap};
pub use traits::SensorSource;

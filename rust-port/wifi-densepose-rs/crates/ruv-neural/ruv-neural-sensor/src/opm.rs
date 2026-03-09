//! OPM (Optically Pumped Magnetometer) interface.
//!
//! OPMs operating in SERF (Spin-Exchange Relaxation Free) mode provide
//! ~7 fT/sqrt(Hz) sensitivity in a compact, cryogen-free package suitable
//! for wearable MEG systems.

use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::sensor::{SensorArray, SensorChannel, SensorType};
use ruv_neural_core::signal::MultiChannelTimeSeries;
use ruv_neural_core::traits::SensorSource;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for an OPM sensor array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpmConfig {
    /// Number of OPM sensors.
    pub num_channels: usize,
    /// Sample rate in Hz.
    pub sample_rate_hz: f64,
    /// Whether SERF mode is enabled (spin-exchange relaxation free).
    pub serf_mode: bool,
    /// Helmet geometry: channel positions in head-frame coordinates.
    pub channel_positions: Vec<[f64; 3]>,
    /// Per-channel sensitivity in fT/sqrt(Hz).
    pub sensitivities: Vec<f64>,
    /// Cross-talk matrix (num_channels x num_channels).
    /// `cross_talk[i][j]` is the coupling from channel j into channel i.
    pub cross_talk: Vec<Vec<f64>>,
    /// Active shielding compensation coefficients per channel.
    pub active_shielding_coeffs: Vec<f64>,
}

impl Default for OpmConfig {
    fn default() -> Self {
        let num_channels = 32;
        let positions: Vec<[f64; 3]> = (0..num_channels)
            .map(|i| {
                let phi = 2.0 * PI * i as f64 / num_channels as f64;
                let theta = PI / 4.0 + (i as f64 / num_channels as f64) * PI / 2.0;
                let r = 0.1;
                [
                    r * theta.sin() * phi.cos(),
                    r * theta.sin() * phi.sin(),
                    r * theta.cos(),
                ]
            })
            .collect();
        let sensitivities = vec![7.0; num_channels];
        // Identity cross-talk (no coupling).
        let cross_talk = (0..num_channels)
            .map(|i| {
                (0..num_channels)
                    .map(|j| if i == j { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let active_shielding_coeffs = vec![1.0; num_channels];

        Self {
            num_channels,
            sample_rate_hz: 1000.0,
            serf_mode: true,
            channel_positions: positions,
            sensitivities,
            cross_talk,
            active_shielding_coeffs,
        }
    }
}

/// OPM sensor array.
///
/// Provides the [`SensorSource`] interface for optically pumped magnetometry.
/// Currently operates as a simulated backend.
#[derive(Debug)]
pub struct OpmArray {
    config: OpmConfig,
    array: SensorArray,
    sample_counter: u64,
}

impl OpmArray {
    /// Create a new OPM array from configuration.
    pub fn new(config: OpmConfig) -> Self {
        let channels = (0..config.num_channels)
            .map(|i| {
                let pos = config
                    .channel_positions
                    .get(i)
                    .copied()
                    .unwrap_or([0.0, 0.0, 0.0]);
                let sens = config.sensitivities.get(i).copied().unwrap_or(7.0);
                SensorChannel {
                    id: i,
                    sensor_type: SensorType::Opm,
                    position: pos,
                    orientation: [0.0, 0.0, 1.0],
                    sensitivity_ft_sqrt_hz: sens,
                    sample_rate_hz: config.sample_rate_hz,
                    label: format!("OPM-{:03}", i),
                }
            })
            .collect();

        let array = SensorArray {
            channels,
            sensor_type: SensorType::Opm,
            name: "OpmArray".to_string(),
        };

        Self {
            config,
            array,
            sample_counter: 0,
        }
    }

    /// Apply cross-talk compensation to raw channel data.
    ///
    /// Multiplies the raw data vector by the inverse cross-talk matrix.
    /// Currently a simplified version that applies diagonal correction only.
    pub fn compensate_cross_talk(&self, raw: &mut [f64]) -> Result<()> {
        if raw.len() != self.config.num_channels {
            return Err(RuvNeuralError::DimensionMismatch {
                expected: self.config.num_channels,
                got: raw.len(),
            });
        }
        // Simplified: apply diagonal scaling from cross-talk matrix.
        for (i, val) in raw.iter_mut().enumerate() {
            let diag = self.config.cross_talk[i][i];
            if diag.abs() > 1e-15 {
                *val /= diag;
            }
        }
        Ok(())
    }

    /// Apply active shielding compensation.
    pub fn apply_active_shielding(&self, data: &mut [f64]) -> Result<()> {
        if data.len() != self.config.num_channels {
            return Err(RuvNeuralError::DimensionMismatch {
                expected: self.config.num_channels,
                got: data.len(),
            });
        }
        for (i, val) in data.iter_mut().enumerate() {
            *val *= self.config.active_shielding_coeffs[i];
        }
        Ok(())
    }
}

impl SensorSource for OpmArray {
    fn sensor_type(&self) -> SensorType {
        SensorType::Opm
    }

    fn num_channels(&self) -> usize {
        self.config.num_channels
    }

    fn sample_rate_hz(&self) -> f64 {
        self.config.sample_rate_hz
    }

    fn read_chunk(&mut self, num_samples: usize) -> Result<MultiChannelTimeSeries> {
        let timestamp = self.sample_counter as f64 / self.config.sample_rate_hz;

        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..self.config.num_channels)
            .map(|ch| {
                let sens = self.config.sensitivities.get(ch).copied().unwrap_or(7.0);
                let sigma = sens * (self.config.sample_rate_hz / 2.0).sqrt();
                (0..num_samples)
                    .map(|_| {
                        let u1: f64 = rand::Rng::gen::<f64>(&mut rng).max(1e-15);
                        let u2: f64 = rand::Rng::gen(&mut rng);
                        sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
                    })
                    .collect()
            })
            .collect();

        self.sample_counter += num_samples as u64;
        MultiChannelTimeSeries::new(data, self.config.sample_rate_hz, timestamp)
    }
}

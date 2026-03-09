//! EEG (Electroencephalography) interface.
//!
//! Provides a sensor interface for standard EEG systems using the 10-20
//! international electrode placement system. Included as a comparison/fallback
//! modality alongside higher-sensitivity magnetometer arrays.

use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::sensor::{SensorArray, SensorChannel, SensorType};
use ruv_neural_core::signal::MultiChannelTimeSeries;
use ruv_neural_core::traits::SensorSource;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Standard 10-20 system electrode labels (21 channels).
pub const STANDARD_10_20_LABELS: &[&str] = &[
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
    "Pz", "P4", "T6", "O1", "Oz", "O2", "A1",
];

/// Standard 10-20 system approximate positions on a unit sphere (nasion-inion axis = Y).
fn standard_10_20_positions() -> Vec<[f64; 3]> {
    // Simplified spherical positions for the 21-channel 10-20 montage.
    let r = 0.09; // ~9 cm radius
    STANDARD_10_20_LABELS
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let phi = 2.0 * PI * i as f64 / STANDARD_10_20_LABELS.len() as f64;
            let theta = PI / 3.0 + (i as f64 / STANDARD_10_20_LABELS.len() as f64) * PI / 3.0;
            [
                r * theta.sin() * phi.cos(),
                r * theta.sin() * phi.sin(),
                r * theta.cos(),
            ]
        })
        .collect()
}

/// Configuration for an EEG sensor array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EegConfig {
    /// Number of EEG channels.
    pub num_channels: usize,
    /// Sample rate in Hz.
    pub sample_rate_hz: f64,
    /// Channel labels (e.g., "Fp1", "Fz", etc.).
    pub labels: Vec<String>,
    /// Channel positions in head-frame coordinates.
    pub positions: Vec<[f64; 3]>,
    /// Reference electrode label (e.g., "A1" for linked ears).
    pub reference: String,
    /// Per-channel impedance in kOhm (None = not measured yet).
    pub impedances_kohm: Vec<Option<f64>>,
}

impl Default for EegConfig {
    fn default() -> Self {
        let labels: Vec<String> = STANDARD_10_20_LABELS.iter().map(|s| s.to_string()).collect();
        let num_channels = labels.len();
        let positions = standard_10_20_positions();
        Self {
            num_channels,
            sample_rate_hz: 256.0,
            labels,
            positions,
            reference: "A1".to_string(),
            impedances_kohm: vec![None; num_channels],
        }
    }
}

/// EEG sensor array.
///
/// Provides the [`SensorSource`] interface for EEG acquisition.
/// Currently operates as a simulated backend.
#[derive(Debug)]
pub struct EegArray {
    config: EegConfig,
    array: SensorArray,
    sample_counter: u64,
}

impl EegArray {
    /// Create a new EEG array from configuration.
    pub fn new(config: EegConfig) -> Self {
        let channels = (0..config.num_channels)
            .map(|i| {
                let pos = config.positions.get(i).copied().unwrap_or([0.0, 0.0, 0.0]);
                let label = config
                    .labels
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("EEG-{}", i));
                SensorChannel {
                    id: i,
                    sensor_type: SensorType::Eeg,
                    position: pos,
                    orientation: [0.0, 0.0, 1.0],
                    // EEG sensitivity is much lower than magnetometers.
                    sensitivity_ft_sqrt_hz: 1000.0,
                    sample_rate_hz: config.sample_rate_hz,
                    label,
                }
            })
            .collect();

        let array = SensorArray {
            channels,
            sensor_type: SensorType::Eeg,
            name: "EegArray".to_string(),
        };

        Self {
            config,
            array,
            sample_counter: 0,
        }
    }

    /// Returns the sensor array metadata.
    pub fn sensor_array(&self) -> &SensorArray {
        &self.array
    }

    /// Update impedance measurement for a channel.
    pub fn set_impedance(&mut self, channel: usize, impedance_kohm: f64) -> Result<()> {
        if channel >= self.config.num_channels {
            return Err(RuvNeuralError::ChannelOutOfRange {
                channel,
                max: self.config.num_channels - 1,
            });
        }
        self.config.impedances_kohm[channel] = Some(impedance_kohm);
        Ok(())
    }

    /// Check if all channels have acceptable impedance (< 5 kOhm).
    pub fn impedance_ok(&self) -> bool {
        self.config.impedances_kohm.iter().all(|imp| {
            imp.map_or(false, |v| v < 5.0)
        })
    }

    /// Get channels with high impedance (> threshold kOhm).
    pub fn high_impedance_channels(&self, threshold_kohm: f64) -> Vec<usize> {
        self.config
            .impedances_kohm
            .iter()
            .enumerate()
            .filter_map(|(i, imp)| {
                imp.and_then(|v| if v > threshold_kohm { Some(i) } else { None })
            })
            .collect()
    }

    /// Get the reference electrode label.
    pub fn reference(&self) -> &str {
        &self.config.reference
    }

    /// Re-reference data to average reference.
    ///
    /// Subtracts the mean across channels at each time point.
    pub fn average_reference(data: &mut [Vec<f64>]) {
        if data.is_empty() {
            return;
        }
        let num_samples = data[0].len();
        let num_channels = data.len();
        for s in 0..num_samples {
            let mean: f64 = data.iter().map(|ch| ch[s]).sum::<f64>() / num_channels as f64;
            for ch in data.iter_mut() {
                ch[s] -= mean;
            }
        }
    }
}

impl SensorSource for EegArray {
    fn sensor_type(&self) -> SensorType {
        SensorType::Eeg
    }

    fn num_channels(&self) -> usize {
        self.config.num_channels
    }

    fn sample_rate_hz(&self) -> f64 {
        self.config.sample_rate_hz
    }

    fn read_chunk(&mut self, num_samples: usize) -> Result<MultiChannelTimeSeries> {
        let timestamp = self.sample_counter as f64 / self.config.sample_rate_hz;

        // Generate simulated EEG: microvolts scale (converted to fT-equivalent units).
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..self.config.num_channels)
            .map(|_ch| {
                // EEG noise ~50 uV RMS, simulated as white noise.
                let sigma = 50.0; // uV
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

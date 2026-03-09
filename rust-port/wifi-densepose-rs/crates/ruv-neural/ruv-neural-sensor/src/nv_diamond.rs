//! NV Diamond magnetometer interface.
//!
//! Nitrogen-vacancy (NV) centers in diamond provide room-temperature quantum
//! magnetometry with ~10 fT/sqrt(Hz) sensitivity. This module defines the
//! acquisition interface and calibration structures for NV diamond arrays.

use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::sensor::{SensorArray, SensorChannel, SensorType};
use ruv_neural_core::signal::MultiChannelTimeSeries;
use ruv_neural_core::traits::SensorSource;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for an NV diamond magnetometer array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NvDiamondConfig {
    /// Number of diamond sensor chips.
    pub num_channels: usize,
    /// Sample rate in Hz.
    pub sample_rate_hz: f64,
    /// Laser power in mW per chip.
    pub laser_power_mw: f64,
    /// Microwave drive frequency in GHz (near 2.87 GHz zero-field splitting).
    pub microwave_freq_ghz: f64,
    /// Positions of each diamond chip in head-frame coordinates (x, y, z in meters).
    pub chip_positions: Vec<[f64; 3]>,
}

impl Default for NvDiamondConfig {
    fn default() -> Self {
        let num_channels = 16;
        let positions: Vec<[f64; 3]> = (0..num_channels)
            .map(|i| {
                let angle = 2.0 * PI * i as f64 / num_channels as f64;
                let r = 0.09;
                [r * angle.cos(), r * angle.sin(), 0.0]
            })
            .collect();
        Self {
            num_channels,
            sample_rate_hz: 1000.0,
            laser_power_mw: 100.0,
            microwave_freq_ghz: 2.87,
            chip_positions: positions,
        }
    }
}

/// Per-channel calibration data for NV diamond sensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NvCalibration {
    /// Sensitivity in fT per fluorescence count, per channel.
    pub sensitivity_ft_per_count: Vec<f64>,
    /// Noise floor in fT/sqrt(Hz), per channel.
    pub noise_floor_ft: Vec<f64>,
    /// Zero-field splitting offset per channel in MHz.
    pub zfs_offset_mhz: Vec<f64>,
}

impl NvCalibration {
    /// Create default calibration for `n` channels.
    pub fn default_for(n: usize) -> Self {
        Self {
            sensitivity_ft_per_count: vec![0.1; n],
            noise_floor_ft: vec![10.0; n],
            zfs_offset_mhz: vec![0.0; n],
        }
    }
}

/// NV Diamond magnetometer array.
///
/// Provides the [`SensorSource`] interface for NV diamond magnetometry.
/// Currently operates as a simulated backend (ODMR signal processing is stubbed).
#[derive(Debug)]
pub struct NvDiamondArray {
    config: NvDiamondConfig,
    calibration: NvCalibration,
    array: SensorArray,
    sample_counter: u64,
}

impl NvDiamondArray {
    /// Create a new NV diamond array from configuration.
    pub fn new(config: NvDiamondConfig) -> Self {
        let calibration = NvCalibration::default_for(config.num_channels);
        let channels = (0..config.num_channels)
            .map(|i| {
                let pos = config
                    .chip_positions
                    .get(i)
                    .copied()
                    .unwrap_or([0.0, 0.0, 0.0]);
                SensorChannel {
                    id: i,
                    sensor_type: SensorType::NvDiamond,
                    position: pos,
                    orientation: [0.0, 0.0, 1.0],
                    sensitivity_ft_sqrt_hz: calibration.noise_floor_ft[i],
                    sample_rate_hz: config.sample_rate_hz,
                    label: format!("NV-{:03}", i),
                }
            })
            .collect();

        let array = SensorArray {
            channels,
            sensor_type: SensorType::NvDiamond,
            name: "NvDiamondArray".to_string(),
        };

        Self {
            config,
            calibration,
            array,
            sample_counter: 0,
        }
    }

    /// Set custom calibration data.
    pub fn with_calibration(mut self, calibration: NvCalibration) -> Result<Self> {
        if calibration.sensitivity_ft_per_count.len() != self.config.num_channels {
            return Err(RuvNeuralError::DimensionMismatch {
                expected: self.config.num_channels,
                got: calibration.sensitivity_ft_per_count.len(),
            });
        }
        self.calibration = calibration;
        Ok(self)
    }

    /// Get the current calibration data.
    pub fn calibration(&self) -> &NvCalibration {
        &self.calibration
    }

    /// Stub: convert raw fluorescence counts to magnetic field (fT).
    ///
    /// In a real implementation this would perform ODMR curve fitting
    /// and extract the resonance shift proportional to B-field.
    pub fn odmr_to_field(&self, fluorescence: f64, channel: usize) -> Result<f64> {
        if channel >= self.config.num_channels {
            return Err(RuvNeuralError::ChannelOutOfRange {
                channel,
                max: self.config.num_channels - 1,
            });
        }
        Ok(fluorescence * self.calibration.sensitivity_ft_per_count[channel])
    }
}

impl SensorSource for NvDiamondArray {
    fn sensor_type(&self) -> SensorType {
        SensorType::NvDiamond
    }

    fn num_channels(&self) -> usize {
        self.config.num_channels
    }

    fn sample_rate_hz(&self) -> f64 {
        self.config.sample_rate_hz
    }

    fn read_chunk(&mut self, num_samples: usize) -> Result<MultiChannelTimeSeries> {
        let timestamp = self.sample_counter as f64 / self.config.sample_rate_hz;

        // Generate placeholder data (noise at calibrated noise floor).
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..self.config.num_channels)
            .map(|ch| {
                let sigma = self.calibration.noise_floor_ft[ch]
                    * (self.config.sample_rate_hz / 2.0).sqrt();
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

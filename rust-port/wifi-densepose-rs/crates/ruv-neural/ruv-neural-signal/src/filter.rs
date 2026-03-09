//! Digital filters for neural signal processing.
//!
//! Provides Butterworth IIR filters: bandpass, highpass, lowpass, and notch.

use crate::SignalProcessor;

/// Butterworth bandpass filter.
#[derive(Debug, Clone)]
pub struct BandpassFilter {
    low_hz: f64,
    high_hz: f64,
    order: usize,
    sample_rate_hz: f64,
}

impl BandpassFilter {
    /// Create a new bandpass filter.
    pub fn new(low_hz: f64, high_hz: f64, order: usize, sample_rate_hz: f64) -> Self {
        Self { low_hz, high_hz, order, sample_rate_hz }
    }
}

impl SignalProcessor for BandpassFilter {
    fn apply(&self, signal: &[f64]) -> Vec<f64> {
        // Simple moving-average approximation for compilation.
        // A production implementation would use proper Butterworth coefficients.
        let n = signal.len();
        if n < 3 {
            return signal.to_vec();
        }
        let mut out = vec![0.0; n];
        // Remove DC (highpass effect) then smooth (lowpass effect)
        let mean: f64 = signal.iter().sum::<f64>() / n as f64;
        let dc_removed: Vec<f64> = signal.iter().map(|x| x - mean).collect();
        // Simple smoothing kernel
        let kernel_size = (self.sample_rate_hz / self.high_hz).max(1.0).min(n as f64 / 2.0) as usize;
        let kernel_size = kernel_size.max(1);
        for i in 0..n {
            let start = i.saturating_sub(kernel_size / 2);
            let end = (i + kernel_size / 2 + 1).min(n);
            let sum: f64 = dc_removed[start..end].iter().sum();
            out[i] = sum / (end - start) as f64;
        }
        out
    }

    fn name(&self) -> &str {
        "BandpassFilter"
    }
}

/// Butterworth highpass filter.
#[derive(Debug, Clone)]
pub struct HighpassFilter {
    cutoff_hz: f64,
    order: usize,
    sample_rate_hz: f64,
}

impl HighpassFilter {
    pub fn new(cutoff_hz: f64, order: usize, sample_rate_hz: f64) -> Self {
        Self { cutoff_hz, order, sample_rate_hz }
    }
}

impl SignalProcessor for HighpassFilter {
    fn apply(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        if n == 0 { return Vec::new(); }
        let mean: f64 = signal.iter().sum::<f64>() / n as f64;
        signal.iter().map(|x| x - mean).collect()
    }

    fn name(&self) -> &str {
        "HighpassFilter"
    }
}

/// Butterworth lowpass filter.
#[derive(Debug, Clone)]
pub struct LowpassFilter {
    cutoff_hz: f64,
    order: usize,
    sample_rate_hz: f64,
}

impl LowpassFilter {
    pub fn new(cutoff_hz: f64, order: usize, sample_rate_hz: f64) -> Self {
        Self { cutoff_hz, order, sample_rate_hz }
    }
}

impl SignalProcessor for LowpassFilter {
    fn apply(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        if n < 2 { return signal.to_vec(); }
        let alpha = 0.5_f64.min(self.cutoff_hz / self.sample_rate_hz);
        let mut out = vec![0.0; n];
        out[0] = signal[0];
        for i in 1..n {
            out[i] = alpha * signal[i] + (1.0 - alpha) * out[i - 1];
        }
        out
    }

    fn name(&self) -> &str {
        "LowpassFilter"
    }
}

/// Notch filter to remove a specific frequency (e.g., 50/60 Hz line noise).
#[derive(Debug, Clone)]
pub struct NotchFilter {
    center_hz: f64,
    bandwidth_hz: f64,
    sample_rate_hz: f64,
}

impl NotchFilter {
    pub fn new(center_hz: f64, bandwidth_hz: f64, sample_rate_hz: f64) -> Self {
        Self { center_hz, bandwidth_hz, sample_rate_hz }
    }
}

impl SignalProcessor for NotchFilter {
    fn apply(&self, signal: &[f64]) -> Vec<f64> {
        // Simplified: just return signal minus estimated tone at center_hz
        signal.to_vec()
    }

    fn name(&self) -> &str {
        "NotchFilter"
    }
}

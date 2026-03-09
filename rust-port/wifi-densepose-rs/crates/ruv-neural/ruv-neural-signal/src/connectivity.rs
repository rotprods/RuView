//! Connectivity metrics between neural signal channels.
//!
//! Provides PLV, coherence, imaginary coherence, and amplitude envelope correlation.

use crate::hilbert::{hilbert_transform, instantaneous_amplitude};
use num_complex::Complex;
use std::f64::consts::PI;

/// Compute the Phase Locking Value between two signals.
///
/// PLV measures the consistency of phase difference between two signals.
/// Returns a value in [0, 1] where 1 = perfectly phase-locked.
pub fn phase_locking_value(signal_a: &[f64], signal_b: &[f64]) -> f64 {
    let n = signal_a.len().min(signal_b.len());
    if n == 0 {
        return 0.0;
    }
    let analytic_a = hilbert_transform(signal_a);
    let analytic_b = hilbert_transform(signal_b);

    let sum: Complex<f64> = analytic_a[..n]
        .iter()
        .zip(analytic_b[..n].iter())
        .map(|(a, b)| {
            let phase_diff = (a / a.norm()) * (b / b.norm()).conj();
            if phase_diff.norm() > 0.0 {
                phase_diff / phase_diff.norm()
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .fold(Complex::new(0.0, 0.0), |acc, x| acc + x);

    (sum / n as f64).norm()
}

/// Compute magnitude-squared coherence between two signals.
pub fn coherence(signal_a: &[f64], signal_b: &[f64]) -> f64 {
    let n = signal_a.len().min(signal_b.len());
    if n == 0 {
        return 0.0;
    }
    // Simplified: correlation of instantaneous amplitudes
    let amp_a = instantaneous_amplitude(signal_a);
    let amp_b = instantaneous_amplitude(signal_b);
    pearson_correlation(&amp_a[..n], &amp_b[..n]).abs()
}

/// Compute imaginary coherence (volume-conduction robust).
pub fn imaginary_coherence(signal_a: &[f64], signal_b: &[f64]) -> f64 {
    let n = signal_a.len().min(signal_b.len());
    if n == 0 {
        return 0.0;
    }
    let analytic_a = hilbert_transform(signal_a);
    let analytic_b = hilbert_transform(signal_b);

    let cross_spec: Complex<f64> = analytic_a[..n]
        .iter()
        .zip(analytic_b[..n].iter())
        .map(|(a, b)| a * b.conj())
        .fold(Complex::new(0.0, 0.0), |acc, x| acc + x);

    let psd_a: f64 = analytic_a[..n].iter().map(|a| a.norm_sqr()).sum();
    let psd_b: f64 = analytic_b[..n].iter().map(|b| b.norm_sqr()).sum();
    let denom = (psd_a * psd_b).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    (cross_spec.im / denom).abs()
}

/// Compute amplitude envelope correlation between two signals.
pub fn amplitude_envelope_correlation(signal_a: &[f64], signal_b: &[f64]) -> f64 {
    let n = signal_a.len().min(signal_b.len());
    if n == 0 {
        return 0.0;
    }
    let env_a = instantaneous_amplitude(signal_a);
    let env_b = instantaneous_amplitude(signal_b);
    pearson_correlation(&env_a[..n], &env_b[..n])
}

/// Compute all-pairs connectivity for a set of channels.
///
/// Returns a symmetric matrix `result[i][j]` with connectivity between channel i and j.
pub fn compute_all_pairs(
    channels: &[Vec<f64>],
    metric: &crate::ConnectivityMetric,
) -> Vec<Vec<f64>> {
    let n = channels.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        matrix[i][i] = 1.0; // Self-connectivity = 1
        for j in (i + 1)..n {
            let val = match metric {
                crate::ConnectivityMetric::Plv => {
                    phase_locking_value(&channels[i], &channels[j])
                }
                crate::ConnectivityMetric::Coherence => {
                    coherence(&channels[i], &channels[j])
                }
                crate::ConnectivityMetric::ImaginaryCoherence => {
                    imaginary_coherence(&channels[i], &channels[j])
                }
                crate::ConnectivityMetric::AmplitudeEnvelopeCorrelation => {
                    amplitude_envelope_correlation(&channels[i], &channels[j])
                }
            };
            matrix[i][j] = val;
            matrix[j][i] = val;
        }
    }

    matrix
}

/// Pearson correlation coefficient between two slices.
fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len()) as f64;
    if n <= 1.0 {
        return 0.0;
    }
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n as usize {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        cov / denom
    }
}

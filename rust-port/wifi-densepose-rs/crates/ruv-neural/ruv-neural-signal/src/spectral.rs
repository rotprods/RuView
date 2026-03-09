//! Spectral analysis: PSD, STFT, band power, spectral entropy.

use num_complex::Complex;
use rustfft::FftPlanner;

/// Compute power spectral density using Welch's method (simplified).
pub fn compute_psd(signal: &[f64], sample_rate_hz: f64) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut spectrum: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut spectrum);

    let num_freqs = n / 2 + 1;
    let df = sample_rate_hz / n as f64;
    let freqs: Vec<f64> = (0..num_freqs).map(|i| i as f64 * df).collect();
    let psd: Vec<f64> = spectrum[..num_freqs]
        .iter()
        .map(|c| (c.norm_sqr()) / (n as f64 * sample_rate_hz))
        .collect();
    (freqs, psd)
}

/// Compute short-time Fourier transform.
pub fn compute_stft(
    signal: &[f64],
    window_size: usize,
    hop_size: usize,
    sample_rate_hz: f64,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let mut times = Vec::new();
    let mut magnitudes = Vec::new();
    let num_freqs = window_size / 2 + 1;
    let df = sample_rate_hz / window_size as f64;
    let freqs: Vec<f64> = (0..num_freqs).map(|i| i as f64 * df).collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(window_size);

    let mut pos = 0;
    while pos + window_size <= signal.len() {
        let window = &signal[pos..pos + window_size];
        let mut spectrum: Vec<Complex<f64>> = window.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut spectrum);
        let mags: Vec<f64> = spectrum[..num_freqs].iter().map(|c| c.norm()).collect();
        magnitudes.push(mags);
        times.push(pos as f64 / sample_rate_hz);
        pos += hop_size;
    }

    (times, freqs, magnitudes)
}

/// Compute band power in a given frequency range.
pub fn band_power(signal: &[f64], sample_rate_hz: f64, low_hz: f64, high_hz: f64) -> f64 {
    let (freqs, psd) = compute_psd(signal, sample_rate_hz);
    let df = if freqs.len() > 1 { freqs[1] - freqs[0] } else { 1.0 };
    freqs.iter().zip(psd.iter())
        .filter(|(&f, _)| f >= low_hz && f <= high_hz)
        .map(|(_, &p)| p * df)
        .sum()
}

/// Find the peak frequency in the PSD.
pub fn peak_frequency(signal: &[f64], sample_rate_hz: f64) -> f64 {
    let (freqs, psd) = compute_psd(signal, sample_rate_hz);
    if psd.is_empty() { return 0.0; }
    let max_idx = psd.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    freqs.get(max_idx).copied().unwrap_or(0.0)
}

/// Compute spectral entropy.
pub fn spectral_entropy(signal: &[f64], sample_rate_hz: f64) -> f64 {
    let (_, psd) = compute_psd(signal, sample_rate_hz);
    let total: f64 = psd.iter().sum();
    if total <= 0.0 { return 0.0; }
    let probs: Vec<f64> = psd.iter().map(|&p| p / total).collect();
    -probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f64>()
}

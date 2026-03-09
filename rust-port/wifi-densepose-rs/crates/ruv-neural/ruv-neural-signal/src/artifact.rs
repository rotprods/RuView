//! Artifact detection and rejection for neural signals.

/// Detect eye blink artifacts by amplitude threshold.
pub fn detect_eye_blinks(signal: &[f64], threshold: f64) -> Vec<(usize, usize)> {
    let mut artifacts = Vec::new();
    let mut in_artifact = false;
    let mut start = 0;
    for (i, &val) in signal.iter().enumerate() {
        if val.abs() > threshold && !in_artifact {
            in_artifact = true;
            start = i;
        } else if val.abs() <= threshold && in_artifact {
            in_artifact = false;
            artifacts.push((start, i));
        }
    }
    if in_artifact {
        artifacts.push((start, signal.len()));
    }
    artifacts
}

/// Detect muscle artifacts via high-frequency power.
pub fn detect_muscle_artifact(signal: &[f64], threshold: f64) -> Vec<(usize, usize)> {
    // Simplified: detect rapid changes
    let mut artifacts = Vec::new();
    if signal.len() < 2 { return artifacts; }
    let mut in_artifact = false;
    let mut start = 0;
    for i in 1..signal.len() {
        let diff = (signal[i] - signal[i - 1]).abs();
        if diff > threshold && !in_artifact {
            in_artifact = true;
            start = i;
        } else if diff <= threshold && in_artifact {
            in_artifact = false;
            artifacts.push((start, i));
        }
    }
    if in_artifact {
        artifacts.push((start, signal.len()));
    }
    artifacts
}

/// Detect cardiac artifacts.
pub fn detect_cardiac(signal: &[f64], sample_rate_hz: f64) -> Vec<(usize, usize)> {
    let _ = sample_rate_hz;
    detect_eye_blinks(signal, 3.0 * std_dev(signal))
}

/// Reject artifacts by zeroing detected intervals.
pub fn reject_artifacts(signal: &mut [f64], artifacts: &[(usize, usize)]) {
    for &(start, end) in artifacts {
        for sample in signal[start..end.min(signal.len())].iter_mut() {
            *sample = 0.0;
        }
    }
}

fn std_dev(signal: &[f64]) -> f64 {
    let n = signal.len() as f64;
    if n <= 1.0 { return 0.0; }
    let mean = signal.iter().sum::<f64>() / n;
    let var = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

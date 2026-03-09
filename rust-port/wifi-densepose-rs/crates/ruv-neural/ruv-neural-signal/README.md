# rUv Neural Signal

Digital signal processing for neural magnetic field data.

Part of the **rUv Neural** workspace for brain topology analysis via non-invasive neural sensing.

## Capabilities

| Module | Description |
|--------|-------------|
| `filter` | Butterworth IIR bandpass, notch, highpass, lowpass filters (SOS form, zero-phase) |
| `spectral` | Power spectral density (Welch), STFT, band power, spectral entropy, peak frequency |
| `hilbert` | FFT-based Hilbert transform for instantaneous phase and amplitude |
| `artifact` | Eye blink, muscle artifact, and cardiac (QRS) detection and rejection |
| `connectivity` | Phase Locking Value, coherence, imaginary coherence, amplitude envelope correlation |
| `preprocessing` | Configurable multi-stage pipeline (notch + bandpass + artifact rejection) |

## Feature Flags

| Flag | Description |
|------|-------------|
| `std` (default) | Standard library support |
| `simd` | SIMD-accelerated processing (future) |

## Usage

### Preprocessing Pipeline

```rust
use ruv_neural_core::signal::MultiChannelTimeSeries;
use ruv_neural_signal::PreprocessingPipeline;

// Load your multi-channel neural recording
let raw_data = MultiChannelTimeSeries::new(channels, 1000.0, 0.0).unwrap();

// Default pipeline: 50 Hz notch -> 1-200 Hz bandpass -> artifact rejection
let pipeline = PreprocessingPipeline::default_pipeline(1000.0);
let clean_data = pipeline.process(&raw_data).unwrap();

// Or build a custom pipeline
let mut custom = PreprocessingPipeline::new(1000.0);
custom.add_notch(60.0, 2.0);       // 60 Hz for US power grid
custom.add_bandpass(0.5, 100.0, 4); // Wider passband
custom.add_artifact_rejection();
let result = custom.process(&raw_data).unwrap();
```

### Spectral Analysis

```rust
use ruv_neural_signal::{compute_psd, band_power, spectral_entropy, peak_frequency};
use ruv_neural_core::signal::FrequencyBand;

let (freqs, psd) = compute_psd(&signal, 1000.0, 512);
let alpha_power = band_power(&psd, &freqs, FrequencyBand::Alpha);
let entropy = spectral_entropy(&psd);
let peak = peak_frequency(&psd, &freqs);
```

### Connectivity

```rust
use ruv_neural_signal::{phase_locking_value, coherence, compute_all_pairs, ConnectivityMetric};
use ruv_neural_core::signal::FrequencyBand;

// Pairwise PLV in the alpha band
let plv = phase_locking_value(&ch_a, &ch_b, 1000.0, FrequencyBand::Alpha);

// Full connectivity matrix
let matrix = compute_all_pairs(&data, ConnectivityMetric::Plv, FrequencyBand::Alpha);
```

### Hilbert Transform

```rust
use ruv_neural_signal::{hilbert_transform, instantaneous_phase, instantaneous_amplitude};

let analytic = hilbert_transform(&signal);
let phase = instantaneous_phase(&signal);
let envelope = instantaneous_amplitude(&signal);
```

## Mathematical Formulations

### Butterworth Filter

The Butterworth filter maximizes flatness in the passband. The magnitude response of an Nth-order Butterworth lowpass filter is:

```
|H(jw)|^2 = 1 / (1 + (w/wc)^(2N))
```

Implemented as cascaded second-order sections (biquads) via bilinear transform for numerical stability. Zero-phase filtering is achieved by forward-backward (filtfilt) application.

### Welch's Method (PSD)

The signal is divided into overlapping segments (50% overlap), each windowed with a Hann window, and the averaged periodogram is computed:

```
PSD(f) = (1 / (M * fs * W)) * sum_m |X_m(f)|^2
```

where M is the number of segments, fs is the sample rate, and W is the window power.

### Phase Locking Value

```
PLV = |<exp(j * (phi_a(t) - phi_b(t)))>|
```

Instantaneous phases are extracted via the Hilbert transform after bandpass filtering.

### Hilbert Transform

The analytic signal is computed via the FFT:
1. Compute X(f) = FFT(x(t))
2. Zero negative frequencies, double positive frequencies
3. z(t) = IFFT(X_analytic(f))

Instantaneous amplitude = |z(t)|, instantaneous phase = arg(z(t)).

### Spectral Entropy

```
H = -sum(p_k * log2(p_k))
```

where p_k = PSD(f_k) / sum(PSD) is the normalized power distribution.

## Performance Notes

- All filters use SOS (second-order sections) cascade for numerical stability with high filter orders
- Zero-phase filtering (forward-backward) eliminates phase distortion at the cost of 2x computation
- FFT operations use the `rustfft` crate (pure Rust, no external dependencies)
- Connectivity matrix computation is O(N^2) in the number of channels; each pair requires bandpass filtering + Hilbert transform
- The `simd` feature flag is reserved for future SIMD-accelerated inner loops

## License

MIT OR Apache-2.0

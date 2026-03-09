# ruv-neural-sensor

**rUv Neural** -- Sensor data acquisition for NV diamond, OPM, EEG, and simulated sources.

Part of the [rUv Neural](https://github.com/ruvnet/RuView) brain topology analysis pipeline.

## Overview

`ruv-neural-sensor` provides a uniform `SensorSource` trait interface for acquiring multi-channel neural signal data from multiple sensor modalities. Each sensor backend is feature-gated so you only compile what you need.

## Supported Sensor Types

| Sensor | Feature Flag | Sensitivity | Description |
|--------|-------------|-------------|-------------|
| Simulated | `simulator` (default) | Configurable | Synthetic data for testing and development |
| NV Diamond | `nv_diamond` | ~10 fT/sqrt(Hz) | Nitrogen-vacancy diamond magnetometer |
| OPM | `opm` | ~7 fT/sqrt(Hz) | Optically pumped magnetometer (SERF mode) |
| EEG | `eeg` | ~1000 fT/sqrt(Hz) | Electroencephalography (10-20 system) |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `simulator` | Yes | Simulated sensor array with configurable noise, oscillations, and events |
| `nv_diamond` | No | NV diamond magnetometer with ODMR signal processing stub |
| `opm` | No | OPM array with SERF mode, cross-talk matrix, active shielding |
| `eeg` | No | EEG with 10-20 electrode system, impedance tracking |

## Usage

### Basic Simulator

```rust
use ruv_neural_sensor::simulator::SimulatedSensorArray;
use ruv_neural_sensor::SensorSource;

// Create a 16-channel simulator at 1000 Hz with 10 fT/sqrt(Hz) noise.
let mut sim = SimulatedSensorArray::new(16, 1000.0);

// Inject alpha rhythm (10 Hz, 100 fT amplitude).
sim.inject_alpha(100.0);

// Acquire 1 second of data.
let data = sim.read_chunk(1000).unwrap();
assert_eq!(data.num_channels, 16);
assert_eq!(data.num_samples, 1000);
```

### Custom Noise Floor

```rust
use ruv_neural_sensor::simulator::SimulatedSensorArray;

let mut sim = SimulatedSensorArray::new(8, 500.0)
    .with_noise(5.0); // 5 fT/sqrt(Hz) noise density
```

### Injecting Events

```rust
use ruv_neural_sensor::simulator::{SimulatedSensorArray, SensorEvent};
use ruv_neural_sensor::SensorSource;

let mut sim = SimulatedSensorArray::new(4, 1000.0);
sim.inject_event(SensorEvent::Spike {
    channel: 0,
    amplitude_ft: 500.0,
    sample_offset: 100,
});
let data = sim.read_chunk(200).unwrap();
```

## Calibration

The `calibration` module provides tools for sensor gain/offset correction and cross-sensor alignment.

```rust
use ruv_neural_sensor::calibration::{CalibrationData, calibrate_channel, estimate_noise_floor, cross_calibrate};

// Define calibration data.
let cal = CalibrationData {
    gains: vec![2.0, 1.5],
    offsets: vec![10.0, 5.0],
    noise_floors: vec![1.0, 2.0],
};

// Apply correction: corrected = (raw - offset) * gain
let corrected = calibrate_channel(100.0, 0, &cal);

// Estimate noise floor from a quiet recording.
let quiet_data = vec![0.1, -0.2, 0.15, -0.1];
let noise = estimate_noise_floor(&quiet_data);

// Cross-calibrate two sensors.
let reference = vec![10.0, 20.0, 30.0];
let target = vec![5.0, 10.0, 15.0];
let (gain, offset) = cross_calibrate(&reference, &target);
```

## Quality Monitoring

The `quality` module tracks real-time signal quality across channels.

```rust
use ruv_neural_sensor::quality::{QualityMonitor, SignalQuality};

let mut monitor = QualityMonitor::new(4);

// Check quality of 4 channels.
let ch0 = vec![/* ... */];
let ch1 = vec![/* ... */];
let ch2 = vec![/* ... */];
let ch3 = vec![/* ... */];
let qualities = monitor.check_quality(&[&ch0, &ch1, &ch2, &ch3]);

for (i, q) in qualities.iter().enumerate() {
    if q.below_threshold() {
        println!("Channel {i}: quality below threshold (SNR={:.1} dB)", q.snr_db);
    }
}
```

**Alert thresholds:**
- SNR < 3 dB
- Artifact probability > 0.5
- Saturation detected

## License

MIT OR Apache-2.0

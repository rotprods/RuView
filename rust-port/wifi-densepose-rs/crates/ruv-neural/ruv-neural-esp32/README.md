# rUv Neural ESP32

ESP32 edge integration for neural sensor data acquisition and preprocessing. This crate provides lightweight processing that runs on ESP32 hardware for real-time sensor data acquisition before sending to the main RuVector backend.

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| MCU | ESP32-S3 (dual-core Xtensa LX7, 240 MHz) |
| Flash | 8 MB minimum |
| PSRAM | 2 MB recommended for multi-channel buffering |
| ADC | 12-bit SAR ADC (built-in), or external 16-bit via SPI |
| WiFi | 802.11 b/g/n (built-in) |
| Battery | 3.7V LiPo, 2000+ mAh recommended |

## Pin Configuration

| GPIO | Function | Module | Notes |
|------|----------|--------|-------|
| 36 | ADC1_CH0 | `adc` | NV diamond sensor input (default) |
| 37 | ADC1_CH1 | `adc` | OPM sensor input |
| 38 | ADC1_CH2 | `adc` | EEG sensor input |
| 39 | ADC1_CH3 | `adc` | Auxiliary sensor input |
| 4 | ADC2_CH0 | `adc` | Battery voltage monitor |
| 16 | UART TX | `protocol` | Backend communication (if wired) |
| 17 | UART RX | `protocol` | Backend communication (if wired) |
| 2 | LED | `power` | Status indicator |

## Modules

### ADC (`adc.rs`)

Configurable multi-channel ADC reader with support for 12-bit and 16-bit resolution. Converts raw ADC values to physical units (femtotesla) using per-channel gain and offset calibration.

### Edge Preprocessing (`preprocessing.rs`)

Lightweight signal conditioning that runs on-device before data transmission:

- 50/60 Hz mains notch filters (IIR biquad)
- Configurable high-pass filter (default 0.5 Hz) for DC removal
- Configurable low-pass filter (default 200 Hz) for anti-aliasing
- Block-averaging downsampler
- Fixed-point IIR path for integer-only ESP32 math

### Communication Protocol (`protocol.rs`)

Binary packet format for ESP32-to-backend data transfer:

```
+--------+-----+--------+----------+------+---------+------+------+----------+
| Magic  | Ver | PktID  | Timestamp| NCh  | Samples | Data | Qual | Checksum |
| 4B     | 1B  | 4B     | 8B       | 1B   | 2B      | var  | var  | 4B       |
+--------+-----+--------+----------+------+---------+------+------+----------+
  "rUvN"   1     u32      u64 us     u8     u16      i16[]  u8[]   CRC32
```

- Magic bytes: `rUvN` (0x72 0x55 0x76 0x4E)
- Fixed-point samples (i16) with per-channel scale factor for bandwidth efficiency
- CRC32 checksum (IEEE polynomial) for integrity verification
- JSON serialization in std mode; compact binary on embedded targets

### TDM Scheduler (`tdm.rs`)

Time-Division Multiplexing for collision-free multi-node operation:

```
|  Node 0  |  Node 1  |  Node 2  |  Node 3  |  Node 0  |  ...
|<-slot_d->|<-slot_d->|<-slot_d->|<-slot_d->|
|<-------------- frame_duration ------------>|
```

Supported sync methods:
- **GPS PPS** -- sub-microsecond accuracy
- **NTP Sync** -- millisecond accuracy over WiFi
- **WiFi Beacon** -- timestamp alignment from AP beacons
- **Leader-Follower** -- leader broadcasts sync pulses (default)

### Power Management (`power.rs`)

Battery life optimization through duty-cycle control:

| Mode | Current Draw | Estimated Runtime (2000 mAh) |
|------|-------------|------------------------------|
| Active | 240 mA | ~6.25 hours |
| LowPower | 80 mA | ~15 hours |
| UltraLowPower | 20 mA | ~60 hours |
| Sleep | 10 uA | ~22 years |

Automatic duty-cycle optimization targets a user-specified runtime by adjusting sample and WiFi duty cycles via binary search.

### Node Aggregator (`aggregator.rs`)

Collects packets from multiple ESP32 nodes and assembles them into a unified `MultiChannelTimeSeries`. Timestamp-based packet matching with configurable sync tolerance (default 1 ms).

## Build Instructions

```bash
# Build for host (std mode, simulation)
cd rust-port/wifi-densepose-rs/crates/ruv-neural
cargo build -p ruv-neural-esp32

# Run tests
cargo test -p ruv-neural-esp32

# Build with simulator feature
cargo build -p ruv-neural-esp32 --features simulator
```

## Features

| Feature | Description |
|---------|-------------|
| `std` (default) | Standard library support, simulated ADC |
| `no_std` | Bare-metal ESP32 deployment (no heap allocator required for core types) |
| `simulator` | ESP32 simulation mode for desktop development |

## License

MIT OR Apache-2.0

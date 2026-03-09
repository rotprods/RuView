# rUv Neural CLI

CLI tool for the rUv Neural brain topology analysis system. Provides commands for
simulating neural sensor data, analyzing brain connectivity graphs, computing
minimum cuts, running full analysis pipelines, and exporting results to multiple
visualization formats.

## Installation

```bash
cargo install --path .
```

Or build from the workspace root:

```bash
cargo build -p ruv-neural-cli --release
```

The binary is named `ruv-neural`.

## Command Reference

| Command    | Description                                           |
|------------|-------------------------------------------------------|
| `simulate` | Generate simulated multi-channel neural sensor data   |
| `analyze`  | Load and analyze a brain connectivity graph (JSON)    |
| `mincut`   | Compute minimum cut (Stoer-Wagner or multi-way)       |
| `pipeline` | Full end-to-end: simulate -> filter -> graph -> decode|
| `export`   | Export brain graph to D3, DOT, GEXF, CSV, or RVF     |
| `info`     | Show system info, crate versions, and capabilities    |

## Usage Examples

### Simulate Neural Data

Generate 64-channel simulated neural data at 1 kHz for 10 seconds:

```bash
ruv-neural simulate -c 64 -d 10.0 -s 1000.0 -o output.json
```

Default parameters (no arguments required):

```bash
ruv-neural simulate
```

### Analyze a Brain Graph

Load a graph from JSON and display topology metrics:

```bash
ruv-neural analyze -i brain_graph.json
```

With ASCII adjacency matrix visualization:

```bash
ruv-neural analyze -i brain_graph.json --ascii
```

Export per-node metrics to CSV:

```bash
ruv-neural analyze -i brain_graph.json --csv metrics.csv
```

### Compute Minimum Cut

Standard two-way Stoer-Wagner minimum cut:

```bash
ruv-neural mincut -i brain_graph.json
```

Multi-way cut with 4 partitions:

```bash
ruv-neural mincut -i brain_graph.json -k 4
```

### Run Full Pipeline

The pipeline command runs all stages end-to-end:

1. Generate simulated sensor data
2. Preprocess (bandpass filter 1-100 Hz)
3. Construct brain connectivity graph (PLV)
4. Compute minimum cut and topology metrics
5. Generate topology and spectral embeddings
6. Decode cognitive state
7. Display results summary

```bash
ruv-neural pipeline -c 32 -d 5.0
```

With ASCII dashboard visualization:

```bash
ruv-neural pipeline -c 16 -d 3.0 --dashboard
```

### Export Graph

Export to D3.js-compatible JSON:

```bash
ruv-neural export -i brain_graph.json -f d3 -o graph.d3.json
```

Export to Graphviz DOT:

```bash
ruv-neural export -i brain_graph.json -f dot -o graph.dot
```

All supported formats:

```bash
ruv-neural export -i graph.json -f d3   -o out.json   # D3.js JSON
ruv-neural export -i graph.json -f dot  -o out.dot    # Graphviz DOT
ruv-neural export -i graph.json -f gexf -o out.gexf   # GEXF XML
ruv-neural export -i graph.json -f csv  -o out.csv    # CSV edge list
ruv-neural export -i graph.json -f rvf  -o out.rvf    # RuVector File
```

### System Info

```bash
ruv-neural info
```

### Verbosity

Use `-v` flags for increased logging detail:

```bash
ruv-neural -v pipeline -c 8 -d 2.0      # INFO level
ruv-neural -vv pipeline -c 8 -d 2.0     # DEBUG level
ruv-neural -vvv pipeline -c 8 -d 2.0    # TRACE level
```

## Output Formats

| Format | Extension | Description                                    |
|--------|-----------|------------------------------------------------|
| D3     | `.json`   | D3.js force-directed graph with nodes and links|
| DOT    | `.dot`    | Graphviz DOT for rendering with `dot` or `neato`|
| GEXF   | `.gexf`   | Graph Exchange XML Format for Gephi            |
| CSV    | `.csv`    | Edge list with source, target, weight, metric  |
| RVF    | `.json`   | RuVector File format with adjacency matrix     |

## Pipeline Walkthrough

The `pipeline` command demonstrates the full rUv Neural analysis flow:

```
simulate -> filter -> PLV graph -> mincut -> embed -> decode
```

**Step 1 - Simulate**: Generates multi-channel neural data with alpha (10 Hz),
beta (20 Hz), and gamma (40 Hz) oscillations plus white noise.

**Step 2 - Filter**: Applies a 4th-order Butterworth bandpass filter (1-100 Hz)
using zero-phase SOS filtering.

**Step 3 - Graph**: Computes Phase Locking Value (PLV) between all channel pairs
and constructs a brain connectivity graph with edges above a PLV threshold.

**Step 4 - Mincut**: Runs the Stoer-Wagner algorithm for the global minimum cut,
revealing the natural partition boundary in the brain network.

**Step 5 - Embed**: Generates both topology-based and spectral (Laplacian
eigenvector) embeddings of the brain graph state.

**Step 6 - Decode**: Classifies the cognitive state (Rest, Focused, MotorPlanning)
using a threshold decoder on topology metrics.

**Step 7 - Results**: Displays a formatted summary table with all computed metrics.

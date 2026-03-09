# rUv Neural WASM

WebAssembly bindings for browser-based brain topology visualization. Part of the **rUv Neural** suite.

## Overview

`ruv-neural-wasm` exposes the core brain graph analysis pipeline to JavaScript via `wasm-bindgen`. It provides lightweight, WASM-compatible implementations of graph algorithms (Stoer-Wagner mincut, spectral embedding, topology metrics) that run entirely in the browser without server round-trips.

## Build

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/):

```bash
# Build for browser (ES modules)
wasm-pack build --target web

# Build for bundler (webpack, vite, etc.)
wasm-pack build --target bundler

# Build for Node.js
wasm-pack build --target nodejs

# Native check (no WASM target required)
cargo check -p ruv-neural-wasm
```

## JavaScript Usage

### Basic Graph Analysis

```javascript
import init, {
  create_brain_graph,
  compute_mincut,
  compute_topology_metrics,
  embed_graph,
  decode_state,
  to_viz_graph,
  version,
} from "./pkg/ruv_neural_wasm.js";

await init();

console.log("rUv Neural WASM v" + version());

// Define a brain connectivity graph
const graphJson = JSON.stringify({
  num_nodes: 4,
  edges: [
    { source: 0, target: 1, weight: 0.9, metric: "Coherence", frequency_band: "Alpha" },
    { source: 1, target: 2, weight: 0.3, metric: "Coherence", frequency_band: "Alpha" },
    { source: 2, target: 3, weight: 0.8, metric: "Coherence", frequency_band: "Alpha" },
    { source: 0, target: 3, weight: 0.7, metric: "Coherence", frequency_band: "Alpha" },
  ],
  timestamp: Date.now() / 1000,
  window_duration_s: 1.0,
  atlas: { Custom: 4 },
});

// Parse and validate
const graph = create_brain_graph(graphJson);

// Compute minimum cut
const mincut = compute_mincut(graphJson);
console.log("Min-cut value:", mincut.cut_value);
console.log("Partition A:", mincut.partition_a);
console.log("Partition B:", mincut.partition_b);

// Compute topology metrics
const metrics = compute_topology_metrics(graphJson);
console.log("Modularity:", metrics.modularity);
console.log("Fiedler value:", metrics.fiedler_value);
console.log("Global efficiency:", metrics.global_efficiency);

// Decode cognitive state
const metricsJson = JSON.stringify(metrics);
const state = decode_state(metricsJson);
console.log("Cognitive state:", state);

// Generate spectral embedding (2D)
const embedding = embed_graph(graphJson, 2);
console.log("Embedding dimension:", embedding.dimension);
```

### D3.js Visualization

```javascript
import { to_viz_graph } from "./pkg/ruv_neural_wasm.js";

const vizGraph = to_viz_graph(graphJson);

// vizGraph.nodes: [{ id, label, x, y, z, group, size, color }, ...]
// vizGraph.edges: [{ source, target, weight, is_cut, color }, ...]
// vizGraph.partitions: [[nodeIds...], [nodeIds...]] or null
// vizGraph.cut_edges: [edgeIndices...] or null

// Use with D3 force simulation
const simulation = d3
  .forceSimulation(vizGraph.nodes)
  .force("link", d3.forceLink(vizGraph.edges).id((d) => d.id))
  .force("charge", d3.forceManyBody().strength(-100))
  .force("center", d3.forceCenter(width / 2, height / 2));

// Color nodes by partition group
svg
  .selectAll("circle")
  .data(vizGraph.nodes)
  .enter()
  .append("circle")
  .attr("r", (d) => d.size * 5)
  .attr("fill", (d) => d.color);

// Highlight cut edges in red
svg
  .selectAll("line")
  .data(vizGraph.edges)
  .enter()
  .append("line")
  .attr("stroke", (d) => d.color)
  .attr("stroke-width", (d) => (d.is_cut ? 3 : 1));
```

### WebSocket Streaming

```javascript
import { StreamProcessor } from "./pkg/ruv_neural_wasm.js";

// Create processor: 256-sample window, 64-sample hop
const processor = new StreamProcessor(256, 64);

const ws = new WebSocket("ws://localhost:8080/neural-stream");

ws.onmessage = (event) => {
  const samples = new Float64Array(event.data);
  const stats = processor.push_samples(samples);

  if (stats) {
    console.log(`Window ${stats.window_index}: mean=${stats.mean.toFixed(3)}`);
    updateVisualization(stats);
  }
};

// Reset when switching sessions
function resetStream() {
  processor.reset();
}
```

### RVF File I/O

```javascript
import { load_rvf, export_rvf } from "./pkg/ruv_neural_wasm.js";

// Export graph to RVF binary
const rvfBytes = export_rvf(graphJson);

// Save as file download
const blob = new Blob([rvfBytes], { type: "application/octet-stream" });
const url = URL.createObjectURL(blob);

// Load RVF from file input
const fileInput = document.getElementById("rvf-file");
fileInput.onchange = async (e) => {
  const buffer = await e.target.files[0].arrayBuffer();
  const rvf = load_rvf(new Uint8Array(buffer));
  console.log("Loaded RVF:", rvf.header.data_type);
};
```

## API Reference

| Function | Description |
|----------|-------------|
| `create_brain_graph(json)` | Parse JSON into a BrainGraph |
| `compute_mincut(json)` | Stoer-Wagner minimum cut (max 500 nodes) |
| `compute_topology_metrics(json)` | Density, efficiency, modularity, Fiedler, entropy |
| `embed_graph(json, dim)` | Spectral embedding via power iteration |
| `decode_state(json)` | Classify cognitive state from metrics |
| `to_viz_graph(json)` | Convert to D3.js/Three.js-ready visualization data |
| `load_rvf(bytes)` | Parse RVF binary file |
| `export_rvf(json)` | Serialize graph to RVF binary |
| `version()` | Get crate version string |
| `StreamProcessor` | Sliding-window streaming data processor |

## Browser Compatibility

- Chrome 57+ / Edge 79+
- Firefox 52+
- Safari 11+
- All modern browsers with WebAssembly support

## Graph Size Limits

The Stoer-Wagner minimum cut algorithm runs in O(V^3) time. For browser performance:

| Nodes | Approximate Time |
|-------|-----------------|
| 68 (DK atlas) | < 10ms |
| 100 (Schaefer) | < 50ms |
| 200 (Schaefer) | < 500ms |
| 400 (Schaefer) | ~2-5s |
| 500 (max) | ~5-10s |

For larger graphs, use the native `ruv-neural-mincut` crate with server-side computation.

## License

MIT OR Apache-2.0

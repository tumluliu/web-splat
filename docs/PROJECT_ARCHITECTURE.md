# Web-Splat: High-Performance 3D Gaussian Splatting Renderer Architecture

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Rendering Pipeline](#rendering-pipeline)
4. [Performance Optimizations](#performance-optimizations)
5. [Core Improvements](#core-improvements)
6. [Platform Support](#platform-support)
7. [Usage Guide](#usage-guide)
8. [Extension Guide](#extension-guide)
9. [Performance Analysis](#performance-analysis)
10. [Development Workflow](#development-workflow)

## Project Overview

Web-Splat is a high-performance 3D Gaussian Splatting renderer implemented in Rust using WebGPU (WGPU). It provides real-time rendering of 3D scenes reconstructed using [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) techniques, delivering exceptional performance across multiple platforms.

### Key Features
- **Cross-Platform**: Runs natively on desktop and in web browsers via WebAssembly
- **High Performance**: Achieves >200 FPS on modern GPUs (NVIDIA 3090 RTX)
- **GPU-Accelerated Sorting**: Custom radix sort implementation for optimal splat ordering
- **Interactive UI**: Real-time scene exploration with egui-based interface
- **3D Scene Understanding**: AI-powered chat interface for scene querying and navigation
- **Compressed Formats**: Support for compressed Gaussian representations
- **Real-time Analytics**: Performance monitoring and debugging tools

## System Architecture

### Component Overview

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Application    │    │   UI Layer      │    │ Rendering Engine│
│                 │    │                 │    │                 │
│ • WindowContext │    │ • egui UI       │    │ • GaussianRenderer
│ • Event Loop    │    │ • Chat Interface│    │ • PreprocessPipe│
│ • State Mgmt    │    │ • Debug Windows │    │ • GPURSSorter   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Data Management │    │Platform Abstract│    │Scene Understanding
│                 │    │                 │    │                 │
│ • PointCloud    │    │ • WGPU          │    │ • Chat System   │
│ • Scene         │    │ • winit         │    │ • MCP Integration
│ • Highlighting  │    │ • Cross-platform│    │ • Object Detection
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Modules

#### 1. **WindowContext** (`src/lib.rs`)
The central application state manager that orchestrates all components:
- **Responsibilities**: Event handling, render loop coordination, state management
- **Key Features**: FPS monitoring, animation control, camera management
- **Interactions**: Coordinates between UI, renderer, and data layers

#### 2. **GaussianRenderer** (`src/renderer.rs`)
The heart of the 3D Gaussian Splatting rendering system:
- **Preprocessing**: Transforms 3D Gaussians to 2D screen-space representations
- **Sorting**: Orders splats by depth for correct alpha blending
- **Rasterization**: Renders sorted splats to screen
- **Performance**: Optimized for real-time rendering with GPU compute shaders

#### 3. **GPURSSorter** (`src/gpu_rs.rs`)
Custom GPU radix sort implementation ported from Fuchsia:
- **Purpose**: Sorts millions of splats by depth every frame
- **Algorithm**: Multi-pass radix sort with histogram computation
- **Performance**: Dramatically faster than CPU sorting for large datasets

#### 4. **PointCloud** (`src/pointcloud.rs`)
Manages 3D Gaussian data and metadata:
- **Formats**: Supports both uncompressed PLY and compressed NPZ formats
- **Optimization**: Efficient GPU buffer management and data layout
- **Features**: Bounding box computation, center-of-mass calculation

#### 5. **Chat System** (`src/chat.rs`, `src/ui.rs`)
AI-powered 3D scene understanding interface:
- **Capabilities**: Natural language querying of 3D scenes
- **Features**: Object detection, path planning, scene navigation
- **Integration**: MCP server communication with fallback to mock responses

## Rendering Pipeline

### High-Level Pipeline Flow

The rendering pipeline consists of three main stages executed every frame:

```
3D Gaussians → Preprocessing → Depth Sorting → Rasterization → Final Image
     ↓              ↓              ↓              ↓              ↓
 Point Cloud    Screen Space   Sorted Splats   Alpha Blend   Display
```

**Stage Breakdown**:
1. **Preprocessing** (Compute Shader): Transform 3D Gaussians to 2D screen space
2. **Sorting** (GPU Radix Sort): Order splats by depth for correct rendering
3. **Rasterization** (Fragment Shader): Render sorted splats with alpha blending

### Detailed Rendering Stages

#### Stage 1: Preprocessing (`preprocess.wgsl`)
**Purpose**: Transform 3D Gaussians into renderable 2D splats

**Key Operations**:
1. **Frustum Culling**: Remove Gaussians outside the view frustum
2. **Perspective Projection**: Transform 3D positions to screen space
3. **Covariance Computation**: Calculate 2D covariance matrices from 3D data
4. **Spherical Harmonics Evaluation**: Compute view-dependent colors
5. **Opacity Calculation**: Apply alpha values and optional mip-splatting

**Performance Optimizations**:
- Compute shader with 256 threads per workgroup
- Early termination for culled Gaussians
- Efficient memory access patterns

#### Stage 2: GPU Radix Sort (`radix_sort.wgsl`)
**Purpose**: Sort splats by depth for correct alpha blending

**Algorithm Phases**:
1. **Histogram Computation**: Count occurrences of each radix digit
2. **Prefix Sum**: Calculate cumulative histograms for scatter addresses
3. **Scatter**: Reorder keys and values based on sorted positions

**Implementation Details**:
- 8-bit radices for optimal performance
- Multi-pass sorting for 32-bit depth keys
- Workgroup-local memory optimization
- Indirect dispatch for dynamic workload sizing

#### Stage 3: Rasterization (`gaussian.wgsl`)
**Purpose**: Render sorted splats to screen with proper alpha blending

**Vertex Shader**:
- Generates screen-space quads for each splat
- Applies 2D covariance matrices for elliptical shapes
- Implements cutoff radius for performance

**Fragment Shader**:
- Evaluates Gaussian function: `exp(-dot(pos, pos))`
- Applies premultiplied alpha blending
- Implements early fragment discard for optimization

## Performance Optimizations

### 1. GPU Radix Sort Implementation

**Why GPU Sorting?**
3D Gaussian Splatting requires correct depth ordering for alpha blending. Traditional CPU sorting becomes a bottleneck with millions of splats.

**Fuchsia Radix Sort Port**:
- Ported from [Fuchsia RadixSort](https://fuchsia.googlesource.com/fuchsia/+/refs/heads/main/src/graphics/lib/compute/radix_sort/)
- Multi-pass algorithm handling 32-bit floating-point keys
- Optimized for modern GPU architectures

**Performance Impact**:
- **Before**: CPU sorting limited scenes to ~100K splats at 60 FPS
- **After**: GPU sorting handles 1M+ splats at >200 FPS

### 2. Compute Shader Preprocessing

**Advantages over Vertex Shaders**:
- Better workgroup utilization for variable workloads
- Shared memory optimizations for covariance calculations
- Early termination support for culled Gaussians

**Memory Access Optimization**:
```rust
// Efficient GPU buffer layout
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
struct Splat {
    v: Vector4<f16>,     // Covariance eigenvectors (packed)
    pos: Vector2<f16>,   // Screen position
    color: Vector4<f16>, // RGBA color
}
```

### 3. Indirect Rendering

**Dynamic Draw Calls**:
- Uses `draw_indirect` for variable splat counts
- GPU determines instance count during preprocessing
- Eliminates CPU-GPU synchronization overhead

### 4. Compressed Data Formats

**NPZ Compression Support**:
- Quantized Gaussian parameters
- Spherical harmonics coefficient compression
- Reduces memory bandwidth and loading times

## Core Improvements

### 1. WebGPU Cross-Platform Rendering
**Innovation**: First high-performance 3DGS renderer to support both native and web platforms
- **Native Performance**: Direct GPU access for maximum performance
- **Web Compatibility**: WebAssembly + WebGPU for browser deployment
- **Unified Codebase**: Same rendering code runs on all platforms

### 2. Real-Time Scene Understanding
**Feature**: AI-powered chat interface for 3D scene interaction
- **Natural Language Queries**: "Where is the table?" → Highlight objects
- **Path Planning**: "How do I get to the kitchen?" → Generate navigation paths
- **Object Detection**: Semantic understanding of 3D scenes

### 3. Advanced Debugging and Analytics
**Tools**: Comprehensive performance monitoring and visualization
- **Frame Time Analysis**: Real-time plots of rendering stages
- **GPU Memory Usage**: Buffer allocation tracking
- **Culling Statistics**: Visibility and performance metrics

### 4. Compressed Gaussian Support
**Optimization**: Support for compressed 3DGS representations
- **C3DGS Integration**: Compatible with compressed formats
- **Memory Efficiency**: Reduced VRAM usage for large scenes
- **Quality Preservation**: Maintains visual fidelity

## Platform Support

### Native Desktop Application

**Target Platforms**:
- Windows (DirectX 12, Vulkan)
- Linux (Vulkan)
- macOS (Metal)

**Build Command**:
```bash
cargo build --release --bin viewer
```

**Runtime Requirements**:
- GPU with compute shader support
- 4GB+ VRAM for large scenes
- OpenGL 4.5+ equivalent drivers

### Web Application (WASM)

**Browser Compatibility**:
- Chrome 113+ (WebGPU enabled)
- Firefox 131+ (with `dom.webgpu.enabled`)
- Safari 18+ (WebGPU support)

**Build Process**:
```bash
# Install WASM target and tools
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli

# Build WASM version
./build_wasm.sh
```

## Usage Guide

### Basic Viewer Usage

**Desktop Application**:
```bash
# Basic scene loading
cargo run --release --bin viewer point_cloud.ply cameras.json

# With compressed format support
cargo run --release --features npz --bin viewer scene.npz cameras.json
```

**Web Application**:
```bash
# Start local development server
cd public
python3 -m http.server 8000

# Open browser to:
# http://localhost:8000/demo.html - Demo gallery
# http://localhost:8000/index.html?file=./scene.ply&scene=./cameras.json
```

### UI Controls

**Camera Movement**:
- **Left Click + Drag**: Orbit camera around center
- **Right Click + Drag**: Pan camera target
- **Mouse Wheel**: Zoom in/out
- **Alt + Drag**: Tilt camera

**Keyboard Shortcuts**:
- **U**: Toggle UI visibility
- **R**: Random view from dataset
- **N**: Snap to nearest dataset camera
- **T**: Start/pause tracking shot animation
- **0-9**: Jump to specific dataset views
- **Page Up/Down**: Navigate through views

### Chat Interface

**Setup**:
1. Start the mock MCP server: `python mock_mcp_server.py`
2. Launch viewer with scene loaded
3. Open "💬 3D Scene Chat" window

**Example Queries**:
- "Where is the table?"
- "Show me the path to the kitchen"
- "Find all the chairs"
- "Navigate to the coffee machine"

## Extension Guide

### Adding Custom Shaders

**Location**: `src/shaders/`

**Adding a new shader**:
```rust
// In renderer.rs
let custom_shader = device.create_shader_module(include_wgsl!("shaders/custom.wgsl"));

let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    vertex: wgpu::VertexState {
        module: &custom_shader,
        entry_point: Some("vs_main"),
        // ...
    },
    // ...
});
```

### Extending the Chat System

**Adding New Query Types**:
```rust
// In src/chat.rs
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum McpResponse {
    Objects { /* existing */ },
    Path { /* existing */ },
    Animation { // New type
        keyframes: Vec<CameraKeyframe>,
        duration: f32,
        description: Option<String>,
    },
    Error { /* existing */ },
}
```

### Adding New Data Formats

**Point Cloud Loaders** (`src/io/`):
```rust
impl PointCloud {
    pub fn from_custom_format(
        device: &wgpu::Device,
        data: &[u8],
    ) -> Result<Self, anyhow::Error> {
        // Parse custom format and create PointCloud
    }
}
```

## Performance Analysis

### Benchmarking Results

**Hardware Performance** (at 1200x799 resolution):

| GPU               | Average FPS | Scene Complexity |
|-------------------|-------------|------------------|
| NVIDIA RTX 3090   | >200 FPS    | 1M+ Gaussians    |
| AMD R9 380 (2015) | ~130 FPS    | 500K Gaussians   |
| Intel Iris Xe     | ~60 FPS     | 250K Gaussians   |

**Memory Requirements**:
- **Minimum**: 2GB VRAM for basic scenes
- **Recommended**: 8GB VRAM for complex scenes
- **Large Scenes**: 16GB+ VRAM for datasets >2M Gaussians

### Performance Tools

**Built-in Measurement**:
```bash
cargo run --release --bin measure scene.ply cameras.json
```

**Profiling**:
```bash
# Frame time breakdown
RUST_LOG=debug cargo run --bin viewer

# GPU profiling with external tools
# Use RenderDoc, Nsight Graphics, or similar
```

## Development Workflow

### Build System

**Cargo Features**:
```toml
[features]
default = []
npz = ["dep:npyz"]           # Compressed format support
video = []                   # Video rendering tools
```

**Development Commands**:
```bash
# Debug build with logging
RUST_LOG=debug cargo run --bin viewer scene.ply

# Release build
cargo build --release

# WASM build
./build_wasm.sh

# Testing
cargo test
cargo test --features npz
```

### Debugging

**GPU Debugging**:
- Enable validation layers in debug builds
- Use RenderDoc or similar GPU profilers
- Monitor WGPU debug output

**Logging Configuration**:
```bash
# Rendering details
RUST_LOG=web_splats::renderer=debug cargo run

# GPU operations
RUST_LOG=wgpu=info cargo run

# All debug info
RUST_LOG=trace cargo run
```

---

This architecture provides a foundation for high-performance 3D Gaussian Splatting rendering with modern GPU techniques, cross-platform compatibility, and extensible design for future enhancements. 
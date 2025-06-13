# Web-Splat Documentation

Welcome to the comprehensive documentation for Web-Splat, a high-performance 3D Gaussian Splatting renderer implemented in Rust with WebGPU.

## 📚 Documentation Index

### Core Documentation
- **[Project Architecture](PROJECT_ARCHITECTURE.md)** - Complete system design, rendering pipeline, and technical architecture
- **[Original Project README](ORIGINAL_README.md)** - Basic project information and quick start guide

### Feature Documentation
- **[3D Scene Chat System](CHAT_README.md)** - AI-powered chat interface for 3D scene understanding
- **[MCP Server Migration Guide](MCP_MIGRATION_GUIDE.md)** - Guide for migrating from mock to production MCP servers

## 🚀 Quick Start

### Desktop Application
```bash
# Clone and build
git clone <repository>
cd web-splat
cargo build --release --bin viewer

# Run with your 3DGS data
cargo run --release --bin viewer point_cloud.ply cameras.json
```

### Web Application
```bash
# Build for web
./build_wasm.sh

# Start local server
cd public
python3 -m http.server 8000

# Open http://localhost:8000/demo.html
```

## 🏗️ Project Overview

Web-Splat is a cutting-edge 3D Gaussian Splatting renderer that achieves exceptional performance through:

- **GPU-Accelerated Rendering**: Custom radix sort and compute shader optimizations
- **Cross-Platform Support**: Native desktop and web browsers via WebAssembly
- **Real-Time Performance**: >200 FPS on modern GPUs with millions of Gaussians
- **3D Scene Understanding**: AI-powered chat interface for scene interaction
- **Advanced Analytics**: Comprehensive performance monitoring and debugging tools

## 🎯 Key Features

### Performance Optimizations
- **Fuchsia Radix Sort Port**: GPU-based sorting for optimal splat ordering
- **Compute Shader Preprocessing**: Efficient 3D to 2D transformation
- **Indirect Rendering**: Dynamic draw calls without CPU-GPU synchronization
- **Compressed Formats**: Support for NPZ compressed Gaussian data

### User Interface
- **Interactive Controls**: Real-time camera movement and scene exploration
- **Debug Visualization**: Frame time analysis and performance metrics
- **Chat Interface**: Natural language queries for scene understanding
- **Object Highlighting**: Visual feedback for detected objects and paths

### Cross-Platform Deployment
- **Native Performance**: Direct GPU access on Windows, Linux, macOS
- **Web Compatibility**: WebGPU + WebAssembly for browser deployment
- **Unified Codebase**: Same rendering code across all platforms

## 📖 Detailed Documentation

### [Project Architecture](PROJECT_ARCHITECTURE.md)
Comprehensive technical overview covering:
- System architecture and component design
- Detailed rendering pipeline explanation
- Performance optimization techniques
- Extension and development guides
- Cross-platform considerations

### [Chat System](CHAT_README.md)
3D scene understanding capabilities:
- Natural language scene querying
- Object detection and highlighting
- Path planning and navigation
- MCP server integration
- Usage examples and API reference

### [MCP Migration](MCP_MIGRATION_GUIDE.md)
Production deployment guide:
- Migration from mock to real MCP servers
- Authentication and security considerations
- Performance optimization for production
- Testing and validation procedures
- Troubleshooting common issues

## 🎮 Usage Examples

### Basic Scene Loading
```bash
# Load PLY point cloud with camera data
cargo run --release --bin viewer scene.ply cameras.json

# Load compressed NPZ format
cargo run --release --features npz --bin viewer scene.npz cameras.json
```

### Performance Measurement
```bash
# Benchmark rendering performance
cargo run --release --bin measure scene.ply cameras.json

# Generate frame sequence for video
cargo run --release --features video --bin render scene.ply cameras.json output/
```

### Interactive Controls
- **Camera**: Left-click drag (orbit), right-click drag (pan), wheel (zoom)
- **Navigation**: Number keys (0-9) for preset views, R for random view
- **Animation**: T for tracking shot, Page Up/Down for view sequence
- **UI**: U to toggle interface, various debug windows available

### Chat Interface
```
User: "Where is the table?"
AI: "Found a large wooden dining table in the center of the room"
[Highlights table object in 3D view]

User: "Show me the path to the kitchen"
AI: "Calculated smooth walking path through the room"
[Displays navigation path as 3D line]
```

## 🔧 Development

### Build Requirements
- Rust 1.70+ with cargo
- GPU with compute shader support
- 4GB+ VRAM for complex scenes

### Development Workflow
```bash
# Debug build with logging
RUST_LOG=debug cargo run --bin viewer

# Release build
cargo build --release

# Web build
./build_wasm.sh

# Testing
cargo test
cargo test --features npz
```

### Extension Points
- **Custom Shaders**: Add new rendering effects in `src/shaders/`
- **Data Formats**: Implement new loaders in `src/io/`
- **Chat Queries**: Extend response types in `src/chat.rs`
- **UI Components**: Add debug panels in `src/ui.rs`

## 📊 Performance Benchmarks

| Hardware   | Resolution | FPS  | Scene Size     |
|------------|------------|------|----------------|
| RTX 3090   | 1200x799   | >200 | 1M+ Gaussians  |
| RTX 2070   | 1200x799   | ~120 | 500K Gaussians |
| AMD R9 380 | 1200x799   | ~130 | 500K Gaussians |

Memory requirements:
- Minimum: 2GB VRAM
- Recommended: 8GB VRAM
- Large scenes: 16GB+ VRAM

## 🤝 Contributing

### Getting Started
1. Read the [Project Architecture](PROJECT_ARCHITECTURE.md) for system overview
2. Set up development environment following build instructions
3. Review extension guides for adding new features
4. Test changes with provided benchmark scenes

### Areas for Contribution
- **Rendering Optimizations**: Shader improvements, algorithm enhancements
- **Platform Support**: Additional OS/browser compatibility
- **Data Formats**: New point cloud and scene formats
- **UI/UX**: Interface improvements and new visualization tools
- **Documentation**: Examples, tutorials, and guides

## 📄 License

[Include license information here]

## 🙏 Acknowledgments

- Original 3D Gaussian Splatting research team
- Fuchsia RadixSort implementation
- WebGPU and wgpu-rs communities
- Rust graphics ecosystem contributors

---

For detailed technical information, start with the [Project Architecture](PROJECT_ARCHITECTURE.md) document. For quick usage, see the [Original README](ORIGINAL_README.md). For 3D scene interaction features, check the [Chat System documentation](CHAT_README.md). 
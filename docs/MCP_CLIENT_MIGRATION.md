# MCP Client Migration Guide

This document explains the migration from HTTP-based MCP communication to the rust-mcp-sdk SSE client implementation.

## Overview

The project now supports both HTTP and MCP client communication methods:

1. **HTTP Client** (legacy): Uses direct HTTP requests to `/query` endpoint
2. **MCP Client** (new): Uses the rust-mcp-sdk with SSE transport (native builds only)

## Platform Support

- **Native builds**: Full MCP client support with SSE transport
- **WASM builds**: HTTP client only (MCP client not available due to dependency limitations)

## Request Format

### HTTP Format (Legacy)
```json
{
  "messages": "where is the coffee machine?",
  "context": "3d_scene_understanding",
  "current_location": [0.0, 1.0, 0.0]
}
```

### MCP Format (New - Native Only)
```json
{
  "message": "where is the coffee machine?",
  "current_location": [0.0, 1.0, 0.0]
}
```

## Implementation Details

### New MCP Client Structure

The MCP client is implemented in `src/mcp_client.rs` with the following components:

1. **MCPClientHandler**: Handles MCP protocol messages
2. **MCPClient**: Main client interface for sending messages and receiving responses
3. **Request/Response Types**: Structured data types for MCP communication

### Key Features

- **SSE Transport**: Uses Server-Sent Events for real-time communication (native only)
- **Fallback Support**: Falls back to HTTP if MCP client fails
- **Mock Responses**: Provides mock responses for testing when no real server is available
- **Async Support**: Full async/await support for non-blocking operations
- **WASM Compatibility**: Graceful degradation to HTTP-only in web builds

### Usage

The client automatically tries MCP first (native builds), then falls back to HTTP:

```rust
// In handle_chat_message function
#[cfg(not(target_arch = "wasm32"))]
{
    // Try MCP client first, fall back to HTTP if it fails
    match rt.block_on(crate::chat::send_chat_message_mcp(msg_clone.clone(), &server_url, current_location)) {
        Ok(response) => {
            log::info!("Received MCP response successfully");
            self.pending_chat_responses.push((message, response));
        }
        Err(e) => {
            log::warn!("MCP request failed: {}, falling back to HTTP", e);
            // Fall back to HTTP request
            match rt.block_on(crate::chat::send_chat_message(msg_clone, &server_url, current_location)) {
                Ok(response) => {
                    log::info!("Received HTTP response successfully");
                    self.pending_chat_responses.push((message, response));
                }
                Err(e) => {
                    log::warn!("HTTP request also failed: {}, using mock response", e);
                    let mock_response = ui::create_mock_response(&message);
                    self.pending_chat_responses.push((message, mock_response));
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
{
    // WASM builds use HTTP only
    match rt.block_on(crate::chat::send_chat_message(msg_clone, &server_url, current_location)) {
        Ok(response) => {
            log::info!("Received HTTP response successfully");
            self.pending_chat_responses.push((message, response));
        }
        Err(e) => {
            log::warn!("HTTP request failed: {}, using mock response", e);
            let mock_response = ui::create_mock_response(&message);
            self.pending_chat_responses.push((message, mock_response));
        }
    }
}
```

## Server Configuration

### MCP Server URL Format

The client automatically converts HTTP URLs to SSE format:

- **Input**: `http://localhost:8080`
- **Output**: `http://localhost:8080/sse`

### Server Requirements

Your MCP server should:

1. **Support SSE Transport**: Implement Server-Sent Events endpoint
2. **Handle Tool Calls**: Respond to `scene_query` tool calls
3. **Return Structured Data**: Provide responses in the expected format

### Example MCP Server Response

```json
{
  "content": [
    {
      "type": "text",
      "text": "Found coffee machine at position [1.0, 0.5, 2.0]"
    }
  ],
  "is_error": false
}
```

## Migration Steps

### 1. Update Server URL

Change your server URL to point to the SSE endpoint:

```bash
# Old format
MCP_SERVER_URL=http://localhost:8080

# New format (automatic conversion)
MCP_SERVER_URL=http://localhost:8080/sse
```

### 2. Update Request Format

The client automatically handles the format conversion, but ensure your server expects:

```json
{
  "message": "where is the coffee machine?",
  "current_location": [0.0, 1.0, 0.0]
}
```

### 3. Test the Integration

Run the application and test chat functionality:

```bash
# Native build (supports MCP + HTTP)
cargo run --bin viewer -- path/to/your/pointcloud.ply --mcp-server-url http://localhost:8080

# WASM build (HTTP only)
./build_wasm.sh
```

## Platform-Specific Behavior

### Native Builds
- **Primary**: MCP client with SSE transport
- **Fallback**: HTTP client
- **Final**: Mock responses

### WASM Builds
- **Primary**: HTTP client
- **Fallback**: Mock responses
- **Note**: MCP client not available due to dependency limitations

## Troubleshooting

### Common Issues

1. **Connection Failed**: Ensure your MCP server supports SSE transport
2. **Tool Not Found**: Verify your server implements the `scene_query` tool
3. **Format Errors**: Check that request/response formats match expectations
4. **WASM Build Errors**: MCP client is not available in WASM builds - use HTTP only

### Debug Logging

Enable debug logging to see detailed MCP communication:

```bash
RUST_LOG=debug cargo run --bin viewer -- path/to/your/pointcloud.ply
```

### Fallback Behavior

If MCP client fails, the system automatically falls back to HTTP:

1. **MCP Client**: Primary method using rust-mcp-sdk (native only)
2. **HTTP Client**: Fallback method using reqwest
3. **Mock Response**: Final fallback for testing

## Future Enhancements

1. **Real MCP Server Integration**: Replace mock responses with actual server responses
2. **WASM MCP Support**: Investigate WASM-compatible MCP client alternatives
3. **Advanced Error Handling**: Improve error recovery and user feedback
4. **Performance Optimization**: Optimize for high-frequency requests

## Dependencies

The MCP client requires these additional dependencies (native builds only):

```toml
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
rust-mcp-sdk = { version = "0.5.0", features = ["client", "2025_06_18"], default-features = false }
async-trait = "0.1"
```

## Testing

Run the test suite to verify MCP client functionality:

```bash
# Native tests
cargo test mcp_client::tests

# WASM tests (HTTP only)
cargo test --target wasm32-unknown-unknown
```

This will test client creation and basic functionality for the appropriate platform. 
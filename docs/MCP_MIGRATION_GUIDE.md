# MCP Server Migration Guide

This document provides a comprehensive guide for migrating from the current mock MCP server implementation to a real MCP server for 3D scene understanding.

## Current Implementation Overview

### Architecture Summary
The current chat system consists of:
- **Frontend**: Rust-based UI with egui (`src/ui.rs`)
- **Chat State Management**: Rust structs in `src/chat.rs`
- **HTTP Client**: Cross-platform HTTP implementation (native + WASM)
- **Mock Server**: Python Flask server (`mock_mcp_server.py`)

### Current API Interface

#### Request Format
**Endpoint**: `POST {server_url}/query`
**Headers**: `Content-Type: application/json`
**Body**:
```json
{
  "query": "Where is the table?",
  "context": "3d_scene_understanding"
}
```

#### Response Format
The server must return one of three response types:

##### 1. Objects Response
```json
{
  "type": "objects",
  "objects": [
    {
      "name": "Dining Table",
      "position": [0.0, 4.0, -0.8],
      "attributes": {
        "type": "dining_table",
        "material": "wood",
        "size": "large",
        "width": "2.0",
        "height": "1.0",
        "depth": "1.5"
      },
      "confidence": 0.95
    }
  ],
  "description": "Found a large wooden dining table in the center of the room"
}
```

##### 2. Path Response
```json
{
  "type": "path",
  "path": {
    "waypoints": [
      [-2.0, 4.1, -3.0],
      [-1.8, 4.1, -2.8],
      [0.0, 4.1, -2.0]
    ],
    "description": "Smooth navigation path from entrance to seating area"
  },
  "description": "Calculated smooth walking path through the room"
}
```

##### 3. Error Response
```json
{
  "type": "error",
  "message": "Could not find any objects matching your query"
}
```

## Migration Checklist

### Phase 1: Pre-Migration Assessment

#### ✅ Server Requirements Analysis
- [ ] **Endpoint Compatibility**: Ensure your MCP server implements the `/query` POST endpoint
- [ ] **Response Format**: Verify the server returns responses in the expected JSON format
- [ ] **CORS Support**: If accessing from WASM, ensure CORS headers are configured
- [ ] **Authentication**: Determine if authentication is required (not currently implemented)
- [ ] **Rate Limiting**: Check if the server has rate limiting that might affect testing

#### ✅ Data Format Validation
- [ ] **Position Coordinates**: Verify the coordinate system matches your 3D scene
- [ ] **Object Attributes**: Ensure attribute names and values match your domain
- [ ] **Confidence Scores**: Confirm confidence values are in the range [0.0, 1.0]
- [ ] **Path Waypoints**: Validate waypoint format and coordinate system

### Phase 2: Server Integration

#### ✅ Configuration Update
1. **Update Server URL**: Modify the default server URL in `src/chat.rs`:
```rust
impl Default for ChatState {
    fn default() -> Self {
        Self {
            // ... other fields
            mcp_server_url: "https://your-mcp-server.com".to_string(), // Update this
        }
    }
}
```

2. **Environment Configuration**: Consider adding environment variable support:
```rust
mcp_server_url: std::env::var("MCP_SERVER_URL")
    .unwrap_or_else(|_| "https://your-mcp-server.com".to_string()),
```

#### ✅ Authentication Implementation (if needed)
If your MCP server requires authentication, update the HTTP client in `src/chat.rs`:

**For Native (non-WASM)**:
```rust
let client = reqwest::Client::new();
let response = client
    .post(&url)
    .header("Content-Type", "application/json")
    .header("Authorization", format!("Bearer {}", api_token)) // Add this
    .json(&request_body)
    .send()
    .await?;
```

**For WASM**:
```rust
headers
    .set("Content-Type", "application/json")
    .map_err(|e| format!("Failed to set content-type: {:?}", e))?;
headers
    .set("Authorization", &format!("Bearer {}", api_token)) // Add this
    .map_err(|e| format!("Failed to set authorization: {:?}", e))?;
```

#### ✅ Error Handling Enhancement
Update error handling in `send_chat_message` functions:

```rust
let status = response.status();
if status.is_success() {
    // ... existing success handling
} else {
    let error_text = response
        .text()
        .await
        .unwrap_or_else(|_| "Unknown error".to_string());
    
    // Enhanced error handling
    match status.as_u16() {
        401 => Ok(McpResponse::Error {
            message: "Authentication failed. Please check your API credentials.".to_string(),
        }),
        429 => Ok(McpResponse::Error {
            message: "Rate limit exceeded. Please try again later.".to_string(),
        }),
        500..=599 => Ok(McpResponse::Error {
            message: format!("Server error ({}): {}", status, error_text),
        }),
        _ => Ok(McpResponse::Error {
            message: format!("Request failed ({}): {}", status, error_text),
        }),
    }
}
```

### Phase 3: Testing and Validation

#### ✅ Connection Testing
Create a simple test to verify server connectivity:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_connection() {
        let response = send_chat_message(
            "test connection".to_string(),
            "https://your-mcp-server.com"
        ).await;
        
        assert!(response.is_ok(), "Server connection failed: {:?}", response);
    }
}
```

#### ✅ Response Validation
Test various query types:

```rust
#[tokio::test]
async fn test_object_query() {
    let response = send_chat_message(
        "Where is the table?".to_string(),
        "https://your-mcp-server.com"
    ).await.unwrap();
    
    match response {
        McpResponse::Objects { objects, .. } => {
            assert!(!objects.is_empty(), "No objects returned");
            // Validate object structure
            for obj in objects {
                assert!(!obj.name.is_empty(), "Object name is empty");
                assert!(obj.position.len() == 3, "Invalid position format");
            }
        },
        _ => panic!("Expected objects response"),
    }
}
```

#### ✅ Performance Testing
Monitor response times and implement timeouts:

```rust
use tokio::time::{timeout, Duration};

let response = timeout(
    Duration::from_secs(30), // 30-second timeout
    client.post(&url)
        .json(&request_body)
        .send()
).await;

match response {
    Ok(Ok(resp)) => {
        // Handle successful response
    },
    Ok(Err(e)) => {
        // Handle HTTP error
    },
    Err(_) => {
        // Handle timeout
        Ok(McpResponse::Error {
            message: "Request timed out. Please try again.".to_string(),
        })
    }
}
```

### Phase 4: Fallback Strategy

#### ✅ Graceful Degradation
Implement fallback to mock responses when the server is unavailable:

```rust
pub async fn send_chat_message_with_fallback(
    message: String,
    server_url: &str,
) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
    // First, try the real server
    match send_chat_message(message.clone(), server_url).await {
        Ok(response) => Ok(response),
        Err(e) => {
            log::warn!("MCP server request failed: {}, falling back to mock", e);
            // Fallback to mock response
            Ok(crate::ui::create_mock_response(&message))
        }
    }
}
```

Update the caller in `src/lib.rs`:
```rust
match rt.block_on(crate::chat::send_chat_message_with_fallback(msg_clone, &server_url)) {
    Ok(response) => {
        log::info!("Received response successfully");
        self.pending_chat_responses.push((message, response));
    }
    Err(e) => {
        log::error!("Both server and fallback failed: {}", e);
        let error_response = McpResponse::Error {
            message: "Service temporarily unavailable. Please try again later.".to_string(),
        };
        self.pending_chat_responses.push((message, error_response));
    }
}
```

### Phase 5: Production Deployment

#### ✅ Configuration Management
- [ ] **Environment Variables**: Use environment variables for server URL and API keys
- [ ] **Configuration Files**: Consider using a configuration file for complex settings
- [ ] **Build Variants**: Create different builds for development/staging/production

#### ✅ Monitoring and Logging
- [ ] **Request Logging**: Log all requests and responses for debugging
- [ ] **Error Metrics**: Track error rates and types
- [ ] **Performance Metrics**: Monitor response times and success rates

#### ✅ Security Considerations
- [ ] **HTTPS Only**: Ensure all communications use HTTPS in production
- [ ] **API Key Security**: Store API keys securely, never in source code
- [ ] **Input Validation**: Validate user inputs before sending to server
- [ ] **Rate Limiting**: Implement client-side rate limiting if needed

## Server Development Guidelines

If you're developing your own MCP server, follow these guidelines:

### Required Endpoints

#### POST /query
- **Purpose**: Process natural language queries about 3D scenes
- **Input**: JSON with `query` and `context` fields
- **Output**: JSON with `type`, data, and optional `description`

### Response Requirements

#### Objects Response
- **`objects`**: Array of objects with `name`, `position` (3D coordinates), `attributes` (key-value pairs), and optional `confidence`
- **`description`**: Human-readable summary of the findings

#### Path Response
- **`path.waypoints`**: Array of 3D coordinates representing the navigation path
- **`path.description`**: Description of the path
- **`description`**: Human-readable summary

#### Error Response
- **`message`**: Clear, user-friendly error message

### Best Practices

1. **Coordinate System**: Use consistent coordinate system (right-handed, Y-up recommended)
2. **Object Attributes**: Include relevant attributes like type, material, size, color
3. **Confidence Scores**: Provide confidence values when possible (0.0 to 1.0)
4. **Error Messages**: Return helpful error messages for debugging
5. **Performance**: Aim for sub-second response times
6. **Validation**: Validate input queries and return appropriate errors

## Troubleshooting

### Common Issues

#### Server Connection Failed
**Symptoms**: "Server returned status: 500" or connection timeout
**Solutions**:
- Verify server URL is correct and accessible
- Check network connectivity
- Ensure server is running and healthy
- Verify CORS configuration for WASM builds

#### Invalid Response Format
**Symptoms**: JSON parsing errors or unexpected response structure
**Solutions**:
- Validate response JSON against expected schema
- Check server logs for errors
- Ensure `type` field matches expected values (`objects`, `path`, `error`)

#### Authentication Issues
**Symptoms**: 401 Unauthorized errors
**Solutions**:
- Verify API credentials are correct
- Check if authentication headers are properly set
- Ensure API key hasn't expired

#### CORS Issues (WASM only)
**Symptoms**: "Access-Control-Allow-Origin" errors in browser console
**Solutions**:
- Configure server to include proper CORS headers
- Ensure preflight requests are handled correctly
- Consider using a proxy for development

### Debug Tools

#### Enable Detailed Logging
Set the `RUST_LOG` environment variable:
```bash
RUST_LOG=debug cargo run --bin viewer
```

#### Test Server Manually
Use curl to test the server directly:
```bash
curl -X POST https://your-mcp-server.com/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is the table?", "context": "3d_scene_understanding"}'
```

## Migration Timeline

### Week 1: Assessment
- [ ] Analyze target MCP server capabilities
- [ ] Validate API compatibility
- [ ] Identify any required changes

### Week 2: Implementation
- [ ] Update configuration
- [ ] Implement authentication if needed
- [ ] Add enhanced error handling
- [ ] Implement fallback strategy

### Week 3: Testing
- [ ] Unit tests for HTTP client
- [ ] Integration tests with real server
- [ ] Performance testing
- [ ] Error scenario testing

### Week 4: Deployment
- [ ] Production configuration
- [ ] Monitoring setup
- [ ] Documentation updates
- [ ] User training if needed

## Support and Resources

### Key Files to Modify
1. **`src/chat.rs`**: HTTP client implementation, data structures
2. **`src/lib.rs`**: Chat message handling, async request management
3. **`src/ui.rs`**: UI updates, mock response removal (optional)

### Testing Resources
1. **Mock Server**: Keep `mock_mcp_server.py` for local testing
2. **Unit Tests**: Add comprehensive test coverage
3. **Integration Tests**: Test with real server in CI/CD

### Documentation
1. Update `CHAT_README.md` with new server information
2. Create API documentation for your MCP server
3. Document deployment and configuration procedures

---

This migration guide provides a comprehensive roadmap for transitioning from the mock MCP server to a production-ready implementation. Follow the checklist systematically to ensure a smooth migration with minimal disruption to the user experience. 
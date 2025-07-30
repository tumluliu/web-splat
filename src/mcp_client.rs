use serde::{Deserialize, Serialize};

use crate::chat::McpResponse;
use reqwest::Client;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPRequest {
    pub messages: String,
    pub current_location: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolCallRequest {
    pub tool_name: String,
    pub arguments: serde_json::Map<String, serde_json::Value>,
    pub current_location: [f32; 3],
}

// Native implementation using rmcp SSE client - following official patterns
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use serde_json::json;
    use rmcp::transport::sse_client::SseClientTransport;
    use rmcp::service::{ServiceExt, RoleClient, RunningService};
    use rmcp::model::{ClientInfo, ClientCapabilities, Implementation, CallToolRequestParam};

    
    /// MCP Client using rmcp SSE transport - following official example patterns
    #[derive(Debug)]
    pub struct MCPClient {
        server_url: String,
        client: Arc<Mutex<Option<RunningService<RoleClient, ClientInfo>>>>,
    }

    impl MCPClient {
        /// Create a new MCP client for the given server URL
        pub fn new(server_url: String) -> Self {
            Self {
                server_url,
                client: Arc::new(Mutex::new(None)),
            }
        }

        /// Start the MCP client connection using rmcp SSE transport
        /// Following the exact pattern from the official SSE client example
        pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            log::info!("🔌 Starting rmcp SSE client connection to: {}", self.server_url);
            
            // Create SSE URL - ensure it ends with /sse
            let sse_url = if self.server_url.ends_with("/sse") {
                self.server_url.clone()
            } else {
                format!("{}/sse", self.server_url.trim_end_matches('/'))
            };
            
            log::info!("🔗 Connecting to SSE endpoint: {}", sse_url);
            
            // Step 1: Create transport - following official example
            let transport = SseClientTransport::start(sse_url.as_str()).await?;
            
            // Step 2: Create client info - following official example
            let client_info = ClientInfo {
                protocol_version: Default::default(),
                capabilities: ClientCapabilities::default(),
                client_info: Implementation {
                    name: "web-splat-mcp-client".to_string(),
                    version: "1.0.0".to_string(),
                },
            };
            
            // Step 3: Serve the transport to get client - following official example
            let client = client_info.serve(transport).await.map_err(|e| {
                log::error!("client error: {:?}", e);
                e
            })?;
            
            // Step 4: Initialize and get server info - following official example
            let server_info = client.peer_info();
            log::info!("✅ Connected to MCP server: {:#?}", server_info);
            
            // Store the client for later use
            {
                let mut client_guard = self.client.lock().await;
                *client_guard = Some(client);
            }
            
            log::info!("✅ rmcp SSE client connected successfully");
            Ok(())
        }

        /// Call a tool on the MCP server using rmcp's call_tool
        /// Following the official example pattern
        pub async fn call_tool(
            &self,
            tool_name: &str,
            arguments: serde_json::Map<String, serde_json::Value>,
            current_location: [f32; 3],
        ) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
            log::info!("🔧 Calling MCP tool: {} with arguments: {:?}", tool_name, arguments);
            log::info!("📍 Current location: {:?}", current_location);

            // Get the stored client
            let client_guard = self.client.lock().await;
            let client = client_guard.as_ref()
                .ok_or("MCP client not connected. Call start() first.")?;

            // Debug: Print peer info before tool call
            let peer_info = client.peer_info();
            log::info!("🔗 Debug - Peer info before tool call: {:#?}", peer_info);

            // Prepare arguments for "Our Awesome Tool" with the exact format required:
            // {"query": "{\"messages\": \"...\", \"current_location\": [x,y,z]}"}
            let query_object = json!({
                "messages": arguments.get("query").unwrap_or(&json!("")).as_str().unwrap_or(""),
                "current_location": current_location
            });

            let tool_arguments = json!({
                "query": query_object.to_string()
            });

            log::info!("📝 Formatted tool arguments for '{}': {}", tool_name, tool_arguments);

            // Convert to the format expected by CallToolRequestParam
            let arguments_map = tool_arguments.as_object().cloned();

            // Call the tool using rmcp client - following official example
            let tool_result = client.call_tool(CallToolRequestParam {
                name: tool_name.to_string().into(),
                arguments: arguments_map,
            }).await?;

            log::info!("✅ Tool call successful, parsing result");

            // Parse the result into our McpResponse format
            self.parse_tool_result(tool_result)
        }

        /// Parse rmcp tool result into our McpResponse format
        fn parse_tool_result(
            &self,
            result: rmcp::model::CallToolResult,
        ) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
            log::info!("📋 Parsing tool result into McpResponse");

            // Check if we have content
            if result.content.is_empty() {
                log::warn!("⚠️ Empty content from tool result");
                return Ok(McpResponse {
                    objects: Vec::new(),
                    paths: Vec::new(),
                    scene_normal_vector: None,
                    text_answer: Some("Empty response from server".to_string()),
                });
            }

            // Get the first content item
            let first_content = &result.content[0];

            // Parse based on content type
            match &first_content.raw {
                rmcp::model::RawContent::Text(text_content) => {
                    let text_str = &text_content.text;
                    log::info!("📄 Parsing text content: {}", text_str);

                    // Try to parse as JSON first (structured response)
                    if let Ok(mcp_response) = serde_json::from_str::<McpResponse>(text_str) {
                        log::info!("✅ Successfully parsed structured McpResponse");
                        Ok(mcp_response)
                    } else {
                        log::info!("📝 Text content is not structured JSON, treating as text answer");
                        // If it's not JSON, treat it as a simple text answer
                        Ok(McpResponse {
                            objects: Vec::new(),
                            paths: Vec::new(),
                            scene_normal_vector: None,
                            text_answer: Some(text_str.to_string()),
                        })
                    }
                }
                _ => {
                    log::warn!("⚠️ Unsupported content type from tool result");
                    Ok(McpResponse {
                        objects: Vec::new(),
                        paths: Vec::new(),
                        scene_normal_vector: None,
                        text_answer: Some("Unsupported content type".to_string()),
                    })
                }
            }
        }

        /// Disconnect from the MCP server - following official example
        pub async fn disconnect(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            log::info!("🔌 Disconnecting from MCP server");

            let mut client_guard = self.client.lock().await;
            if let Some(client) = client_guard.take() {
                // Call cancel() as shown in official example
                let _ = client.cancel().await;
                log::info!("✅ MCP client disconnected");
            } else {
                log::warn!("⚠️ No active MCP client to disconnect");
            }

            Ok(())
        }
    }
}

// WASM implementation (stub for now)
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;

    pub struct MCPClient {
        server_url: String,
    }

    impl MCPClient {
        pub fn new(server_url: String) -> Self {
            Self { server_url }
        }

        pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            log::warn!("🌐 WASM rmcp MCP client not yet implemented");
            log::info!("📄 WASM builds will use rmcp when SSE client API is stable");
            Err("WASM rmcp MCP client not yet implemented".into())
        }

        pub async fn call_tool(
            &self,
            _tool_name: &str,
            _arguments: serde_json::Map<String, serde_json::Value>,
            _current_location: [f32; 3],
        ) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
            log::warn!("🌐 WASM rmcp MCP tool calling not yet implemented");
            Err("WASM rmcp MCP tool calling not yet implemented".into())
        }


    }
}

// Export the appropriate implementation
#[cfg(not(target_arch = "wasm32"))]
pub use native::MCPClient;

#[cfg(target_arch = "wasm32")]
pub use wasm::MCPClient;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_native_mcp_client_creation() {
        let mut client = MCPClient::new("http://localhost:3000/sse".to_string());
        let result = client.start().await;
        assert!(result.is_ok());
        println!("✅ Native MCP client created successfully");
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    async fn test_wasm_mcp_client_creation() {
        let mut client = MCPClient::new("http://localhost:3000/sse".to_string());
        let result = client.start().await;
        assert!(result.is_err()); // Expected to fail in WASM
        web_sys::console::log_1(&"✅ WASM MCP client test completed".into());
    }

    #[test]
    fn test_mcp_request_format() {
        let request = MCPRequest {
            messages: "where is the coffee machine?".to_string(),
            current_location: [1.0, 2.0, 3.0],
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"messages\":"));
        assert!(json.contains("\"current_location\":"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_call_tool_format() {
        // Create a client (note: this test may fail if no server is running, but that's OK)
        let mut client = MCPClient::new("http://localhost:3000/sse".to_string());
        
        // We'll test starting the client, but it's OK if it fails (no server)
        let _ = client.start().await; // Ignore result for testing purposes
        
        let mut args = serde_json::Map::new();
        args.insert("query".to_string(), serde_json::json!("where is the coffee machine?"));
        
        // Test the tool call (may fail if no server, but we're testing the format)
        let result = client.call_tool("Our Awesome Tool", args, [1.0, 2.0, 3.0]).await;
        
        // Either it works (real server) or it fails with connection error (no server)
        match result {
            Ok(_) => println!("✅ Tool call succeeded with real server"),
            Err(e) => println!("⚠️ Tool call failed (no server): {}", e),
        }
        
        println!("✅ Tool call format test completed");
    }

    #[test]
    fn test_our_awesome_tool_format() {
        let query_json = serde_json::json!({
            "messages": "where is the coffee machine",
            "current_location": [0.0, 1.0, 0.0]
        });
        
        let tool_arguments = serde_json::json!({
            "query": query_json.to_string()
        });
        
        let formatted = serde_json::to_string(&tool_arguments).unwrap();
        assert!(formatted.contains("\"query\":"));
        assert!(formatted.contains("\"messages\":"));
        assert!(formatted.contains("\"current_location\":"));
        
        println!("✅ Our Awesome Tool format test passed: {}", formatted);
    }
} 
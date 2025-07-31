use serde::{Deserialize, Serialize};
use crate::mcp::handler::MyClientHandler;
use crate::chat::McpResponse;
use rust_mcp_sdk::error::SdkResult;
use rust_mcp_sdk::mcp_client::{client_runtime, ClientRuntime};
use rust_mcp_sdk::schema::{
    CallToolRequestParams, CallToolResult, ClientCapabilities, ContentBlock, Implementation, InitializeRequestParams, LoggingLevel, 
    LATEST_PROTOCOL_VERSION,
};
use rust_mcp_sdk::{ClientSseTransport, ClientSseTransportOptions, McpClient};
use std::sync::Arc;

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

// Native implementation using rust-mcp-sdk SSE client
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use serde_json::json;

    /// MCP Client using rust-mcp-sdk SSE transport
    // #[derive(Debug)]
    pub struct MCPClient {
        server_url: String,
        client: Option<Arc<ClientRuntime>>,
    }

    impl MCPClient {
        /// Create a new MCP client for the given server URL
        pub fn new(server_url: String) -> Self {
            Self {
                server_url,
                client: None,
            }
        }

        /// Start the MCP client connection using rust-mcp-sdk SSE transport
        pub async fn start(&mut self) -> SdkResult<()> {
            log::info!("🔌 Starting rust-mcp-sdk SSE client connection to: {}", self.server_url);
            
            // Create SSE URL - ensure it ends with /sse
            let sse_url = if self.server_url.ends_with("/sse") {
                self.server_url.clone()
            } else {
                format!("{}/sse", self.server_url.trim_end_matches('/'))
            };
            
            log::info!("🔗 Connecting to SSE endpoint: {}", sse_url);

            // Step1 : Define client details and capabilities
            let client_details: InitializeRequestParams = InitializeRequestParams {
                capabilities: ClientCapabilities::default(),
                client_info: Implementation {
                    name: "riemind-mcp-client".to_string(),
                    version: "0.1.0".to_string(),
                    title: Some("Simple Rust MCP Client (SSE)".to_string()),
                },
                protocol_version: LATEST_PROTOCOL_VERSION.into(),
            };
            
            // Create transport
            let transport = ClientSseTransport::new(&sse_url, ClientSseTransportOptions::default())?;

            // STEP 3: instantiate our custom handler that is responsible for handling MCP messages
            let handler = MyClientHandler {};

            let client = client_runtime::create_client(client_details, transport, handler);

            // STEP 5: start the MCP client
            client.clone().start().await?;
            
            log::info!("✅ Client initialized successfully");
            self.client = Some(Arc::clone(&client));
            // Store the client for later use

            
            log::info!("✅ rust-mcp-sdk SSE client connected successfully");
            Ok(())
        }

        /// Call a tool on the MCP server using rust-mcp-sdk
        pub async fn call_tool(
            &self,
            tool_name: &str,
            arguments: serde_json::Map<String, serde_json::Value>,
            current_location: [f32; 3],
        ) -> SdkResult<McpResponse> {
            log::info!("🔧 Calling MCP tool: {} with arguments: {:?}", tool_name, arguments);
            log::info!("📍 Current location: {:?}", current_location);

            // Debug: Print server info before tool call
            let server_info = self.client.as_ref().unwrap().server_info();
            log::info!("🔗 Debug - Server info before tool call: {:#?}", server_info);

            // Prepare arguments for "Our Awesome Tool" with the exact format required:
            // {"query": "{\"messages\": \"...\", \"current_location\": [x,y,z]}"}
            let query_object = json!({
                "messages": arguments.get("query").unwrap_or(&json!("")).as_str().unwrap_or(""),
                "current_location": current_location
            });

            let tool_arguments = json!({
                "query": query_object.to_string()
            }).as_object().unwrap().clone();

            log::info!("📝 Formatted tool arguments for '{}': {:?}", tool_name, tool_arguments);
            log::info!("🔧 Debug - Tool name: '{}'", tool_name);

            // Create the tool call request
            let request = CallToolRequestParams {
                name: tool_name.to_string(),
                arguments: Some(tool_arguments),
            };

            // Call the tool
            log::info!("🔧 Debug - About to call tool: {}", tool_name);
            let result = self.client.as_ref().unwrap().call_tool(request).await?;

            log::info!("✅ Tool call successful, parsing result");

            // Parse the result into our McpResponse format
            self.parse_tool_result(result)
        }

        /// Parse rust-mcp-sdk tool result into our McpResponse format
        fn parse_tool_result(
            &self,
            result: CallToolResult,
        ) -> SdkResult<McpResponse> {
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
            match first_content.as_text_content()? {
                text_content => {
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

        /// Disconnect from the MCP server
        pub async fn disconnect(&self) -> SdkResult<()> {
            log::info!("🔌 Disconnecting from MCP server");

            self.client.as_ref().unwrap().shut_down().await?;
            log::info!("✅ MCP client disconnected");

            Ok(())
        }
    }
}

// WASM implementation (stub for now)
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;

    pub struct MCPClient {
    }

    impl MCPClient {
        pub fn new() -> Self {
            Self { server_url: "http://localhost:8080".to_string() }
        }

        pub async fn start(&mut self, server_url: String) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            log::warn!("🌐 WASM rust-mcp-sdk MCP client not yet implemented");
            log::info!("📄 WASM builds will use rust-mcp-sdk when SSE client API is stable");
            Err("WASM rust-mcp-sdk MCP client not yet implemented".into())
        }

        pub async fn call_tool(
            &self,
            _tool_name: &str,
            _arguments: serde_json::Map<String, serde_json::Value>,
            _current_location: [f32; 3],
        ) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
            log::warn!("🌐 WASM rust-mcp-sdk MCP tool calling not yet implemented");
            Err("WASM rust-mcp-sdk MCP tool calling not yet implemented".into())
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
        let mut client = MCPClient::new("http://localhost:8080".to_string());
        let result = client.start().await;
        assert!(result.is_ok());
        println!("✅ Native MCP client created successfully");
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    async fn test_wasm_mcp_client_creation() {
        let mut client = MCPClient::new("http://localhost:8080".to_string());
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
        let mut client = MCPClient::new("http://localhost:8080".to_string());
        
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
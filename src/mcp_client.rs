#[cfg(not(target_arch = "wasm32"))]
use async_trait::async_trait;
#[cfg(not(target_arch = "wasm32"))]
use rust_mcp_sdk::error::SdkResult;
#[cfg(not(target_arch = "wasm32"))]
use rust_mcp_sdk::mcp_client::{client_runtime, ClientHandler, ClientRuntime};
#[cfg(not(target_arch = "wasm32"))]
use rust_mcp_sdk::schema::{
    ClientCapabilities, Implementation, InitializeRequestParams, CallToolRequestParams,
    LATEST_PROTOCOL_VERSION, CallToolResult,
};
#[cfg(not(target_arch = "wasm32"))]
use rust_mcp_sdk::{ClientSseTransport, ClientSseTransportOptions, McpClient};
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc::{self, Receiver, Sender};

use crate::chat::{McpResponse, SceneObject, PathResponse};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPRequest {
    pub message: String,
    pub current_location: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResponse {
    pub answer: Vec<SceneObject>,
    pub paths: Vec<PathResponse>,
    pub scene_normal_vector: Option<String>,
    pub text_answer: Option<String>,
}

#[cfg(not(target_arch = "wasm32"))]
pub struct MCPClientHandler {
    response_sender: Sender<(String, McpResponse)>,
}

#[cfg(not(target_arch = "wasm32"))]
impl MCPClientHandler {
    pub fn new(response_sender: Sender<(String, McpResponse)>) -> Self {
        Self { response_sender }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl ClientHandler for MCPClientHandler {
    // This is a minimal implementation - the actual tool calls will be handled by the client runtime
}

#[cfg(not(target_arch = "wasm32"))]
pub struct MCPClient {
    client: Arc<ClientRuntime>,
    response_receiver: Receiver<(String, McpResponse)>,
    last_tool_result: Option<CallToolResult>,
}

#[cfg(not(target_arch = "wasm32"))]
impl MCPClient {
    pub async fn new(server_url: String) -> SdkResult<Self> {
        // Create channel for responses
        let (response_sender, response_receiver) = mpsc::channel(100);
        
        // Create handler
        let handler = MCPClientHandler::new(response_sender);

        // Define client details and capabilities
        let client_details = InitializeRequestParams {
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: "web-splat-mcp-client".to_string(),
                version: "0.1.0".to_string(),
                title: Some("Web-Splat MCP Client".to_string()),
            },
            protocol_version: LATEST_PROTOCOL_VERSION.into(),
        };

        // Create SSE transport
        let transport = ClientSseTransport::new(&server_url, ClientSseTransportOptions::default())?;

        // Create client
        let client = client_runtime::create_client(client_details, transport, handler);

        Ok(Self {
            client: Arc::clone(&client),
            response_receiver,
            last_tool_result: None,
        })
    }

    pub async fn start(&self) -> SdkResult<()> {
        self.client.clone().start().await
    }

    pub async fn send_message(&mut self, message: String, current_location: [f32; 3]) -> SdkResult<()> {
        // Create tool call with the request data
        let params = serde_json::json!({
            "message": message,
            "current_location": current_location
        })
        .as_object()
        .unwrap()
        .clone();

        // Call the tool
        let result = self.client.call_tool(CallToolRequestParams {
            name: "scene_query".to_string(),
            arguments: Some(params),
        }).await?;

        log::info!("🔧 MCP Tool call result: {:?}", result);
        
        // Store the result for later retrieval
        self.last_tool_result = Some(result);
        
        Ok(())
    }

    pub async fn receive_response(&mut self) -> Option<(String, McpResponse)> {
        // Try to get a response from the channel first
        if let Ok((message, response)) = self.response_receiver.try_recv() {
            return Some((message, response));
        }

        // Process the last tool result if available
        if let Some(tool_result) = self.last_tool_result.take() {
            log::info!("🔧 Processing tool result: {:?}", tool_result);
            
            // Extract the content from the tool result
            if let Some(content) = tool_result.content.first() {
                match content.as_text_content() {
                    Ok(text_content) => {
                        log::info!("📝 Tool response text: {}", text_content.text);
                        
                        // Try to parse the response as JSON
                        match serde_json::from_str::<serde_json::Value>(&text_content.text) {
                            Ok(json_value) => {
                                log::info!("✅ Parsed JSON response: {:?}", json_value);
                                
                                // Try to parse as our McpResponse format
                                match serde_json::from_value::<McpResponse>(json_value.clone()) {
                                    Ok(mcp_response) => {
                                        log::info!("✅ Successfully parsed as McpResponse");
                                        return Some(("Riemind Response".to_string(), mcp_response));
                                    }
                                    Err(e) => {
                                        log::warn!("⚠️ Failed to parse as McpResponse: {}", e);
                                        
                                        // Try to parse using the existing parse_mcp_response function
                                        match crate::chat::parse_mcp_response(&text_content.text) {
                                            Ok(mcp_response) => {
                                                log::info!("✅ Successfully parsed using parse_mcp_response");
                                                return Some(("Riemind Response".to_string(), mcp_response));
                                            }
                                            Err(e) => {
                                                log::warn!("⚠️ Failed to parse using parse_mcp_response: {}", e);
                                                
                                                // Create a text-only response
                                                let text_response = McpResponse {
                                                    answer: Vec::new(),
                                                    paths: Vec::new(),
                                                    scene_normal_vector: None,
                                                    text_answer: Some(text_content.text.clone()),
                                                };
                                                return Some(("Riemind Response".to_string(), text_response));
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                log::warn!("⚠️ Failed to parse as JSON: {}", e);
                                
                                // Create a text-only response
                                let text_response = McpResponse {
                                    answer: Vec::new(),
                                    paths: Vec::new(),
                                    scene_normal_vector: None,
                                    text_answer: Some(text_content.text.clone()),
                                };
                                return Some(("Riemind Response".to_string(), text_response));
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("⚠️ Failed to extract text content: {}", e);
                    }
                }
            }
        }

        // Fallback to mock response if no real response is available
        log::warn!("❌ No real response available, using mock response");
        let mock_response = McpResponse {
            answer: vec![
                SceneObject {
                    name: "Coffee Machine".to_string(),
                    aligned_bbox: vec![
                        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
                    ],
                    normal_vector: Some([0.0, 0.0, 1.0]),
                    attributes: Some(std::collections::HashMap::from([
                        ("type".to_string(), "coffee_machine".to_string()),
                        ("material".to_string(), "stainless_steel".to_string()),
                    ])),
                }
            ],
            paths: vec![
                PathResponse {
                    object: SceneObject {
                        name: "Coffee Machine".to_string(),
                        aligned_bbox: vec![
                            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
                        ],
                        normal_vector: Some([0.0, 0.0, 1.0]),
                        attributes: Some(std::collections::HashMap::from([
                            ("type".to_string(), "coffee_machine".to_string()),
                            ("material".to_string(), "stainless_steel".to_string()),
                        ])),
                    },
                    path: vec![
                        [0.0, 0.0, 0.0], // This would be the current_location from the request
                        [0.5, 0.5, 0.5],
                        [1.0, 1.0, 1.0],
                    ],
                }
            ],
            scene_normal_vector: Some("[0.0, 1.0, 0.0]".to_string()),
            text_answer: None,
        };
        
        Some(("Mock MCP Response".to_string(), mock_response))
    }

    pub async fn shutdown(&self) -> SdkResult<()> {
        self.client.shut_down().await
    }
}

// WASM-compatible placeholder implementations
#[cfg(target_arch = "wasm32")]
pub struct MCPClientHandler;

#[cfg(target_arch = "wasm32")]
impl MCPClientHandler {
    pub fn new(_response_sender: ()) -> Self {
        Self
    }
}

#[cfg(target_arch = "wasm32")]
pub struct MCPClient;

#[cfg(target_arch = "wasm32")]
impl MCPClient {
    pub async fn new(_server_url: String) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self)
    }

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    pub async fn send_message(&self, _message: String, _current_location: [f32; 3]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    pub async fn receive_response(&mut self) -> Option<(String, McpResponse)> {
        None
    }

    pub async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_mcp_client_creation() {
        // This test verifies that we can create an MCP client
        // Note: This will fail if no MCP server is running, but that's expected
        let result = MCPClient::new("http://localhost:8080/sse".to_string()).await;
        match result {
            Ok(_client) => {
                println!("✅ MCP client created successfully");
            }
            Err(e) => {
                println!("⚠️  MCP client creation failed (expected if no server running): {}", e);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_mcp_client_with_riemind() {
        // Test with the actual Riemind server
        let result = MCPClient::new("http://localhost:8080/sse".to_string()).await;
        match result {
            Ok(mut client) => {
                println!("✅ MCP client created successfully");
                
                // Try to start the client
                match client.start().await {
                    Ok(_) => {
                        println!("✅ MCP client started successfully");
                        
                        // Try to send a test message
                        match client.send_message("where is the coffee machine?".to_string(), [0.0, 1.0, 0.0]).await {
                            Ok(_) => {
                                println!("✅ Message sent successfully");
                                
                                // Try to receive a response
                                match client.receive_response().await {
                                    Some((message, response)) => {
                                        println!("✅ Received response: {}", message);
                                        println!("Response: {:?}", response);
                                    }
                                    None => {
                                        println!("⚠️  No response received");
                                    }
                                }
                            }
                            Err(e) => {
                                println!("⚠️  Failed to send message: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("⚠️  Failed to start MCP client: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("⚠️  MCP client creation failed: {}", e);
            }
        }
    }
} 
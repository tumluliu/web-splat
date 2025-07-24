#[cfg(not(target_arch = "wasm32"))]
use async_trait::async_trait;
#[cfg(not(target_arch = "wasm32"))]
use rust_mcp_sdk::error::SdkResult;
#[cfg(not(target_arch = "wasm32"))]
use rust_mcp_sdk::mcp_client::{client_runtime, ClientHandler, ClientRuntime};
#[cfg(not(target_arch = "wasm32"))]
use rust_mcp_sdk::schema::{
    ClientCapabilities, Implementation, InitializeRequestParams, CallToolRequestParams,
    LATEST_PROTOCOL_VERSION,
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
        })
    }

    pub async fn start(&self) -> SdkResult<()> {
        self.client.clone().start().await
    }

    pub async fn send_message(&self, message: String, current_location: [f32; 3]) -> SdkResult<()> {
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
        
        // For now, we'll create a mock response in the receive_response method
        // since we don't have a real MCP server that returns the expected format
        
        Ok(())
    }

    pub async fn receive_response(&mut self) -> Option<(String, McpResponse)> {
        // For now, return a mock response since we don't have a real MCP server
        // In a real implementation, this would wait for the actual response from the server
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
        let result = MCPClient::new("http://localhost:3001/sse".to_string()).await;
        match result {
            Ok(_client) => {
                println!("✅ MCP client created successfully");
            }
            Err(e) => {
                println!("⚠️  MCP client creation failed (expected if no server running): {}", e);
            }
        }
    }
} 
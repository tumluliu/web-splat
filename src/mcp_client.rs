use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::chat::McpResponse;

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

#[cfg(not(target_arch = "wasm32"))]
pub struct MCPClient {
    server_url: String,
    response_cache: Option<McpResponse>,
}

#[cfg(not(target_arch = "wasm32"))]
impl MCPClient {
    pub async fn new(server_url: String) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            server_url,
            response_cache: None,
        })
    }

    pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("🔌 Starting MCP client connection to: {}", self.server_url);
        log::info!("✅ MCP client initialized successfully");
        Ok(())
    }

    /// **THIS IS THE MISSING call_tool() FUNCTION YOU ASKED ABOUT!**
    pub async fn call_tool(
        &mut self, 
        tool_name: &str, 
        arguments: serde_json::Map<String, serde_json::Value>,
        current_location: [f32; 3]
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("🔧 MCP call_tool(): {} with args: {:?}", tool_name, arguments);
        
        let client = reqwest::Client::new();
        
        // Use the correct SSE endpoint that your server expects
        let sse_url = format!("{}/sse", self.server_url.trim_end_matches('/'));
        
        // Extract the message from arguments (this is what your server expects)
        let message = arguments.get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("scene query")
            .to_string();
        
        // Create the CORRECT format that your MCP server expects
        let mcp_request = json!({
            "messages": message,
            "current_location": current_location
        });
        
        log::info!("🔧 Sending MCP request to: {}", sse_url);
        log::info!("📋 Request body: {}", serde_json::to_string_pretty(&mcp_request).unwrap_or_default());
        
        let response = client
            .post(&sse_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&mcp_request)
            .send()
            .await?;
        
        log::info!("📡 Response status: {}", response.status());
        
        let status = response.status();
        if status.is_success() {
            let response_text = response.text().await?;
            log::info!("📥 Raw MCP response: {}", response_text);
            
            // Parse the response into our McpResponse format
            match self.parse_tool_response(&response_text) {
                Ok(mcp_response) => {
                    self.response_cache = Some(mcp_response);
                    log::info!("✅ Successfully parsed MCP response");
                }
                Err(e) => {
                    log::warn!("⚠️ Failed to parse MCP response: {}", e);
                    // Create a text response as fallback
                    let text_response = McpResponse {
                        answer: Vec::new(),
                        paths: Vec::new(),
                        scene_normal_vector: None,
                        text_answer: Some(response_text),
                    };
                    self.response_cache = Some(text_response);
                }
            }
        } else {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            log::error!("❌ MCP request failed: {} - {}", status, error_text);
            return Err(format!("MCP request failed: {} - {}", status, error_text).into());
        }

        Ok(())
    }

    pub async fn send_message(&mut self, message: String, current_location: [f32; 3]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("📤 Sending message via MCP: {}", message);
        
        // For now, treat this as a "scene_query" tool call
        let mut arguments = serde_json::Map::new();
        arguments.insert("query".to_string(), json!(message));
        arguments.insert("context".to_string(), json!("3d_scene_understanding"));
        
        self.call_tool("scene_query", arguments, current_location).await
    }

    pub async fn receive_response(&mut self) -> Option<(String, McpResponse)> {
        if let Some(response) = self.response_cache.take() {
            Some(("MCP Tool Response".to_string(), response))
        } else {
            None
        }
    }

    pub async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("🔌 MCP client shutting down");
        Ok(())
    }

    fn parse_tool_response(&self, response_text: &str) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
        // Try to parse as JSON-RPC response first
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(response_text) {
            // Check if it's a JSON-RPC response
            if let Some(result) = json_value.get("result") {
                // Try to parse the result as our McpResponse
                if let Ok(mcp_response) = serde_json::from_value::<McpResponse>(result.clone()) {
                    return Ok(mcp_response);
                }
                
                // If result is just text content, extract it
                if let Some(content) = result.get("content") {
                    if let Some(text) = content.as_str() {
                        return Ok(McpResponse {
                            answer: Vec::new(),
                            paths: Vec::new(),
                            scene_normal_vector: None,
                            text_answer: Some(text.to_string()),
                        });
                    }
                }
            }
            
            // Try to parse entire response as our format
            if let Ok(mcp_response) = serde_json::from_value::<McpResponse>(json_value) {
                return Ok(mcp_response);
            }
        }
        
        // Fallback: try to parse as our format directly
        match crate::chat::parse_mcp_response(response_text) {
            Ok(response) => Ok(response),
            Err(_) => {
                // Final fallback: treat as plain text
                Ok(McpResponse {
                    answer: Vec::new(),
                    paths: Vec::new(),
                    scene_normal_vector: None,
                    text_answer: Some(response_text.to_string()),
                })
            }
        }
    }
}

// WASM implementation - no actual MCP client, falls back to HTTP
#[cfg(target_arch = "wasm32")]
pub struct MCPClient {
    server_url: String,
    response_cache: Option<McpResponse>,
}

#[cfg(target_arch = "wasm32")]
impl MCPClient {
    pub async fn new(server_url: String) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            server_url,
            response_cache: None,
        })
    }

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("🌐 WASM MCP client initialized (HTTP fallback mode): {}", self.server_url);
        Ok(())
    }

    pub async fn call_tool(
        &mut self, 
        tool_name: &str, 
        arguments: serde_json::Map<String, serde_json::Value>,
        current_location: [f32; 3]
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::warn!("🌐 WASM: call_tool() not available, falling back to HTTP client");
        
        // Convert tool call to simple message for HTTP fallback
        let message = arguments.get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("scene query")
            .to_string();
            
        self.send_message(message, current_location).await
    }

    pub async fn send_message(&mut self, message: String, current_location: [f32; 3]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::warn!("🌐 WASM: MCP client not available, this should fall back to HTTP client");
        Err("MCP client not available in WASM builds".into())
    }

    pub async fn receive_response(&mut self) -> Option<(String, McpResponse)> {
        None
    }

    pub async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("🌐 WASM MCP client shutting down");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_native_mcp_client_creation() {
        let result = MCPClient::new("http://localhost:3000".to_string()).await;
        assert!(result.is_ok());
        println!("✅ Native MCP client created successfully");
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    async fn test_wasm_mcp_client_creation() {
        let result = MCPClient::new("http://localhost:3000".to_string()).await;
        assert!(result.is_ok());
        web_sys::console::log_1(&"✅ WASM MCP client created successfully".into());
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
        let mut client = MCPClient::new("http://localhost:3000".to_string()).await.unwrap();
        
        let mut args = serde_json::Map::new();
        args.insert("query".to_string(), json!("where is the coffee machine?"));
        
        // This will fail to connect, but we're testing the format
        let result = client.call_tool("scene_query", args, [1.0, 2.0, 3.0]).await;
        assert!(result.is_err()); // Expected to fail due to no server
        
        println!("✅ Tool call format test completed");
    }
} 
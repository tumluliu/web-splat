use serde::{Deserialize, Serialize};

use crate::chat::McpResponse;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPRequest {
    pub message: String,
    pub current_location: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEEvent {
    pub data: String,
    #[serde(default)]
    pub event: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub retry: Option<u32>,
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

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("🔌 Native MCP client connected to: {}", self.server_url);
        Ok(())
    }

    pub async fn send_message(&mut self, message: String, current_location: [f32; 3]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("📤 Sending message via SSE: {}", message);
        
        let request_body = serde_json::json!({
            "message": message,
            "current_location": current_location
        });

        // For native builds, use reqwest to connect to SSE endpoint
        let client = reqwest::Client::new();
        
        let response = client
            .post(&format!("{}/query", self.server_url))
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .json(&request_body)
            .send()
            .await?;

        if response.status().is_success() {
            // Read the response as SSE stream
            let response_text = response.text().await?;
            log::info!("📥 Raw SSE response: {}", response_text);
            
            // Parse SSE events
            let events = self.parse_sse_events(&response_text);
            log::info!("🔍 Parsed {} SSE events", events.len());
            
            // Process the last data event
            for event in events {
                if let Some(data) = event.data.strip_prefix("data: ") {
                    match crate::chat::parse_mcp_response(data) {
                        Ok(mcp_response) => {
                            self.response_cache = Some(mcp_response);
                            log::info!("✅ Successfully parsed MCP response from SSE");
                            return Ok(());
                        }
                        Err(e) => {
                            log::warn!("⚠️ Failed to parse MCP response: {}", e);
                        }
                    }
                }
            }
        }

        // Fallback: try direct JSON response
        let direct_response = client
            .post(&format!("{}/query", self.server_url))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if direct_response.status().is_success() {
            let response_text = direct_response.text().await?;
            match crate::chat::parse_mcp_response(&response_text) {
                Ok(mcp_response) => {
                    self.response_cache = Some(mcp_response);
                    log::info!("✅ Successfully parsed MCP response from direct JSON");
                }
                Err(e) => {
                    log::warn!("⚠️ Failed to parse direct MCP response: {}", e);
                }
            }
        }

        Ok(())
    }

    pub async fn receive_response(&mut self) -> Option<(String, McpResponse)> {
        if let Some(response) = self.response_cache.take() {
            Some(("SSE Response".to_string(), response))
        } else {
            None
        }
    }

    pub async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("🔌 Native MCP client shutting down");
        Ok(())
    }

    fn parse_sse_events(&self, text: &str) -> Vec<SSEEvent> {
        let mut events = Vec::new();
        let mut current_event = SSEEvent {
            data: String::new(),
            event: None,
            id: None,
            retry: None,
        };
        
        for line in text.lines() {
            let line = line.trim();
            
            if line.is_empty() {
                // Empty line marks end of event
                if !current_event.data.is_empty() || current_event.event.is_some() {
                    events.push(current_event.clone());
                }
                current_event = SSEEvent {
                    data: String::new(),
                    event: None,
                    id: None,
                    retry: None,
                };
                continue;
            }
            
            if let Some(colon_pos) = line.find(':') {
                let field = &line[..colon_pos].trim();
                let value = line[colon_pos + 1..].trim();
                
                match *field {
                    "data" => {
                        if !current_event.data.is_empty() {
                            current_event.data.push('\n');
                        }
                        current_event.data.push_str(value);
                    }
                    "event" => current_event.event = Some(value.to_string()),
                    "id" => current_event.id = Some(value.to_string()),
                    "retry" => {
                        if let Ok(retry_val) = value.parse::<u32>() {
                            current_event.retry = Some(retry_val);
                        }
                    }
                    _ => {} // Ignore unknown fields
                }
            }
        }
        
        // Don't forget the last event if there's no trailing empty line
        if !current_event.data.is_empty() || current_event.event.is_some() {
            events.push(current_event);
        }
        
        events
    }
}

// WASM implementation using browser's EventSource API
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
        log::info!("🌐 WASM MCP client initialized for: {}", self.server_url);
        Ok(())
    }

    pub async fn send_message(&mut self, message: String, current_location: [f32; 3]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use wasm_bindgen::prelude::*;
        use wasm_bindgen_futures::JsFuture;
        use web_sys::{Request, RequestInit, RequestMode, Response};

        log::info!("📤 WASM: Sending message: {}", message);

        let request_body = serde_json::json!({
            "message": message,
            "current_location": current_location
        });

        // First try SSE approach using fetch with EventSource-like behavior
        let url = format!("{}/query", self.server_url);
        log::info!("🌐 Making WASM request to: {}", url);

        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_mode(RequestMode::Cors);

        // Set headers
        let headers = web_sys::Headers::new()
            .map_err(|e| format!("Failed to create headers: {:?}", e))?;
        headers
            .set("Content-Type", "application/json")
            .map_err(|e| format!("Failed to set content-type: {:?}", e))?;
        headers
            .set("Accept", "text/event-stream, application/json")
            .map_err(|e| format!("Failed to set accept header: {:?}", e))?;
        opts.set_headers(&headers);

        // Set body
        let body_string = serde_json::to_string(&request_body)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;
        opts.set_body(&JsValue::from_str(&body_string));

        // Create request
        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| format!("Failed to create request: {:?}", e))?;

        // Get window and make fetch request
        let window = web_sys::window().ok_or("No global window object")?;
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| format!("Fetch failed: {:?}", e))?;

        // Cast to Response
        let resp: Response = resp_value
            .dyn_into()
            .map_err(|_| "Response is not a Response object")?;

        log::info!("📡 WASM Response status: {}", resp.status());

        if resp.ok() {
            // Get response text
            let text_promise = resp
                .text()
                .map_err(|e| format!("Failed to get response text promise: {:?}", e))?;
            let text_value = JsFuture::from(text_promise)
                .await
                .map_err(|e| format!("Failed to get response text: {:?}", e))?;

            let response_text = text_value
                .as_string()
                .ok_or("Response text is not a string")?;

            log::info!("📝 WASM Raw response: {}", response_text);

            // Try to parse as MCP response directly
            match crate::chat::parse_mcp_response(&response_text) {
                Ok(mcp_response) => {
                    self.response_cache = Some(mcp_response);
                    log::info!("✅ WASM: Successfully parsed MCP response");
                    return Ok(());
                }
                Err(e) => {
                    log::warn!("⚠️ WASM: Failed to parse response as MCP: {}", e);
                    
                    // Try parsing as SSE events
                    let events = self.parse_sse_events(&response_text);
                    for event in events {
                        if let Some(data) = event.data.strip_prefix("data: ") {
                            match crate::chat::parse_mcp_response(data) {
                                Ok(mcp_response) => {
                                    self.response_cache = Some(mcp_response);
                                    log::info!("✅ WASM: Successfully parsed MCP response from SSE");
                                    return Ok(());
                                }
                                Err(e) => {
                                    log::warn!("⚠️ WASM: Failed to parse SSE data: {}", e);
                                }
                            }
                        }
                    }
                    
                    // Create a text-only response as fallback
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
            let error_text = if let Ok(text_promise) = resp.text() {
                match JsFuture::from(text_promise).await {
                    Ok(text_value) => text_value
                        .as_string()
                        .unwrap_or_else(|| "Unknown error".to_string()),
                    Err(_) => "Failed to read error text".to_string(),
                }
            } else {
                "Unknown error".to_string()
            };

            log::warn!("❌ WASM Server error: {} - {}", resp.status(), error_text);
            
            // Create an error response
            let error_response = McpResponse {
                answer: Vec::new(),
                paths: Vec::new(),
                scene_normal_vector: None,
                text_answer: Some(format!("Server error: {}", error_text)),
            };
            self.response_cache = Some(error_response);
        }

        Ok(())
    }

    pub async fn receive_response(&mut self) -> Option<(String, McpResponse)> {
        if let Some(response) = self.response_cache.take() {
            Some(("WASM SSE Response".to_string(), response))
        } else {
            None
        }
    }

    pub async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("🌐 WASM MCP client shutting down");
        Ok(())
    }

    fn parse_sse_events(&self, text: &str) -> Vec<SSEEvent> {
        let mut events = Vec::new();
        let mut current_event = SSEEvent {
            data: String::new(),
            event: None,
            id: None,
            retry: None,
        };
        
        for line in text.lines() {
            let line = line.trim();
            
            if line.is_empty() {
                // Empty line marks end of event
                if !current_event.data.is_empty() || current_event.event.is_some() {
                    events.push(current_event.clone());
                }
                current_event = SSEEvent {
                    data: String::new(),
                    event: None,
                    id: None,
                    retry: None,
                };
                continue;
            }
            
            if let Some(colon_pos) = line.find(':') {
                let field = &line[..colon_pos].trim();
                let value = line[colon_pos + 1..].trim();
                
                match *field {
                    "data" => {
                        if !current_event.data.is_empty() {
                            current_event.data.push('\n');
                        }
                        current_event.data.push_str(value);
                    }
                    "event" => current_event.event = Some(value.to_string()),
                    "id" => current_event.id = Some(value.to_string()),
                    "retry" => {
                        if let Ok(retry_val) = value.parse::<u32>() {
                            current_event.retry = Some(retry_val);
                        }
                    }
                    _ => {} // Ignore unknown fields
                }
            }
        }
        
        // Don't forget the last event if there's no trailing empty line
        if !current_event.data.is_empty() || current_event.event.is_some() {
            events.push(current_event);
        }
        
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_native_mcp_client_creation() {
        let result = MCPClient::new("http://localhost:8080".to_string()).await;
        assert!(result.is_ok());
        println!("✅ Native MCP client created successfully");
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    async fn test_wasm_mcp_client_creation() {
        let result = MCPClient::new("http://localhost:8080".to_string()).await;
        assert!(result.is_ok());
        web_sys::console::log_1(&"✅ WASM MCP client created successfully".into());
    }

    #[test]
    fn test_sse_event_parsing() {
        let client = MCPClient {
            server_url: "test".to_string(),
            response_cache: None,
        };

        let sse_text = "data: test message\nevent: message\nid: 123\n\ndata: second message\n\n";
        let events = client.parse_sse_events(sse_text);
        
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "test message");
        assert_eq!(events[0].event, Some("message".to_string()));
        assert_eq!(events[0].id, Some("123".to_string()));
        assert_eq!(events[1].data, "second message");
    }
} 
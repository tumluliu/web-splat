use serde::{Deserialize, Serialize};

use crate::chat::McpResponse;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPRequest {
    pub messages: String,
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
        log::info!("📤 Sending message via proper SSE: {}", message);
        
        let client = reqwest::Client::new();
        
        // Use SSE endpoint with query parameters (standard SSE)
        let sse_url = if self.server_url.ends_with("/sse") {
            self.server_url.clone()
        } else {
            format!("{}/sse", self.server_url.trim_end_matches('/'))
        };
        
        // Encode parameters in URL for GET request (standard SSE)
        let encoded_message = message.replace(" ", "%20").replace("?", "%3F").replace("&", "%26");
        let url_with_params = format!(
            "{}?messages={}&current_location={},{},{}",
            sse_url,
            encoded_message,
            current_location[0],
            current_location[1],
            current_location[2]
        );
        
        log::info!("🔌 Connecting to SSE endpoint (GET): {}", url_with_params);
        
        let response = client
            .get(&url_with_params)  // Use GET instead of POST
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache") 
            .header("Connection", "keep-alive")
            .send()
            .await?;

        let status = response.status();
        if status.is_success() {
            // Read the response as SSE stream
            let response_text = response.text().await?;
            log::info!("📥 Raw SSE response: {}", response_text);
            
            // Parse SSE events
            let events = self.parse_sse_events(&response_text);
            log::info!("🔍 Parsed {} SSE events", events.len());
            
            // Process SSE events looking for MCP data
            for event in events {
                // Handle different SSE data formats
                let data_to_parse = if event.data.starts_with("data: ") {
                    &event.data[6..] // Remove "data: " prefix
                } else {
                    &event.data
                };
                
                if !data_to_parse.trim().is_empty() && data_to_parse != "[DONE]" {
                    match crate::chat::parse_mcp_response(data_to_parse) {
                        Ok(mcp_response) => {
                            self.response_cache = Some(mcp_response);
                            log::info!("✅ Successfully parsed MCP response from SSE");
                            return Ok(());
                        }
                        Err(e) => {
                            log::debug!("🔍 Couldn't parse as MCP response: {} - Data: {}", e, data_to_parse);
                        }
                    }
                }
            }
            
            // If no structured data found, create a text response
            if !response_text.trim().is_empty() {
                let text_response = crate::chat::McpResponse {
                    answer: Vec::new(),
                    paths: Vec::new(),
                    scene_normal_vector: None,
                    text_answer: Some("Received SSE response but couldn't parse as structured data".to_string()),
                };
                self.response_cache = Some(text_response);
                log::info!("📝 Created text response from SSE data");
            }
        } else {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            log::error!("❌ SSE request failed: {} - {}", status, error_text);
            return Err(format!("SSE request failed: {} - {}", status, error_text).into());
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

// WASM implementation
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

        log::info!("📤 WASM: Sending message via proper SSE: {}", message);

        // Use SSE endpoint with query parameters (standard SSE)
        let sse_url = if self.server_url.ends_with("/sse") {
            self.server_url.clone()
        } else {
            format!("{}/sse", self.server_url.trim_end_matches('/'))
        };
        
        // Encode parameters in URL for GET request (standard SSE)
        let encoded_message = message.replace(" ", "%20").replace("?", "%3F").replace("&", "%26");
        let url_with_params = format!(
            "{}?messages={}&current_location={},{},{}",
            sse_url,
            encoded_message,
            current_location[0],
            current_location[1],
            current_location[2]
        );
        
        log::info!("🌐 WASM: Connecting to SSE endpoint (GET): {}", url_with_params);

        let opts = RequestInit::new();
        opts.set_method("GET");  // Use GET instead of POST
        opts.set_mode(RequestMode::Cors);

        // Set headers for SSE
        let headers = web_sys::Headers::new()
            .map_err(|e| format!("Failed to create headers: {:?}", e))?;
        headers
            .set("Accept", "text/event-stream")
            .map_err(|e| format!("Failed to set accept header: {:?}", e))?;
        headers
            .set("Cache-Control", "no-cache")
            .map_err(|e| format!("Failed to set cache-control: {:?}", e))?;
        headers
            .set("Connection", "keep-alive")
            .map_err(|e| format!("Failed to set connection: {:?}", e))?;
        opts.set_headers(&headers);

        // No body needed for GET request

        // Create request
        let request = Request::new_with_str_and_init(&url_with_params, &opts)
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

            log::info!("📝 WASM Raw SSE response: {}", response_text);

            // Parse SSE events looking for MCP data
            let events = self.parse_sse_events(&response_text);
            log::info!("🔍 WASM: Parsed {} SSE events", events.len());
            
            for event in events {
                // Handle different SSE data formats
                let data_to_parse = if event.data.starts_with("data: ") {
                    &event.data[6..] // Remove "data: " prefix
                } else {
                    &event.data
                };
                
                if !data_to_parse.trim().is_empty() && data_to_parse != "[DONE]" {
                    match crate::chat::parse_mcp_response(data_to_parse) {
                        Ok(mcp_response) => {
                            self.response_cache = Some(mcp_response);
                            log::info!("✅ WASM: Successfully parsed MCP response from SSE");
                            return Ok(());
                        }
                        Err(e) => {
                            log::debug!("🔍 WASM: Couldn't parse as MCP response: {} - Data: {}", e, data_to_parse);
                        }
                    }
                }
            }
            
            // If no structured data found, create a text response
            if !response_text.trim().is_empty() {
                let text_response = crate::chat::McpResponse {
                    answer: Vec::new(),
                    paths: Vec::new(),
                    scene_normal_vector: None,
                    text_answer: Some("Received SSE response but couldn't parse as structured data".to_string()),
                };
                self.response_cache = Some(text_response);
                log::info!("📝 WASM: Created text response from SSE data");
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

            log::error!("❌ WASM SSE request failed: {} - {}", resp.status(), error_text);
            return Err(format!("SSE request failed: {} - {}", resp.status(), error_text).into());
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
    fn test_mcp_request_format() {
        let request = MCPRequest {
            messages: "where is the coffee machine?".to_string(),
            current_location: [1.0, 2.0, 3.0],
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"messages\":"));
        assert!(json.contains("\"current_location\":"));
    }
} 
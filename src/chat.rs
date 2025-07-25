use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;
#[cfg(target_arch = "wasm32")]
use web_time::SystemTime;

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub content: String,
    pub is_user: bool,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SceneObject {
    pub name: String,
    pub aligned_bbox: Vec<[f32; 3]>, // 8 3D points representing the aligned bounding box
    #[serde(default)]
    pub normal_vector: Option<[f32; 3]>, // Semantic front face normal vector
    #[serde(default)]
    pub attributes: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScenePath {
    pub waypoints: Vec<[f32; 3]>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PathResponse {
    pub object: SceneObject,
    pub path: Vec<[f32; 3]>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpResponse {
    #[serde(default)]
    pub answer: Vec<SceneObject>,
    #[serde(default)]
    pub paths: Vec<PathResponse>,
    #[serde(default)]
    pub scene_normal_vector: Option<String>, // Format: "[x,y,z]" string from MCP server
    #[serde(default)]
    pub text_answer: Option<String>, // For simple text/number responses like "6" or "There are 3 chairs"
}

#[derive(Debug, Clone, PartialEq)]
pub enum MCPConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ChatState {
    pub messages: VecDeque<ChatMessage>,
    pub current_input: String,
    pub is_sending: bool,
    pub highlighted_objects: Vec<SceneObject>,
    pub highlighted_path: Option<ScenePath>,
    pub mcp_server_url: String,
    pub font_size: f32,
    pub mcp_connection_status: MCPConnectionStatus,
    pub last_connection_attempt: Option<SystemTime>,
    pub use_mcp_client: bool, // Whether to use MCP client or fallback to HTTP
    #[cfg(target_arch = "wasm32")]
    pub pending_request_id: Option<String>,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            messages: VecDeque::new(),
            current_input: String::new(),
            is_sending: false,
            highlighted_objects: Vec::new(),
            highlighted_path: None,
            mcp_server_url: "http://localhost:8080".to_string(),
            font_size: 14.0, // Default font size
            mcp_connection_status: MCPConnectionStatus::Disconnected,
            last_connection_attempt: None,
            use_mcp_client: true, // Default to trying MCP client first
            #[cfg(target_arch = "wasm32")]
            pending_request_id: None,
        }
    }
}

impl ChatState {
    pub fn add_message(&mut self, content: String, is_user: bool) {
        let timestamp = SystemTime::now();

        self.messages.push_back(ChatMessage {
            content,
            is_user,
            timestamp,
        });

        // Keep only the last 100 messages to prevent memory bloat
        if self.messages.len() > 100 {
            self.messages.pop_front();
        }
    }

    pub fn clear_highlights(&mut self) {
        self.highlighted_objects.clear();
        self.highlighted_path = None;
    }

    pub fn set_highlights(&mut self, response: McpResponse) {
        self.clear_highlights();
        self.highlighted_objects = response.answer;

        // If we have paths, convert the first path to ScenePath format
        if !response.paths.is_empty() {
            let path_response = &response.paths[0];
            self.highlighted_path = Some(ScenePath {
                waypoints: path_response.path.clone(),
                description: Some(format!("Path to {}", path_response.object.name)),
            });

            // Also highlight the target object
            self.highlighted_objects.push(path_response.object.clone());

            log::info!(
                "Set highlighted path with {} waypoints to object: {}",
                path_response.path.len(),
                path_response.object.name
            );
        }
    }

    pub fn set_connection_status(&mut self, status: MCPConnectionStatus) {
        if self.mcp_connection_status != status {
            log::info!("🔗 MCP connection status changed: {:?} -> {:?}", self.mcp_connection_status, status);
            self.mcp_connection_status = status;
            
            match &self.mcp_connection_status {
                MCPConnectionStatus::Connected => {
                    self.add_message("✅ Connected to MCP server".to_string(), false);
                }
                MCPConnectionStatus::Error(err) => {
                    self.add_message(format!("❌ MCP connection error: {}", err), false);
                }
                _ => {}
            }
        }
    }

    pub fn can_retry_connection(&self) -> bool {
        match &self.mcp_connection_status {
            MCPConnectionStatus::Disconnected | MCPConnectionStatus::Error(_) => {
                if let Some(last_attempt) = self.last_connection_attempt {
                    // Allow retry after 5 seconds
                    last_attempt.elapsed().unwrap_or_default().as_secs() >= 5
                } else {
                    true
                }
            }
            _ => false,
        }
    }

    pub fn mark_connection_attempt(&mut self) {
        self.last_connection_attempt = Some(SystemTime::now());
        self.set_connection_status(MCPConnectionStatus::Connecting);
    }

    pub fn get_connection_status_text(&self) -> String {
        match &self.mcp_connection_status {
            MCPConnectionStatus::Disconnected => "⚪ Disconnected".to_string(),
            MCPConnectionStatus::Connecting => "🟡 Connecting...".to_string(),
            MCPConnectionStatus::Connected => "🟢 Connected".to_string(),
            MCPConnectionStatus::Error(err) => format!("🔴 Error: {}", err),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn send_chat_message(
    message: String,
    server_url: &str,
    current_location: [f32; 3],
) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
    log::info!("🔥 send_chat_message called");
    log::info!("📍 Server URL: {}", server_url);
    log::info!("💬 Message: {}", message);
    log::info!(
        "📍 Current camera location: [{:.3}, {:.3}, {:.3}]",
        current_location[0],
        current_location[1],
        current_location[2]
    );

    let client = reqwest::Client::new();

    let request_body = serde_json::json!({
        "messages": message,
        "context": "3d_scene_understanding",
        "current_location": current_location
    });

    log::info!(
        "📦 Request body: {}",
        serde_json::to_string_pretty(&request_body).unwrap_or_else(|_| "Invalid JSON".to_string())
    );

    let url = format!("{}/query", server_url);
    log::info!("🌐 Making POST request to: {}", url);

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await?;

    log::info!("📡 Response status: {}", response.status());
    log::info!("📋 Response headers: {:?}", response.headers());

    let status = response.status();
    if status.is_success() {
        let response_text = response.text().await?;
        log::info!("📝 Raw response: {}", response_text);

        let mcp_response = parse_mcp_response(&response_text)?;
        log::info!("✅ Successfully parsed MCP response");
        Ok(mcp_response)
    } else {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        log::warn!("❌ Server error: {} - {}", status, error_text);
        Ok(McpResponse {
            answer: Vec::new(),
            paths: Vec::new(),
            scene_normal_vector: None,
            text_answer: None,
        })
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn send_chat_message(
    message: String,
    server_url: &str,
    current_location: [f32; 3],
) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    log::info!("🔥 send_chat_message called (WASM version)");
    log::info!("📍 Server URL: {}", server_url);
    log::info!("💬 Message: {}", message);
    log::info!(
        "📍 Current camera location: [{:.3}, {:.3}, {:.3}]",
        current_location[0],
        current_location[1],
        current_location[2]
    );

    let request_body = serde_json::json!({
        "messages": message,
        "context": "3d_scene_understanding",
        "current_location": current_location
    });

    log::info!(
        "📦 Request body: {}",
        serde_json::to_string_pretty(&request_body).unwrap_or_else(|_| "Invalid JSON".to_string())
    );

    let url = format!("{}/query", server_url);
    log::info!("🌐 Making POST request to: {}", url);

    // Create request options
    let opts = RequestInit::new();
    opts.set_method("POST");
    opts.set_mode(RequestMode::Cors);

    // Set headers
    let headers =
        web_sys::Headers::new().map_err(|e| format!("Failed to create headers: {:?}", e))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| format!("Failed to set content-type: {:?}", e))?;
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

    log::info!("📡 Response status: {}", resp.status());

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

        log::info!("📝 Raw response: {}", response_text);

        let mcp_response = parse_mcp_response(&response_text)
            .map_err(|e| format!("Failed to parse JSON response: {}", e))?;

        log::info!("✅ Successfully parsed MCP response");
        Ok(mcp_response)
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

        log::warn!("❌ Server error: {} - {}", resp.status(), error_text);
        Ok(McpResponse {
            answer: Vec::new(),
            paths: Vec::new(),
            scene_normal_vector: None,
            text_answer: None,
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn send_chat_message_mcp(
    message: String,
    server_url: &str,
    current_location: [f32; 3],
) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
    use crate::mcp_client::MCPClient;
    
    log::info!("🔥 send_chat_message_mcp called");
    log::info!("📍 Server URL: {}", server_url);
    log::info!("💬 Message: {}", message);
    log::info!(
        "📍 Current camera location: [{:.3}, {:.3}, {:.3}]",
        current_location[0],
        current_location[1],
        current_location[2]
    );

    // Use the server URL directly (remove /sse suffix if present for our implementation)
    let base_url = server_url.trim_end_matches("/sse").to_string();
    log::info!("🔄 Using base URL: {}", base_url);

    // Create MCP client
    let mut mcp_client = MCPClient::new(base_url).await
        .map_err(|e| format!("Failed to create MCP client: {}", e))?;

    // Start the client
    mcp_client.start().await
        .map_err(|e| format!("Failed to start MCP client: {}", e))?;

    // Send the message
    mcp_client.send_message(message.clone(), current_location).await
        .map_err(|e| format!("Failed to send message: {}", e))?;

    // Wait for response
    match mcp_client.receive_response().await {
        Some((_original_message, response)) => {
            log::info!("✅ Successfully received MCP response");
            Ok(response)
        }
        None => {
            log::warn!("❌ No response received from MCP server");
            Ok(McpResponse {
                answer: Vec::new(),
                paths: Vec::new(),
                scene_normal_vector: None,
                text_answer: Some("No response from MCP server".to_string()),
            })
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn send_chat_message_mcp(
    message: String,
    server_url: &str,
    current_location: [f32; 3],
) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
    use crate::mcp_client::MCPClient;
    
    log::info!("🔥 WASM send_chat_message_mcp called");
    log::info!("📍 Server URL: {}", server_url);
    log::info!("💬 Message: {}", message);
    log::info!(
        "📍 Current camera location: [{:.3}, {:.3}, {:.3}]",
        current_location[0],
        current_location[1],
        current_location[2]
    );

    // Use the server URL directly (remove /sse suffix if present for our implementation)
    let base_url = server_url.trim_end_matches("/sse").to_string();
    log::info!("🔄 WASM: Using base URL: {}", base_url);

    // Create MCP client
    let mut mcp_client = MCPClient::new(base_url).await
        .map_err(|e| format!("Failed to create WASM MCP client: {}", e))?;

    // Start the client
    mcp_client.start().await
        .map_err(|e| format!("Failed to start WASM MCP client: {}", e))?;

    // Send the message
    mcp_client.send_message(message.clone(), current_location).await
        .map_err(|e| format!("Failed to send WASM message: {}", e))?;

    // Wait for response
    match mcp_client.receive_response().await {
        Some((_original_message, response)) => {
            log::info!("✅ WASM: Successfully received MCP response");
            Ok(response)
        }
        None => {
            log::warn!("❌ WASM: No response received from MCP server");
            Ok(McpResponse {
                answer: Vec::new(),
                paths: Vec::new(),
                scene_normal_vector: None,
                text_answer: Some("No response from MCP server (WASM)".to_string()),
            })
        }
    }
}

/// Enhanced async chat message handler that properly uses MCP client
#[cfg(not(target_arch = "wasm32"))]
pub async fn send_chat_message_enhanced(
    message: String,
    server_url: &str,
    current_location: [f32; 3],
    use_mcp_client: bool,
) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
    log::info!("🚀 Enhanced chat message handler called");
    log::info!("📍 Server URL: {}", server_url);
    log::info!("💬 Message: {}", message);
    log::info!("🔗 Use MCP client: {}", use_mcp_client);
    log::info!(
        "📍 Current location: [{:.3}, {:.3}, {:.3}]",
        current_location[0], current_location[1], current_location[2]
    );

    if use_mcp_client {
        log::info!("🔧 Attempting MCP client connection...");
        
        // Try our new MCP client implementation
        match send_chat_message_mcp(message.clone(), server_url, current_location).await {
            Ok(response) => {
                log::info!("✅ MCP client succeeded");
                return Ok(response);
            }
            Err(e) => {
                log::warn!("⚠️ MCP client failed: {}, falling back to HTTP", e);
                // Fall back to HTTP
                match send_chat_message(message, server_url, current_location).await {
                    Ok(response) => {
                        log::info!("✅ HTTP fallback succeeded");
                        return Ok(response);
                    }
                    Err(http_err) => {
                        log::error!("❌ Both MCP and HTTP failed. MCP: {}, HTTP: {}", e, http_err);
                        return Err(format!("Both MCP and HTTP failed. MCP: {}, HTTP: {}", e, http_err).into());
                    }
                }
            }
        }
    } else {
        log::info!("🌐 Using HTTP client directly");
        send_chat_message(message, server_url, current_location).await
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn send_chat_message_enhanced(
    message: String,
    server_url: &str,
    current_location: [f32; 3],
    use_mcp_client: bool,
) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
    log::info!("🌐 Enhanced WASM chat message handler called");
    log::info!("📍 Server URL: {}", server_url);
    log::info!("💬 Message: {}", message);
    log::info!("🔗 Use MCP client: {}", use_mcp_client);
    log::info!(
        "📍 Current location: [{:.3}, {:.3}, {:.3}]",
        current_location[0], current_location[1], current_location[2]
    );

    if use_mcp_client {
        log::info!("🔧 Attempting WASM MCP client...");
        
        // Try our new WASM-compatible MCP client implementation
        match send_chat_message_mcp(message.clone(), server_url, current_location).await {
            Ok(response) => {
                log::info!("✅ WASM MCP client succeeded");
                return Ok(response);
            }
            Err(e) => {
                log::warn!("⚠️ WASM MCP client failed: {}, falling back to HTTP", e);
                // Fall back to HTTP
                match send_chat_message(message, server_url, current_location).await {
                    Ok(response) => {
                        log::info!("✅ WASM HTTP fallback succeeded");
                        return Ok(response);
                    }
                    Err(http_err) => {
                        log::error!("❌ Both WASM MCP and HTTP failed. MCP: {}, HTTP: {}", e, http_err);
                        return Err(format!("Both MCP and HTTP failed. MCP: {}, HTTP: {}", e, http_err).into());
                    }
                }
            }
        }
    } else {
        log::info!("🌐 Using WASM HTTP client directly");
        send_chat_message(message, server_url, current_location).await
    }
}

/// Helper function to parse MCP response that handles multiple formats
pub fn parse_mcp_response(
    response_text: &str,
) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
    #[derive(Debug, Clone, Deserialize)]
    struct RawMcpResponse {
        answer: serde_json::Value,
        #[serde(default)]
        scene_normal_vector: Option<String>,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct ObjectsWrapper {
        objects: Vec<SceneObject>,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct PathsWrapper {
        paths: Vec<PathResponse>,
    }

    // First, parse the outer JSON structure
    let raw_response: RawMcpResponse = serde_json::from_str(response_text)?;
    log::info!("Raw answer value: {:?}", raw_response.answer);
    log::info!(
        "Scene normal vector: {:?}",
        raw_response.scene_normal_vector
    );

    let (answer, paths, text_answer) = match raw_response.answer {
        // Case 1: answer is already a JSON array (legacy object format)
        serde_json::Value::Array(arr) => {
            log::info!("Answer is direct JSON array with {} items", arr.len());
            let objects =
                serde_json::from_value::<Vec<SceneObject>>(serde_json::Value::Array(arr))?;
            (objects, Vec::new(), None)
        }
        // Case 2: answer is a simple string or number (e.g., "6", "There are 3 chairs")
        serde_json::Value::String(simple_string) => {
            // First check if it's a parseable JSON string containing objects/paths
            if simple_string.trim().starts_with('[') || simple_string.trim().starts_with('{') {
                log::info!("Answer is JSON string, parsing it: {}", simple_string);

                // Try to parse as paths wrapper first (new navigation format)
                if let Ok(paths_wrapper) = serde_json::from_str::<PathsWrapper>(&simple_string) {
                    log::info!(
                        "Parsed as paths wrapper with {} paths",
                        paths_wrapper.paths.len()
                    );
                    (Vec::new(), paths_wrapper.paths, None)
                }
                // Try to parse as array first (old object format)
                else if let Ok(objects) = serde_json::from_str::<Vec<SceneObject>>(&simple_string)
                {
                    log::info!("Parsed as direct object array with {} items", objects.len());
                    (objects, Vec::new(), None)
                }
                // Try to parse as object with "objects" key (new object format)
                else if let Ok(wrapper) = serde_json::from_str::<ObjectsWrapper>(&simple_string) {
                    log::info!(
                        "Parsed as objects wrapper with {} items",
                        wrapper.objects.len()
                    );
                    (wrapper.objects, Vec::new(), None)
                }
                // Failed to parse as structured data, treat as simple text
                else {
                    log::info!(
                        "Failed to parse as structured data, treating as simple text answer: '{}'",
                        simple_string
                    );
                    (Vec::new(), Vec::new(), Some(simple_string))
                }
            } else {
                // This is a simple text response (e.g., "6", "There are 3 chairs in this room")
                log::info!("Answer is simple text response: '{}'", simple_string);
                (Vec::new(), Vec::new(), Some(simple_string))
            }
        }
        // Case 3: answer is a simple number
        serde_json::Value::Number(num) => {
            let text_response = num.to_string();
            log::info!("Answer is simple number response: '{}'", text_response);
            (Vec::new(), Vec::new(), Some(text_response))
        }
        // Case 4: Unexpected format
        other => {
            log::warn!("Unexpected answer format: {:?}", other);
            return Ok(McpResponse {
                answer: Vec::new(),
                paths: Vec::new(),
                scene_normal_vector: None,
                text_answer: None,
            });
        }
    };

    log::info!(
        "Successfully parsed {} objects, {} paths, and text answer: {:?} from MCP response",
        answer.len(),
        paths.len(),
        text_answer
    );
    Ok(McpResponse {
        answer,
        paths,
        scene_normal_vector: raw_response.scene_normal_vector,
        text_answer,
    })
}

/// Parse scene normal vector from string format "[x,y,z]"
pub fn parse_scene_normal_vector(normal_str: &str) -> Option<cgmath::Vector3<f32>> {
    use cgmath::InnerSpace;

    // Remove brackets and split by comma
    let cleaned = normal_str.trim_matches(|c| c == '[' || c == ']');
    let parts: Vec<&str> = cleaned.split(',').collect();

    if parts.len() == 3 {
        if let (Ok(x), Ok(y), Ok(z)) = (
            parts[0].trim().parse::<f32>(),
            parts[1].trim().parse::<f32>(),
            parts[2].trim().parse::<f32>(),
        ) {
            let vector = cgmath::Vector3::new(x, y, z);
            if vector.magnitude() > 0.001 {
                return Some(vector.normalize());
            }
        }
    }

    log::warn!("Failed to parse scene normal vector from: '{}'", normal_str);
    None
}

// Global storage for async responses in WASM
#[cfg(target_arch = "wasm32")]
use std::collections::HashMap;
#[cfg(target_arch = "wasm32")]
use std::sync::Mutex;

#[cfg(target_arch = "wasm32")]
lazy_static::lazy_static! {
    static ref ASYNC_RESPONSES: Mutex<HashMap<String, (String, McpResponse)>> = Mutex::new(HashMap::new());
}

#[cfg(target_arch = "wasm32")]
pub fn store_async_response(request_id: String, message: String, response: McpResponse) {
    log::info!("Storing async response for request_id: {}", request_id);
    if let Ok(mut responses) = ASYNC_RESPONSES.lock() {
        responses.insert(request_id, (message, response));
        log::info!("Successfully stored async response");
    } else {
        log::error!("Failed to acquire lock for async responses");
    }
}

#[cfg(target_arch = "wasm32")]
pub fn check_async_response(request_id: &str) -> Option<(String, McpResponse)> {
    if let Ok(mut responses) = ASYNC_RESPONSES.lock() {
        let result = responses.remove(request_id);
        if result.is_some() {
            log::info!("Retrieved async response for request_id: {}", request_id);
        }
        result
    } else {
        log::error!("Failed to acquire lock for async responses");
        None
    }
}

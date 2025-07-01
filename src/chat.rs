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
        })
    }
}

/// Helper function to parse MCP response that handles multiple formats
fn parse_mcp_response(
    response_text: &str,
) -> Result<McpResponse, Box<dyn std::error::Error + Send + Sync>> {
    #[derive(Debug, Clone, Deserialize)]
    struct RawMcpResponse {
        answer: serde_json::Value,
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

    let (answer, paths) = match raw_response.answer {
        // Case 1: answer is already a JSON array (legacy object format)
        serde_json::Value::Array(arr) => {
            log::info!("Answer is direct JSON array with {} items", arr.len());
            let objects =
                serde_json::from_value::<Vec<SceneObject>>(serde_json::Value::Array(arr))?;
            (objects, Vec::new())
        }
        // Case 2: answer is a JSON string that needs to be parsed
        serde_json::Value::String(json_string) => {
            log::info!("Answer is JSON string, parsing it: {}", json_string);

            // Try to parse as paths wrapper first (new navigation format)
            if let Ok(paths_wrapper) = serde_json::from_str::<PathsWrapper>(&json_string) {
                log::info!(
                    "Parsed as paths wrapper with {} paths",
                    paths_wrapper.paths.len()
                );
                (Vec::new(), paths_wrapper.paths)
            }
            // Try to parse as array first (old object format)
            else if let Ok(objects) = serde_json::from_str::<Vec<SceneObject>>(&json_string) {
                log::info!("Parsed as direct object array with {} items", objects.len());
                (objects, Vec::new())
            }
            // Try to parse as object with "objects" key (new object format)
            else if let Ok(wrapper) = serde_json::from_str::<ObjectsWrapper>(&json_string) {
                log::info!(
                    "Parsed as objects wrapper with {} items",
                    wrapper.objects.len()
                );
                (wrapper.objects, Vec::new())
            }
            // Failed to parse any format
            else {
                log::warn!("Failed to parse JSON string as objects, paths, or objects wrapper");
                (Vec::new(), Vec::new())
            }
        }
        // Case 3: Unexpected format
        other => {
            log::warn!("Unexpected answer format: {:?}", other);
            return Ok(McpResponse {
                answer: Vec::new(),
                paths: Vec::new(),
            });
        }
    };

    log::info!(
        "Successfully parsed {} objects and {} paths from MCP response",
        answer.len(),
        paths.len()
    );
    Ok(McpResponse { answer, paths })
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

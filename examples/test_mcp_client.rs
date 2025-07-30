use web_splats::mcp_client::MCPClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    env_logger::init();
    
    println!("🚀 Testing simplified SSE-based MCP client");
    
    // Create a new MCP client
    let mut client = MCPClient::new("http://localhost:8080".to_string());
    println!("✅ MCP client created successfully");
    
    // Start the client
    client.start().await?;
    println!("✅ MCP client started");
    
    // Send a test message
    let test_message = "where is the coffee machine?";
    let test_location = [0.0, 1.0, 0.0];
    
    println!("📤 Sending test message: '{}'", test_message);
    let mut args = serde_json::Map::new();
    args.insert("query".to_string(), serde_json::json!(test_message));
    
    // Call the tool and get the response directly
    match client.call_tool("Our Awesome Tool", args, test_location).await {
        Ok(response) => {
            println!("✅ Received response from rmcp tool call");
            if let Some(text) = &response.text_answer {
                println!("💬 Text response: {}", text);
            }
            if !response.answer.is_empty() {
                println!("🎯 Found {} objects", response.answer.len());
                for obj in &response.answer {
                    println!("   - {}", obj.name);
                }
            }
            if !response.paths.is_empty() {
                println!("🗺️ Found {} navigation paths", response.paths.len());
            }
        }
        Err(e) => {
            println!("⚠️ Tool call failed: {} (this is expected if no MCP server is running)", e);
        }
    }
    
    // Shutdown the client
    // client.shutdown().await?;
    println!("✅ MCP client shut down cleanly");
    
    Ok(())
} 
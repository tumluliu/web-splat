use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("🔧 MCP Request Format Demonstration");
    println!("===================================");
    
    // This is exactly what your MCP client now sends to the server
    let message = "where is the coffee machine?";
    let current_location = [1.5, 0.0, 2.3];
    
    // The CORRECT format that your rust-mcp-sdk server expects
    let mcp_request = json!({
        "messages": message,
        "current_location": current_location
    });
    
    println!("📋 Request format sent to your MCP server:");
    println!("{}", serde_json::to_string_pretty(&mcp_request)?);
    
    println!("\n✅ This matches exactly what you specified:");
    println!("   {{\"messages\": \"xxxx\", \"current_location\": [x,y,z]}}");
    
    println!("\n📡 The request will be sent via POST to: http://your-server:3000/sse");
    println!("🔧 Content-Type: application/json");
    
    Ok(())
} 
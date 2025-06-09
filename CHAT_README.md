# 3D Scene Chat Integration

This document describes the new chat functionality added to the 3D Gaussian Splatting viewer for scene understanding and navigation.

## Features

The chat interface allows users to:
- Ask questions about objects in the 3D scene (e.g., "Where is the biggest table?")
- Request navigation paths between locations (e.g., "How do I get from the kitchen to the dining room?")
- Get information about scene objects with their positions and attributes

## Components

### 1. Chat UI
- **Location**: Integrated into the viewer's UI as a new window
- **Features**: 
  - Message history display
  - Text input for queries
  - Highlight controls for found objects/paths
  - Server configuration settings

### 2. Chat State Management
- **File**: `src/chat.rs`
- **Structures**:
  - `ChatMessage`: Individual chat messages with timestamps
  - `SceneObject`: 3D objects with positions and attributes
  - `ScenePath`: Navigation paths with waypoints
  - `McpResponse`: Server response format
  - `ChatState`: Overall chat state management

### 3. Mock MCP Server
- **File**: `mock_mcp_server.py`
- **Purpose**: Provides realistic responses for testing the chat functionality
- **Features**:
  - Object recognition and search
  - Navigation path generation
  - Natural language query processing

## Usage

### Starting the Mock Server

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Start the mock MCP server:
```bash
python mock_mcp_server.py
```

The server will start on `http://localhost:8080` and provide example queries.

### Using the Chat Interface

1. **Build and run the viewer**:
```bash
cargo run --bin viewer -- path/to/your/scene.ply
```

2. **Open the chat window**: The "💬 3D Scene Chat" window should appear in the UI

3. **Ask questions about the scene**:
   - "Where is the biggest table?"
   - "Show me all the chairs"
   - "Find the kitchen counter"
   - "How do I get from the kitchen to the dining table?"
   - "Navigate to the sofa"

### Example Queries

#### Object Queries
- **"Where is the table?"** → Returns dining table location and attributes
- **"Show me all chairs"** → Lists all chairs with positions
- **"Find the biggest table"** → Returns the largest table in the scene
- **"Where is the kitchen counter?"** → Returns kitchen counter location

#### Navigation Queries
- **"How do I get from kitchen to dining room?"** → Returns waypoint path
- **"Navigate to the sofa"** → Provides navigation instructions
- **"Show me the path to the entrance"** → Generates route to entrance

## Technical Details

### Message Flow
1. User types query in chat input
2. Query is sent to MCP server via HTTP POST to `/query`
3. Server processes query and returns structured response
4. Response is parsed and displayed in chat
5. Objects/paths are highlighted in the 3D view (planned feature)

### Response Format
The MCP server returns JSON responses in one of three formats:

#### Objects Response
```json
{
  "type": "objects",
  "objects": [
    {
      "name": "Dining Table",
      "position": [2.5, 0.0, 1.0],
      "attributes": {
        "type": "dining_table",
        "material": "wood",
        "size": "large"
      },
      "confidence": 0.95
    }
  ],
  "description": "Found a large wooden dining table"
}
```

#### Path Response
```json
{
  "type": "path",
  "path": {
    "waypoints": [
      [0.0, 0.0, 2.5],
      [1.0, 0.0, 2.0],
      [2.5, 0.0, 1.0]
    ],
    "description": "Path from kitchen to dining table"
  },
  "description": "Navigation path calculated"
}
```

#### Error Response
```json
{
  "type": "error",
  "message": "Could not find any objects matching your query"
}
```

## Configuration

### Server URL
The chat interface allows configuring the MCP server URL in the Settings section. Default is `http://localhost:8080`.

### Mock Data
The mock server includes predefined objects:
- Dining Table (large, wood, brown)
- Kitchen Counter (medium, granite, black)
- Chairs (small, wood, brown)
- Refrigerator (large, stainless steel, silver)
- Sofa (large, fabric, blue)

## Future Enhancements

1. **Real-time highlighting**: Highlight found objects in the 3D view
2. **Path visualization**: Draw navigation paths as 3D lines
3. **Voice input**: Add speech-to-text for hands-free queries
4. **Advanced queries**: Support for complex spatial relationships
5. **Scene analysis**: Integration with actual 3D scene understanding models
6. **Multi-modal responses**: Include images or 3D annotations

## Development Notes

### Adding New Object Types
To add new object types to the mock server:
1. Add objects to `MOCK_OBJECTS` list in `mock_mcp_server.py`
2. Update `find_objects_by_query()` function to handle new keywords
3. Test with appropriate queries

### Extending Query Types
To support new query types:
1. Modify the query processing logic in `handle_query()`
2. Add new response types to `McpResponse` enum in `src/chat.rs`
3. Update the UI to handle new response types

### Real MCP Server Integration
To connect to a real MCP server:
1. Replace the mock response generation with actual HTTP requests
2. Handle async responses properly in the UI
3. Add error handling for network issues
4. Implement authentication if required

## Troubleshooting

### Common Issues

1. **Server not responding**: Ensure the mock server is running on port 8080
2. **CORS errors**: The mock server includes CORS headers for browser compatibility
3. **Chat not appearing**: Check that the chat UI window is not minimized
4. **Input not working**: Ensure the text input field has focus

### Debug Information
The mock server prints received queries and responses to the console for debugging.

## Dependencies

### Rust Dependencies
- `reqwest`: HTTP client for server communication
- `tokio`: Async runtime (for future async implementation)
- `serde`: JSON serialization/deserialization
- `egui`: UI framework

### Python Dependencies
- `Flask`: Web server framework
- `Flask-CORS`: Cross-origin resource sharing support 
#!/bin/bash

# Start Mock MCP Server for 3D Scene Understanding

echo "Starting Mock MCP Server..."
echo "Make sure you have Python and the required dependencies installed:"
echo "  pip install -r requirements.txt"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if Flask is installed
if ! python3 -c "import flask" &> /dev/null; then
    echo "Error: Flask is not installed. Please run: pip install -r requirements.txt"
    exit 1
fi

# Start the server
echo "Starting server on http://localhost:8080..."
python3 mock_mcp_server.py 
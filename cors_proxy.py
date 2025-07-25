#!/usr/bin/env python3
"""
CORS Proxy for MCP Server (SSE Mode)
Forwards SSE requests to a real rust-mcp-sdk server while adding CORS headers for web browsers.
Supports both /sse endpoint and legacy /query endpoint (redirected to /sse).
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import os
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
REAL_MCP_SERVER_URL = os.environ.get(
    "REAL_MCP_SERVER_URL", "http://mcp_server_ip:3000"
)

print(f"🔄 CORS Proxy starting...")
print(f"📡 Will forward requests to: {REAL_MCP_SERVER_URL}")
print(f"🌐 Proxy will be available at: http://localhost:8080")


@app.route("/sse", methods=["POST", "OPTIONS"])
def proxy_sse():
    """Proxy the /sse endpoint to the real MCP server"""

    if request.method == "OPTIONS":
        # Handle preflight request
        response = Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Cache-Control, Connection"
        return response

    try:
        # Forward the request to the real MCP server
        real_url = f"{REAL_MCP_SERVER_URL}/sse"

        print(f"🔄 Forwarding SSE request to: {real_url}")
        print(f"📦 Request data: {request.get_json()}")

        # Forward with SSE headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }

        response = requests.post(
            real_url,
            json=request.get_json(),
            headers=headers,
            timeout=30,
        )

        print(f"📡 Response status: {response.status_code}")
        print(f"📝 Response data: {response.text}")

        # Return the response with CORS headers for SSE
        return Response(
            response.content,
            status=response.status_code,
            headers={
                "Content-Type": response.headers.get("Content-Type", "text/event-stream"),
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Accept, Cache-Control, Connection",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        print(f"❌ Error forwarding SSE request: {e}")
        return jsonify({"answer": []}), 500


@app.route("/query", methods=["POST", "OPTIONS"])
def proxy_query():
    """Legacy /query endpoint - redirects to /sse for compatibility"""
    print(f"⚠️  Legacy /query endpoint hit - redirecting to /sse")
    return proxy_sse()


@app.route("/health", methods=["GET"])
def health_check():
    """Health check that also tests the real server"""
    try:
        # Try to check if the server has a health endpoint
        real_url = f"{REAL_MCP_SERVER_URL}/health"
        response = requests.get(real_url, timeout=5)
        return jsonify(
            {
                "proxy_status": "healthy",
                "real_server_status": response.status_code,
                "real_server_response": response.text,
                "endpoints": {
                    "sse": f"{REAL_MCP_SERVER_URL}/sse",
                    "health": f"{REAL_MCP_SERVER_URL}/health"
                }
            }
        )
    except Exception as e:
        return jsonify(
            {
                "proxy_status": "healthy",
                "real_server_status": "error",
                "real_server_error": str(e),
                "note": "Health endpoint failed, but SSE endpoint might still work",
                "endpoints": {
                    "sse": f"{REAL_MCP_SERVER_URL}/sse"
                }
            }
        ), 200


@app.route("/", methods=["GET"])
def index():
    """Information about the proxy"""
    return jsonify(
        {
            "name": "CORS Proxy for MCP Server",
            "purpose": "Forwards requests to real MCP server while adding CORS headers",
            "real_server": REAL_MCP_SERVER_URL,
            "proxy_endpoints": {
                "/sse": "POST - Forwards to real MCP server (SSE endpoint)",
                "/query": "POST - Legacy endpoint (redirects to /sse)",
                "/health": "GET - Health check with real server test",
                "/": "GET - This information",
            },
        }
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        REAL_MCP_SERVER_URL = sys.argv[1]
        print(f"📡 Using MCP server URL from command line: {REAL_MCP_SERVER_URL}")

    print("=" * 80)
    print("🚀 CORS PROXY STARTING (SSE MODE)")
    print(f"📡 Real MCP Server: {REAL_MCP_SERVER_URL}")
    print("🌐 Proxy Server: http://localhost:8080")
    print("💡 Usage:")
    print("   1. Start this proxy: python cors_proxy.py [real-server-url]")
    print("   2. Configure web app to use: http://localhost:8080")
    print("   3. SSE requests to /sse will be forwarded to the real server")
    print("   4. Legacy /query requests will redirect to /sse")
    print("🔗 Forwarding:")
    print(f"   • /sse → {REAL_MCP_SERVER_URL}/sse")
    print(f"   • /query → {REAL_MCP_SERVER_URL}/sse (redirected)")
    print("=" * 80)

    app.run(host="0.0.0.0", port=8080, debug=True)

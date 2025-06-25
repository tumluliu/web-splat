#!/usr/bin/env python3
"""
CORS Proxy for MCP Server
Forwards requests to a real MCP server while adding CORS headers for web browsers.
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
    "REAL_MCP_SERVER_URL", "http://external-server:8080"
)

print(f"🔄 CORS Proxy starting...")
print(f"📡 Will forward requests to: {REAL_MCP_SERVER_URL}")
print(f"🌐 Proxy will be available at: http://localhost:8080")


@app.route("/query", methods=["POST", "OPTIONS"])
def proxy_query():
    """Proxy the /query endpoint to the real MCP server"""

    if request.method == "OPTIONS":
        # Handle preflight request
        response = Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    try:
        # Forward the request to the real MCP server
        real_url = f"{REAL_MCP_SERVER_URL}/query"

        print(f"🔄 Forwarding request to: {real_url}")
        print(f"📦 Request data: {request.get_json()}")

        response = requests.post(
            real_url,
            json=request.get_json(),
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        print(f"📡 Response status: {response.status_code}")
        print(f"📝 Response data: {response.text}")

        # Return the response with CORS headers
        return Response(
            response.content,
            status=response.status_code,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    except Exception as e:
        print(f"❌ Error forwarding request: {e}")
        return jsonify({"answer": []}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check that also tests the real server"""
    try:
        real_url = f"{REAL_MCP_SERVER_URL}/health"
        response = requests.get(real_url, timeout=5)
        return jsonify(
            {
                "proxy_status": "healthy",
                "real_server_status": response.status_code,
                "real_server_response": response.text,
            }
        )
    except Exception as e:
        return jsonify(
            {
                "proxy_status": "healthy",
                "real_server_status": "error",
                "real_server_error": str(e),
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
                "/query": "POST - Forwards to real MCP server",
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
    print("🚀 CORS PROXY STARTING")
    print(f"📡 Real MCP Server: {REAL_MCP_SERVER_URL}")
    print("🌐 Proxy Server: http://localhost:8080")
    print("💡 Usage:")
    print("   1. Start this proxy: python cors_proxy.py [real-server-url]")
    print("   2. Configure web app to use: http://localhost:8080")
    print("   3. All requests will be forwarded to the real server")
    print("=" * 80)

    app.run(host="0.0.0.0", port=8080, debug=True)

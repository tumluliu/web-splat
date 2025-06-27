#!/bin/bash

set -e  # Exit on any error

# Configuration
REAL_MCP_SERVER_URL="${1:-http://localhost:8080}"  # Default to localhost if no arg provided
WEB_PORT=8000
PROXY_PORT=8080

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}🚀 Web-Splat Development Stack Launcher${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Function to kill process on port
kill_port() {
    if port_in_use $1; then
        echo -e "${YELLOW}⚠️  Port $1 is in use, killing existing process...${NC}"
        lsof -ti :$1 | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

echo -e "${GREEN}📋 Configuration:${NC}"
echo -e "   Real MCP Server: ${REAL_MCP_SERVER_URL}"
echo -e "   CORS Proxy Port: ${PROXY_PORT}"
echo -e "   Web Server Port: ${WEB_PORT}"
echo ""

# Check dependencies
echo -e "${GREEN}🔍 Checking dependencies...${NC}"

if ! command_exists python3; then
    echo -e "${RED}❌ Python3 not found. Please install Python 3.x${NC}"
    exit 1
fi

if ! command_exists cargo; then
    echo -e "${RED}❌ Cargo not found. Please install Rust${NC}"
    exit 1
fi

if ! python3 -c "import flask, flask_cors, requests" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Python dependencies missing. Installing...${NC}"
    pip install -r requirements.txt || {
        echo -e "${RED}❌ Failed to install Python dependencies${NC}"
        exit 1
    }
fi

if ! cargo --list | grep -q wasm-bindgen; then
    echo -e "${YELLOW}⚠️  wasm-bindgen-cli not found. Installing...${NC}"
    cargo install wasm-bindgen-cli
fi

if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
    echo -e "${YELLOW}⚠️  WASM target not found. Installing...${NC}"
    rustup target add wasm32-unknown-unknown
fi

echo -e "${GREEN}✅ All dependencies OK${NC}"
echo ""

# Build WASM if needed
if [ ! -f "public/pkg/web_splats.js" ] || [ "src/lib.rs" -nt "public/pkg/web_splats.js" ]; then
    echo -e "${GREEN}🔨 Building WASM...${NC}"
    ./build_wasm.sh || {
        echo -e "${RED}❌ WASM build failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ WASM build complete${NC}"
else
    echo -e "${GREEN}✅ WASM already up to date${NC}"
fi
echo ""

# Clean up any existing processes
echo -e "${GREEN}🧹 Cleaning up existing processes...${NC}"
kill_port $PROXY_PORT
kill_port $WEB_PORT

# Start CORS proxy in background
echo -e "${GREEN}🔄 Starting CORS proxy...${NC}"
echo -e "   Forwarding: localhost:${PROXY_PORT} → ${REAL_MCP_SERVER_URL}"

python3 cors_proxy.py "$REAL_MCP_SERVER_URL" > cors_proxy.log 2>&1 &
PROXY_PID=$!

# Wait for proxy to start
sleep 2
if ! port_in_use $PROXY_PORT; then
    echo -e "${RED}❌ Failed to start CORS proxy${NC}"
    echo -e "${RED}   Check cors_proxy.log for details${NC}"
    exit 1
fi

echo -e "${GREEN}✅ CORS proxy running (PID: $PROXY_PID)${NC}"

# Start web server in background
echo -e "${GREEN}🌐 Starting web server...${NC}"
cd public
python3 -m http.server $WEB_PORT > ../web_server.log 2>&1 &
WEB_PID=$!
cd ..

# Wait for web server to start
sleep 2
if ! port_in_use $WEB_PORT; then
    echo -e "${RED}❌ Failed to start web server${NC}"
    echo -e "${RED}   Check web_server.log for details${NC}"
    kill $PROXY_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}✅ Web server running (PID: $WEB_PID)${NC}"
echo ""

# Test the proxy connection
echo -e "${GREEN}🧪 Testing CORS proxy connection...${NC}"
if curl -s "http://localhost:${PROXY_PORT}/health" > /dev/null; then
    echo -e "${GREEN}✅ CORS proxy responding${NC}"
else
    echo -e "${YELLOW}⚠️  CORS proxy health check failed (this might be OK if the real server doesn't have /health)${NC}"
fi
echo ""

# Show access URLs
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}🎉 DEVELOPMENT STACK READY!${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "${GREEN}📱 Web App URLs:${NC}"
echo -e "   Demo Gallery: ${BLUE}http://localhost:${WEB_PORT}/demo.html${NC}"
echo -e "   Direct Scene:  ${BLUE}http://localhost:${WEB_PORT}/index.html${NC}"
echo ""
echo -e "${GREEN}🔧 Configuration:${NC}"
echo -e "   CORS Proxy:    ${BLUE}http://localhost:${PROXY_PORT}${NC}"
echo -e "   Real MCP:      ${BLUE}${REAL_MCP_SERVER_URL}${NC}"
echo ""
echo -e "${GREEN}📊 Monitoring:${NC}"
echo -e "   Proxy logs:    ${BLUE}tail -f cors_proxy.log${NC}"
echo -e "   Web logs:      ${BLUE}tail -f web_server.log${NC}"
echo ""
echo -e "${GREEN}💡 Setup Instructions:${NC}"
echo -e "   1. Open: ${BLUE}http://localhost:${WEB_PORT}/demo.html${NC}"
echo -e "   2. In MCP Configuration, set URL: ${BLUE}http://localhost:${PROXY_PORT}${NC}"
echo -e "   3. Click 'Save' to store the configuration"
echo -e "   4. Load any scene and test the chat functionality"
echo ""
echo -e "${YELLOW}🛑 To stop everything: ${BLUE}kill $PROXY_PID $WEB_PID${NC}"
echo ""

# Create cleanup script
cat > stop_dev_stack.sh << EOF
#!/bin/bash
echo "🛑 Stopping development stack..."
kill $PROXY_PID $WEB_PID 2>/dev/null || true
echo "✅ All processes stopped"
rm -f stop_dev_stack.sh
EOF
chmod +x stop_dev_stack.sh

echo -e "${GREEN}📄 Created stop_dev_stack.sh for easy cleanup${NC}"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop this script, then run ./stop_dev_stack.sh to clean up${NC}"

# Keep script running and show logs
echo -e "${GREEN}📊 Live logs (Ctrl+C to stop):${NC}"
echo ""

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down...${NC}"
    kill $PROXY_PID $WEB_PID 2>/dev/null || true
    echo -e "${GREEN}✅ All processes stopped${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Show live logs
tail -f cors_proxy.log &
TAIL_PID=$!

# Wait for interrupt
wait

# Cleanup tail process
kill $TAIL_PID 2>/dev/null || true
cleanup 
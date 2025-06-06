#!/usr/bin/env python3
import http.server
import ssl
import socketserver
import os

# Create self-signed certificate if it doesn't exist
if not os.path.exists("server.crt"):
    os.system(
        'openssl req -new -x509 -keyout server.key -out server.crt -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Org/CN=localhost"'
    )

PORT = 8443
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain("server.crt", "server.key")
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    print(f"Serving at https://localhost:{PORT}")
    httpd.serve_forever()

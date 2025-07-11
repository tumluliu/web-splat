#!/usr/bin/env python3
"""
Simple Mock MCP Server for 3D Scene Understanding
This server provides mock responses for testing the chat functionality.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import random
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

print("=" * 80)
print("🚀 MOCK MCP SERVER STARTING UP")
print("=" * 80)


# Add logging for every request
@app.before_request
def log_request_info():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 80}")
    print(f"📥 [{timestamp}] INCOMING REQUEST")
    print(f"🔗 URL: {request.url}")
    print(f"📝 Method: {request.method}")
    print(f"🌐 Headers: {dict(request.headers)}")
    print(f"📍 Remote Address: {request.remote_addr}")
    print(f"🔍 User Agent: {request.headers.get('User-Agent', 'N/A')}")
    if request.method == "POST":
        print(f"📦 Raw Data: {request.get_data()}")
        try:
            print(f"📋 JSON Data: {request.get_json()}")
        except:
            print("📋 JSON Data: Failed to parse")
    print("=" * 80)


# Mock scene objects database
MOCK_OBJECTS = [
    {
        "name": "Dining Table",
        "position": [2.5, 0.0, 1.0],
        "attributes": {
            "type": "dining_table",
            "material": "wood",
            "size": "large",
            "color": "brown",
        },
        "confidence": 0.95,
    },
    {
        "name": "Kitchen Counter",
        "position": [0.0, 0.0, 2.5],
        "attributes": {
            "type": "counter",
            "material": "granite",
            "size": "medium",
            "color": "black",
        },
        "confidence": 0.90,
    },
    {
        "name": "Chair 1",
        "position": [1.0, 0.0, 1.0],
        "attributes": {
            "type": "chair",
            "material": "wood",
            "size": "small",
            "color": "brown",
        },
        "confidence": 0.85,
    },
    {
        "name": "Chair 2",
        "position": [4.0, 0.0, 1.0],
        "attributes": {
            "type": "chair",
            "material": "wood",
            "size": "small",
            "color": "brown",
        },
        "confidence": 0.85,
    },
    {
        "name": "Refrigerator",
        "position": [-1.0, 0.0, 3.0],
        "attributes": {
            "type": "appliance",
            "subtype": "refrigerator",
            "material": "stainless_steel",
            "size": "large",
            "color": "silver",
        },
        "confidence": 0.92,
    },
    {
        "name": "Sofa",
        "position": [5.0, 0.0, -2.0],
        "attributes": {
            "type": "seating",
            "material": "fabric",
            "size": "large",
            "color": "blue",
        },
        "confidence": 0.88,
    },
    {
        "name": "Coffee Machine",
        "position": [18.610, 2.469, 4.476],
        "attributes": {
            "type": "appliance",
            "subtype": "coffee_machine",
            "material": "stainless_steel",
            "size": "medium",
            "color": "black",
            "width": "6.634",
            "height": "6.785",
            "depth": "5.602",
            "brand": "premium",
            "function": "brewing",
        },
        "confidence": 0.94,
    },
]

# Mock navigation paths
MOCK_PATHS = {
    "kitchen_to_dining": {
        "waypoints": [
            [0.0, 0.0, 2.5],  # Kitchen counter
            [1.5, 0.5, 2.0],  # Midpoint 1
            [2.0, 0.2, 1.5],  # Midpoint 2
            [2.5, 0.0, 1.0],  # Dining table
        ],
        "description": "Path from kitchen counter to dining table",
    },
    "entrance_to_sofa": {
        "waypoints": [
            [0.0, 0.0, 0.0],  # Entrance
            [1.5, 0.0, -0.5],  # Midpoint 1
            [3.0, 0.0, -1.0],  # Midpoint 2
            [4.0, 0.0, -1.5],  # Midpoint 3
            [5.0, 0.0, -2.0],  # Sofa
        ],
        "description": "Path from entrance to living room sofa",
    },
    "kitchen_to_coffee": {
        "waypoints": [
            [0.0, 0.0, 2.5],  # Kitchen counter
            [3.0, 0.5, 3.0],  # Midpoint 1
            [6.0, 1.0, 3.2],  # Midpoint 2
            [9.0, 1.5, 3.5],  # Midpoint 3
            [12.0, 2.0, 3.8],  # Midpoint 4
            [15.0, 2.2, 4.0],  # Approaching coffee area
            [18.610, 2.469, 4.476],  # Coffee machine
        ],
        "description": "Path from kitchen counter to coffee machine",
    },
    "dining_to_coffee": {
        "waypoints": [
            [2.5, 0.0, 1.0],  # Dining table
            [4.5, 0.3, 1.5],  # Midpoint 1
            [7.0, 1.0, 2.0],  # Midpoint 2
            [10.0, 1.3, 2.8],  # Midpoint 3
            [12.0, 1.5, 3.0],  # Midpoint 4
            [15.0, 2.0, 3.5],  # Midpoint 5
            [16.0, 2.0, 4.0],  # Approaching coffee area
            [18.610, 2.469, 4.476],  # Coffee machine
        ],
        "description": "Path from dining table to coffee machine",
    },
    "table_to_anywhere": {
        "waypoints": [
            [2.5, 0.0, 1.0],  # Dining table (start)
            [4.0, 0.2, 0.5],  # Curve around furniture
            [6.0, 0.5, 0.0],  # Navigate around room
            [8.0, 0.8, -0.5],  # Continue path
            [10.0, 1.0, -1.0],  # Approach destination area
            [12.0, 1.2, -1.5],  # Final approach
        ],
        "description": "Scenic path from dining table to destination",
    },
}


def is_navigation_query(query):
    """Check if the query is asking for navigation/path information"""
    query = query.lower()
    navigation_keywords = [
        "go",
        "navigate",
        "path",
        "route",
        "way",
        "walk",
        "move",
        "how to get",
        "direction",
        "travel",
        "journey",
        "from",
        "to",
    ]
    return any(keyword in query for keyword in navigation_keywords)


def find_objects_by_query(query):
    """Find objects based on natural language query"""
    query = query.lower()
    matching_objects = []

    # Check for specific object types
    if "table" in query:
        matching_objects.extend(
            [
                obj
                for obj in MOCK_OBJECTS
                if "table" in obj["attributes"].get("type", "")
            ]
        )

    if "chair" in query:
        matching_objects.extend(
            [
                obj
                for obj in MOCK_OBJECTS
                if "chair" in obj["attributes"].get("type", "")
            ]
        )

    if "kitchen" in query or "counter" in query:
        matching_objects.extend(
            [
                obj
                for obj in MOCK_OBJECTS
                if "counter" in obj["attributes"].get("type", "")
            ]
        )

    if "refrigerator" in query or "fridge" in query:
        matching_objects.extend(
            [
                obj
                for obj in MOCK_OBJECTS
                if "refrigerator" in obj["attributes"].get("subtype", "")
            ]
        )

    if "sofa" in query or "couch" in query:
        matching_objects.extend(
            [
                obj
                for obj in MOCK_OBJECTS
                if "seating" in obj["attributes"].get("type", "")
            ]
        )

    if "coffee" in query or "espresso" in query or "brew" in query:
        matching_objects.extend(
            [
                obj
                for obj in MOCK_OBJECTS
                if "coffee_machine" in obj["attributes"].get("subtype", "")
            ]
        )

    # Handle size queries
    if "big" in query or "large" in query or "biggest" in query:
        matching_objects = [
            obj for obj in matching_objects if obj["attributes"].get("size") == "large"
        ]

    if "small" in query or "smallest" in query:
        matching_objects = [
            obj for obj in matching_objects if obj["attributes"].get("size") == "small"
        ]

    # Remove duplicates
    seen = set()
    unique_objects = []
    for obj in matching_objects:
        if obj["name"] not in seen:
            unique_objects.append(obj)
            seen.add(obj["name"])

    return unique_objects


def find_path_by_query(query):
    """Find navigation path based on query"""
    query = query.lower()

    if ("kitchen" in query and "dining" in query) or (
        "counter" in query and "table" in query
    ):
        return MOCK_PATHS["kitchen_to_dining"]

    if ("entrance" in query and "sofa" in query) or (
        "door" in query and "living" in query
    ):
        return MOCK_PATHS["entrance_to_sofa"]

    if ("kitchen" in query and "coffee" in query) or (
        "counter" in query and "coffee" in query
    ):
        return MOCK_PATHS["kitchen_to_coffee"]

    if ("dining" in query and "coffee" in query) or (
        "table" in query and "coffee" in query
    ):
        return MOCK_PATHS["dining_to_coffee"]

    # Handle generic coffee machine navigation
    if "coffee" in query and any(
        word in query for word in ["go", "navigate", "path", "route"]
    ):
        return MOCK_PATHS["dining_to_coffee"]  # Default path to coffee machine

    # Handle generic table navigation with the new scenic path
    if "table" in query and any(
        word in query for word in ["go", "navigate", "path", "route", "to"]
    ):
        return MOCK_PATHS["table_to_anywhere"]  # Use scenic path for table navigation

    # Generate a longer, more interesting random path if no specific match
    start = [0.0, 0.0, 0.0]
    end = [random.uniform(8, 15), random.uniform(0, 2), random.uniform(-2, 4)]

    # Create a curved path with multiple waypoints
    waypoints = [start]
    for i in range(1, 5):  # Add 3 intermediate waypoints
        t = i / 5.0
        # Use a curved interpolation instead of straight line
        x = (
            start[0]
            + (end[0] - start[0]) * t
            + random.uniform(-2, 2) * (1 - abs(t - 0.5) * 2)
        )
        y = start[1] + (end[1] - start[1]) * t + random.uniform(0, 1) * t
        z = (
            start[2]
            + (end[2] - start[2]) * t
            + random.uniform(-1, 1) * (1 - abs(t - 0.5) * 2)
        )
        waypoints.append([x, y, z])
    waypoints.append(end)

    return {"waypoints": waypoints, "description": "Generated curved navigation path"}


def create_object_from_mock_data(obj):
    """Convert mock object data to the new aligned_bbox format"""
    pos = obj["position"]
    width = float(obj["attributes"].get("width", "2.0"))
    height = float(obj["attributes"].get("height", "2.0"))
    depth = float(obj["attributes"].get("depth", "2.0"))

    # Create 8 corners of the bounding box
    aligned_bbox = [
        [
            pos[0] - width / 2,
            pos[1] - height / 2,
            pos[2] - depth / 2,
        ],  # Bottom-front-left
        [
            pos[0] + width / 2,
            pos[1] - height / 2,
            pos[2] - depth / 2,
        ],  # Bottom-front-right
        [
            pos[0] + width / 2,
            pos[1] + height / 2,
            pos[2] - depth / 2,
        ],  # Top-front-right
        [pos[0] - width / 2, pos[1] + height / 2, pos[2] - depth / 2],  # Top-front-left
        [
            pos[0] - width / 2,
            pos[1] - height / 2,
            pos[2] + depth / 2,
        ],  # Bottom-back-left
        [
            pos[0] + width / 2,
            pos[1] - height / 2,
            pos[2] + depth / 2,
        ],  # Bottom-back-right
        [pos[0] + width / 2, pos[1] + height / 2, pos[2] + depth / 2],  # Top-back-right
        [pos[0] - width / 2, pos[1] + height / 2, pos[2] + depth / 2],  # Top-back-left
    ]

    # Generate semantic front face normal vector based on object type
    normal_vector = [0.0, 0.0, 1.0]  # Default forward direction
    object_type = obj["attributes"].get("type", "")
    object_subtype = obj["attributes"].get("subtype", "")

    # Define semantic front directions for different object types
    if "chair" in obj["name"].lower():
        normal_vector = [
            0.0,
            0.0,
            -1.0,
        ]  # Chairs face negative Z (toward where person sits)
    elif "table" in object_type:
        normal_vector = [1.0, 0.0, 0.0]  # Tables face positive X (longest side)
    elif "sofa" in obj["name"].lower():
        normal_vector = [0.0, 0.0, -1.0]  # Sofas face where people sit
    elif "counter" in object_type:
        normal_vector = [0.0, 0.0, -1.0]  # Counters face the user side
    elif object_subtype == "refrigerator":
        normal_vector = [1.0, 0.0, 0.0]  # Refrigerator door faces positive X
    elif object_subtype == "coffee_machine":
        normal_vector = [
            0.0,
            1.0,
            0.0,
        ]  # Coffee machine front faces positive Y (upward for testing)
    else:
        # For unknown objects, use a reasonable default
        normal_vector = [1.0, 0.0, 0.0]

    return {
        "name": obj["name"],
        "aligned_bbox": aligned_bbox,
        "normal_vector": normal_vector,
        "attributes": obj.get("attributes"),
    }


@app.route("/query", methods=["POST"])
def handle_query():
    """Handle 3D scene understanding queries"""
    print(f"\n🔥🔥🔥 QUERY ENDPOINT HIT! 🔥🔥🔥")
    print(f"📝 Method: {request.method}")
    print(f"🌐 Headers: {dict(request.headers)}")
    print(f"📦 Raw data: {request.get_data()}")

    try:
        print("🔄 Attempting to parse JSON...")
        data = request.json
        messages = data.get("messages", "")
        context = data.get("context", "")
        current_location = data.get("current_location", [0.0, 0.0, 0.0])

        print(f"✅ JSON PARSED SUCCESSFULLY!")
        print(f"💬 Messages: '{messages}'")
        print(f"🔍 Context: '{context}'")
        print(
            f"📍 Current location: [{current_location[0]:.3f}, {current_location[1]:.3f}, {current_location[2]:.3f}]"
        )

        # Check if this is a navigation query
        if is_navigation_query(messages):
            print("🗺️  Detected NAVIGATION query - generating path response")

            # Find the target object for navigation
            target_objects = find_objects_by_query(messages)
            path_data = find_path_by_query(messages)

            if target_objects and path_data:
                target_object = target_objects[0]  # Use the first matching object
                target_object_formatted = create_object_from_mock_data(target_object)

                # Create path response format
                path_response = {
                    "object": target_object_formatted,
                    "path": path_data["waypoints"],
                }

                paths_dict = {"paths": [path_response]}
                response = {"answer": json.dumps(paths_dict)}

                print(f"✅ Sending PATH response:")
                print(f"  Target object: {target_object['name']}")
                print(f"  Path waypoints: {len(path_data['waypoints'])}")
                print(f"  Response: {json.dumps(response, indent=2)}")
            else:
                print("❌ Could not find target object or path for navigation query")
                response = {"answer": json.dumps({"paths": []})}
        else:
            print("🎯 Detected OBJECT query - generating object response")

            # Find objects based on the messages
            objects = find_objects_by_query(messages)

            # Convert objects to new format with aligned_bbox
            answer_objects = []
            for obj in objects:
                answer_objects.append(create_object_from_mock_data(obj))

            # New format: embed objects in a dict with "objects" key, then stringify
            objects_dict = {"objects": answer_objects}
            response = {"answer": json.dumps(objects_dict)}

            print(f"✅ Sending OBJECT response:")
            print(f"  Found objects: {len(answer_objects)}")
            print(f"  Response: {json.dumps(response, indent=2)}")

        print(f"🔴 REQUEST COMPLETE\n")
        return jsonify(response)

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        print(f"🔴 REQUEST FAILED\n")
        return jsonify({"answer": json.dumps({"objects": [], "paths": []})}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Mock MCP Server is running"})


@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API documentation"""
    return jsonify(
        {
            "name": "Mock MCP Server for 3D Scene Understanding",
            "version": "1.0.0",
            "endpoints": {
                "/query": "POST - Send 3D scene understanding queries (expects JSON with 'messages', 'context', and 'current_location' fields)",
                "/health": "GET - Health check",
                "/": "GET - This documentation",
            },
            "example_queries": [
                "Where is the biggest table?",
                "Show me all the chairs",
                "Find the kitchen counter",
                "How do I get from the kitchen to the dining table?",
                "Navigate to the sofa",
                "Go to the coffee machine",
                "Path to the refrigerator",
                "Route from dining table to coffee machine",
            ],
        }
    )


if __name__ == "__main__":
    print("🚀 Starting Mock MCP Server for 3D Scene Understanding...")
    print("🌐 Server will be available at http://localhost:8080")
    print("\n📝 Example queries:")
    print("  - Where is the biggest table?")
    print("  - Show me all the chairs")
    print("  - How do I get from kitchen to dining room?")
    print("\n⚠️  IMPORTANT: Watch for request logs above!")
    print("    If you don't see logs when sending chat messages,")
    print("    then HTTP requests are NOT reaching this server!")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("=" * 80)
    print("🔥 SERVER STARTING NOW - WATCHING FOR REQUESTS...")
    print("=" * 80)

    app.run(host="0.0.0.0", port=8080, debug=True)

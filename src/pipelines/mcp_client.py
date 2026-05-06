"""
MCP HTTP Client for communicating with the YOLO MCP Server
Uses streamable-http transport to call detection tools
"""
import json
import requests
from typing import Dict, Any, Optional


class MCPClient:
    """Client for MCP servers using streamable-http transport"""

    def __init__(self, base_url: str = "http://localhost:8099"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream'
        })
        self._initialized = False

    def _initialize(self) -> bool:
        """Initialize MCP session"""
        if self._initialized:
            return True

        try:
            # Send initialize request
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "yolo-mcp-client",
                        "version": "1.0.0"
                    }
                }
            }

            response = self.session.post(
                f"{self.base_url}/mcp",
                json=init_payload,
                timeout=10
            )
            response.raise_for_status()

            # Send initialized notification
            notif_payload = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            self.session.post(f"{self.base_url}/mcp", json=notif_payload, timeout=5)

            self._initialized = True
            return True

        except Exception as e:
            raise ConnectionError(f"Failed to initialize MCP session: {e}")

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool with given arguments

        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of tool arguments

        Returns:
            Tool response as dictionary
        """
        self._initialize()

        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }

            response = self.session.post(
                f"{self.base_url}/mcp",
                json=payload,
                timeout=60  # Image processing can take time
            )
            response.raise_for_status()

            # Parse response - handle both JSON and SSE formats
            content_type = response.headers.get('content-type', '')

            if 'text/event-stream' in content_type:
                # Parse SSE response
                return self._parse_sse_response(response.text)
            else:
                # Parse JSON response
                return self._parse_json_response(response.text)

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to MCP server at {self.base_url}. "
                "Make sure the server is running: cd image_detection && python object_detection_server.py"
            )
        except Exception as e:
            raise RuntimeError(f"MCP tool call failed: {e}")

    def _parse_sse_response(self, sse_text: str) -> Dict[str, Any]:
        """Parse Server-Sent Events response"""
        lines = sse_text.strip().split('\n')
        result_data = None

        for line in lines:
            if line.startswith('data: '):
                data_str = line[6:]  # Remove 'data: ' prefix
                if data_str == '[DONE]':
                    break
                try:
                    data = json.loads(data_str)
                    # Extract the actual result
                    if 'result' in data:
                        result_data = data['result']
                        break
                    elif 'error' in data:
                        raise RuntimeError(f"MCP tool error: {data['error']}")
                except json.JSONDecodeError:
                    continue

        if result_data and 'content' in result_data:
            content = result_data['content']
            for item in content:
                if item.get("type") == "text":
                    try:
                        return json.loads(item["text"])
                    except (json.JSONDecodeError, KeyError):
                        return {"text": item["text"]}

        return {"error": "No valid content in SSE response"}

    def _parse_json_response(self, json_text: str) -> Dict[str, Any]:
        """Parse JSON response"""
        result = json.loads(json_text)

        if "error" in result:
            raise RuntimeError(f"MCP tool error: {result['error']}")

        # Extract content from MCP response
        content = result.get("result", {}).get("content", [])

        # Find text content
        for item in content:
            if item.get("type") == "text":
                # Try to parse as JSON
                try:
                    return json.loads(item["text"])
                except (json.JSONDecodeError, KeyError):
                    return {"text": item["text"]}

        return {"error": "No valid content in response"}

    def list_tools(self) -> list:
        """List available tools on the MCP server"""
        self._initialize()

        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/list"
            }

            response = self.session.post(
                f"{self.base_url}/mcp",
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            result = response.json()
            return result.get("result", {}).get("tools", [])

        except Exception as e:
            raise RuntimeError(f"Failed to list tools: {e}")


def test_connection(base_url: str = "http://localhost:8099") -> bool:
    """Test connection to MCP server"""
    try:
        client = MCPClient(base_url)
        tools = client.list_tools()
        return len(tools) > 0
    except Exception:
        return False

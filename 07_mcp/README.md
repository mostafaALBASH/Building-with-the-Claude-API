# Basic MCP Server & Client

This repository provides a complete, streamlined implementation of the **Model Context Protocol (MCP)**. It includes a server to host custom tools and a client to orchestrate them using an LLM. This project was built by following a tutorial from Claude Academy. It serves as a learning project and a foundation for further development.

---

## 🛠️ Prerequisites

* **Python 3.10+**
* [uv](https://astral.sh/uv/) (Recommended for dependency management)
* **Anthropic API Key** (To run the client)

---

## 🚀 Quick Start (Python)

### 1. The Server (`server.py`)
The server defines the tools available to the AI.

```python
from fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("My-Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

if __name__ == "__main__":
    mcp.run()
```

### 2. The Client (`client.py`)
The client connects the AI to your server.

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_client():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {tools}")

if __name__ == "__main__":
    asyncio.run(run_client())
```

---

## ⚙️ Setup & Running

1. **Install Dependencies:**
   ```bash
   uv add fastmcp mcp anthropic
   ```

2. **Run the Server (Dev Mode):**
   This starts the server and a web-based inspector to test your tools.
   ```bash
   uv run mcp dev server.py
   ```

3. **Run the Client:**
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   uv run client.py
   ```

---

## 🤖 Integration: Claude Desktop

To use these tools directly inside the Claude Desktop app:

1. **Open Config:**
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

2. **Add this configuration:**
   ```json
   {
     "mcpServers": {
       "my-basic-mcp": {
         "command": "uv",
         "args": [
           "--directory",
           "/absolute/path/to/your/project",
           "run",
           "server.py"
         ]
       }
     }
   }
   ```
3. **Restart Claude Desktop.**

---

## 🔍 Debugging
Use the official inspector to test your server via the command line:
```bash
npx @modelcontextprotocol/inspector uv run server.py
```

---

## 📜 License
MIT
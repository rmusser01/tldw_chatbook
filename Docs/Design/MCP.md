# MCP (Model Context Protocol) Integration

## What is MCP?

The Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context to Large Language Models (LLMs). Think of it as a universal adapter that allows AI assistants to securely connect to local services and data sources on your computer.

### Key Concepts

**MCP enables AI models to:**
- Access local data and services through a standardized interface
- Execute tools and functions with proper authorization
- Retrieve contextual information from various sources
- Maintain security boundaries between AI and local resources

**MCP consists of three main primitives:**
1. **Tools** - Functions that AI can execute (like POST endpoints in REST APIs)
2. **Resources** - Data that AI can read (like GET endpoints in REST APIs)
3. **Prompts** - Reusable templates for common AI interactions

### Why MCP Matters

Without MCP, each AI application needs custom integrations for every data source or service. MCP provides:
- **Standardization**: One protocol for all AI-to-application communication
- **Security**: Controlled access with user consent
- **Modularity**: Mix and match servers and clients
- **Simplicity**: Easy to implement and use

### How It Works

1. **MCP Servers** expose functionality (like tldw_chatbook does)
2. **MCP Clients** (like Claude Desktop) connect to servers
3. **Communication** happens via JSON-RPC 2.0 over stdio or HTTP
4. **Users maintain control** over what data and functions are accessible

## Overview

tldw_chatbook includes comprehensive MCP support, allowing it to function as an MCP server that exposes its functionality to AI applications like Claude Desktop. This document describes the architecture, implementation, and usage of MCP within tldw_chatbook.

By implementing MCP, tldw_chatbook transforms from a standalone TUI application into a powerful context provider that AI assistants can use to:
- Search through your notes and conversations
- Manage and retrieve information from your knowledge base
- Interact with ingested media content
- Generate documents and summaries
- Have character-based conversations

## Table of Contents

1. [Architecture](#architecture)
2. [Components](#components)
3. [MCP Tools](#mcp-tools)
4. [MCP Resources](#mcp-resources)
5. [MCP Prompts](#mcp-prompts)
6. [Configuration](#configuration)
7. [Installation and Setup](#installation-and-setup)
8. [Running the MCP Server](#running-the-mcp-server)
9. [MCP Client](#mcp-client)
10. [Security Considerations](#security-considerations)
11. [Development Guide](#development-guide)
12. [Future Enhancements](#future-enhancements)

## Architecture

The MCP integration is designed as a modular system that exposes tldw_chatbook's core functionality through the Model Context Protocol standard.

### Directory Structure
```
tldw_chatbook/MCP/
├── __init__.py          # Module initialization and availability checking
├── __main__.py          # Entry point for running as module
├── server.py            # Main MCP server implementation
├── tools.py             # Tool implementations
├── resources.py         # Resource providers
├── prompts.py           # Prompt templates
└── client.py            # MCP client for external servers
```

### Design Principles
- **Modular Architecture**: Clear separation between server, tools, resources, and prompts
- **Async-First**: All operations use async/await for optimal performance
- **Security**: API keys and sensitive data are never exposed through MCP
- **Extensibility**: Easy to add new tools, resources, or prompts
- **Error Handling**: Comprehensive error handling with detailed logging

## Components

### Server (`server.py`)
The main MCP server implementation using FastMCP framework:
- Initializes database connections
- Registers tools, resources, and prompts
- Handles transport (stdio for Claude Desktop, HTTP planned)
- Manages server lifecycle

### Tools (`tools.py`)
Implementation of MCP tools that expose tldw_chatbook functionality:
- Encapsulates business logic for each tool
- Handles database operations
- Provides consistent error handling
- Returns structured responses

### Resources (`resources.py`)
Resource providers for accessing tldw_chatbook data:
- Formats data as markdown for readability
- Includes metadata in resource responses
- Supports dynamic resource listing
- Handles resource URIs with templates

### Prompts (`prompts.py`)
Reusable prompt templates for common workflows:
- Generates context-aware prompts
- Supports various output formats
- Includes customization parameters
- Integrates with RAG search results

### Client (`client.py`)
MCP client for connecting to external MCP servers:
- Manages multiple server connections
- Discovers server capabilities
- Provides unified interface for tool/resource access
- Handles connection lifecycle

## MCP Tools

### Chat Tools

#### `chat_with_llm`
Send messages to Large Language Models.
- **Parameters**:
  - `message`: The message to send
  - `provider`: LLM provider (openai, anthropic, etc.)
  - `model`: Optional model override
  - `system_prompt`: Optional system prompt
  - `temperature`: Generation temperature (0-2)
  - `max_tokens`: Maximum response tokens
  - `conversation_id`: Optional conversation to continue
- **Returns**: Response and conversation ID

#### `chat_with_character`
Have conversations with specific characters.
- **Parameters**:
  - `message`: The message to send
  - `character_id`: ID of the character
  - `provider`: LLM provider
  - `model`: Optional model override
  - `temperature`: Generation temperature
  - `max_tokens`: Maximum response tokens
  - `conversation_id`: Optional conversation to continue
- **Returns**: Response, conversation ID, and character name

### Search Tools

#### `search_rag`
Search the RAG (Retrieval-Augmented Generation) database.
- **Parameters**:
  - `query`: Search query
  - `limit`: Maximum results (default: 10)
  - `media_types`: Optional media type filter
  - `use_semantic`: Use semantic search if available
- **Returns**: List of search results with content and metadata

#### `search_conversations`
Search through conversation history.
- **Parameters**:
  - `query`: Search query
  - `limit`: Maximum results
  - `character_id`: Optional character filter
- **Returns**: List of matching conversations

#### `search_notes`
Search through notes.
- **Parameters**:
  - `query`: Search query
  - `limit`: Maximum results
- **Returns**: List of matching notes with previews

### Content Management Tools

#### `create_note`
Create a new note.
- **Parameters**:
  - `title`: Note title
  - `content`: Note content
  - `tags`: Optional list of tags
  - `template`: Optional template name
- **Returns**: Note ID and creation details

#### `list_characters`
List all available characters.
- **Returns**: List of character profiles with basic info

#### `get_conversation_history`
Retrieve conversation details and messages.
- **Parameters**:
  - `conversation_id`: ID of the conversation
  - `limit`: Optional message limit
- **Returns**: Conversation details and messages

#### `export_conversation`
Export conversations in various formats.
- **Parameters**:
  - `conversation_id`: ID of the conversation
  - `format`: Export format (markdown, json, text)
- **Returns**: Formatted conversation content

#### `ingest_media`
Ingest media from URLs or files (placeholder).
- **Parameters**:
  - `url`: Optional URL to ingest
  - `file_path`: Optional local file path
  - `media_type`: Type of media
  - `title`: Optional title
  - `tags`: Optional tags
- **Returns**: Ingestion status and media ID

## MCP Resources

Resources provide direct access to tldw_chatbook data through URI templates:

### Resource Types

#### `conversation://{id}`
Access individual conversations formatted as markdown.
- Includes conversation metadata
- Shows all messages with roles
- Displays character information if applicable

#### `note://{id}`
Access individual notes.
- Includes title, tags, and timestamps
- Full note content in markdown
- Metadata about creation and updates

#### `character://{id}`
Access character profiles.
- Character description and personality
- Scenario and greeting information
- Example dialogue if available

#### `media://{id}`
Access ingested media content.
- Media metadata (type, source, duration)
- Transcript or content
- Creation timestamp

#### `rag-chunk://{id}`
Access individual RAG chunks.
- Parent media information
- Chunk position (start/end characters)
- Raw chunk text

### Dynamic Resource Listing
The server provides dynamic resource discovery:
- Lists recent conversations
- Lists recent notes
- Configurable limits
- Returns resource metadata

## MCP Prompts

Pre-built prompt templates for common AI workflows:

### Available Prompts

#### `summarize_conversation`
Generate conversation summaries.
- **Parameters**:
  - `conversation_id`: Conversation to summarize
  - `style`: Summary style (concise, detailed, bullet_points, executive)
  - `focus`: Optional focus area (action_items, decisions, technical_details)
- **Returns**: Prompt messages for summarization

#### `generate_document`
Create documents from conversations.
- **Parameters**:
  - `conversation_id`: Source conversation
  - `doc_type`: Document type (summary, report, timeline, study_guide, briefing)
  - `format`: Output format (markdown, html, plain_text)
- **Returns**: Prompt messages for document generation

#### `analyze_media`
Analyze ingested media content.
- **Parameters**:
  - `media_id`: Media to analyze
  - `analysis_type`: Type of analysis (summary, transcript, key_points, themes, sentiment)
  - `detail_level`: Level of detail (brief, medium, comprehensive)
- **Returns**: Prompt messages for analysis

#### `search_and_synthesize`
Search RAG and synthesize results.
- **Parameters**:
  - `query`: Search query
  - `num_sources`: Number of sources to include
  - `synthesis_type`: Type of synthesis (overview, comparison, deep_dive, answer)
- **Returns**: Prompt messages with search results

#### `character_writing`
Character-based creative writing.
- **Parameters**:
  - `character_id`: Character to use
  - `writing_type`: Type of writing (response, story, dialogue, monologue)
  - `context`: Optional context or scenario
  - `style_notes`: Optional style guidelines
- **Returns**: System and user prompts for character writing

## Configuration

MCP is configured through the `[mcp]` section in config.toml:

```toml
[mcp]
# Basic settings
enabled = false  # Enable MCP server functionality
server_name = "tldw_chatbook"
server_version = "0.1.0"
transport = "stdio"  # "stdio" for Claude Desktop, "http" for web
http_port = 3000  # Port for HTTP transport
allowed_clients = ["claude-desktop", "localhost"]

# Feature toggles
expose_tools = true  # Expose tools (chat, search, etc.)
expose_resources = true  # Expose resources (conversations, notes, etc.)
expose_prompts = true  # Expose prompt templates

# Security settings
require_auth = false  # Require authentication (not implemented yet)
rate_limit = 100  # Max requests per minute per client
max_concurrent_requests = 10  # Max concurrent requests

# Tool-specific settings
[mcp.tools]
chat_default_provider = "openai"
chat_default_temperature = 0.7
chat_default_max_tokens = 4096
search_default_limit = 10
enable_media_ingestion = true

# Resource-specific settings
[mcp.resources]
max_list_limit = 100  # Maximum items in list operations
default_list_limit = 10  # Default items in list operations
enable_binary_resources = false  # Allow binary resources

# Prompt-specific settings
[mcp.prompts]
enable_custom_prompts = true  # Allow custom prompts
max_prompt_length = 10000  # Maximum prompt length
```

## Installation and Setup

### Prerequisites
- Python 3.11 or higher
- tldw_chatbook installed
- MCP dependencies

### Installation Steps

1. **Install with MCP support**:
   ```bash
   pip install tldw-chatbook[mcp]
   ```
   Or for development:
   ```bash
   pip install -e ".[mcp]"
   ```

2. **Enable MCP in configuration**:
   Edit `~/.config/tldw_cli/config.toml`:
   ```toml
   [mcp]
   enabled = true
   ```

3. **Configure API keys** (if using chat tools):
   ```toml
   [API]
   openai_api_key = "your-api-key"
   anthropic_api_key = "your-api-key"
   ```

## Running the MCP Server

### Standalone Mode
Run the MCP server directly:
```bash
python -m tldw_chatbook.MCP
```

### Claude Desktop Integration

1. **Install as MCP server**:
   ```bash
   mcp install /path/to/tldw_chatbook/MCP/server.py
   ```

2. **Or add to Claude Desktop config manually**:
   Edit Claude Desktop's MCP config to include:
   ```json
   {
     "mcpServers": {
       "tldw_chatbook": {
         "command": "python",
         "args": ["-m", "tldw_chatbook.MCP"],
         "env": {
           "PYTHONPATH": "/path/to/tldw_chatbook"
         }
       }
     }
   }
   ```

### Development Mode
Use MCP Inspector for testing:
```bash
mcp dev /path/to/tldw_chatbook/MCP/server.py
```

### Verification
Check server is running:
1. Look for log output: "MCP Server 'tldw_chatbook' initialized"
2. In Claude Desktop, check available tools
3. Test with a simple tool call

## MCP Client

The MCP client allows tldw_chatbook to connect to external MCP servers:

### Usage Example
```python
from tldw_chatbook.MCP.client import MCPClient

# Create client
client = MCPClient()

# Connect to server
await client.connect_to_server(
    server_id="my_server",
    command="python",
    args=["-m", "some_mcp_server"]
)

# List available tools
tools = client.get_server_tools("my_server")

# Call a tool
result = await client.call_tool(
    server_id="my_server",
    tool_name="some_tool",
    arguments={"param": "value"}
)

# Disconnect
await client.disconnect_from_server("my_server")
```

### Client Features
- Multiple simultaneous server connections
- Automatic capability discovery
- Unified interface for tools, resources, and prompts
- Connection management and error handling

## Security Considerations

### API Key Protection
- API keys are never exposed through MCP
- Keys are read from config or environment only
- No sensitive data in logs or error messages

### Input Validation
- All tool parameters are validated
- SQL injection prevention via parameterized queries
- Path traversal prevention for file operations

### Access Control
- Client allowlisting via `allowed_clients` config
- Rate limiting per client
- Maximum concurrent request limits

### Data Security
- Database operations use existing tldw_chatbook security
- No direct database access exposed
- Resource access is read-only

### Future Security Enhancements
- Authentication support (OAuth, API keys)
- Encryption for HTTP transport
- Audit logging for all operations
- Role-based access control

## Development Guide

### Adding a New Tool

1. **Define the tool in `tools.py`**:
   ```python
   async def my_new_tool(self, param1: str, param2: int) -> Dict[str, Any]:
       """Tool description."""
       try:
           # Implementation
           result = await some_operation(param1, param2)
           return {"result": result}
       except Exception as e:
           logger.error(f"Error in my_new_tool: {e}")
           return {"error": str(e)}
   ```

2. **Register in `server.py`**:
   ```python
   @self.mcp.tool()
   async def my_new_tool(param1: str, param2: int) -> Dict[str, Any]:
       """User-facing description."""
       return await self.tools.my_new_tool(param1, param2)
   ```

### Adding a New Resource

1. **Define in `resources.py`**:
   ```python
   async def get_my_resource(self, resource_id: str) -> Dict[str, Any]:
       """Get my resource."""
       try:
           # Fetch and format resource
           data = self.db.get_something(resource_id)
           return {
               "uri": f"myresource://{resource_id}",
               "name": data.name,
               "mimeType": "text/markdown",
               "content": format_as_markdown(data)
           }
       except Exception as e:
           logger.error(f"Error: {e}")
           return {"error": str(e)}
   ```

2. **Register in `server.py`**:
   ```python
   @self.mcp.resource("myresource://{resource_id}")
   async def get_my_resource(resource_id: str) -> Dict[str, Any]:
       return await self.resources.get_my_resource(resource_id)
   ```

### Adding a New Prompt

1. **Define in `prompts.py`**:
   ```python
   async def my_prompt(self, param: str) -> List[Dict[str, str]]:
       """Generate my prompt."""
       try:
           # Build prompt
           return [
               {"role": "system", "content": "System instructions"},
               {"role": "user", "content": f"User prompt with {param}"}
           ]
       except Exception as e:
           logger.error(f"Error: {e}")
           return [{"role": "user", "content": f"Error: {str(e)}"}]
   ```

2. **Register in `server.py`**:
   ```python
   @self.mcp.prompt()
   async def my_prompt(param: str) -> List[Dict[str, str]]:
       return await self.prompts.my_prompt(param)
   ```

### Testing
- Use MCP Inspector for interactive testing
- Write unit tests for tool/resource/prompt functions
- Test error cases and edge conditions
- Verify security constraints

## Future Enhancements

### Planned Features
1. **HTTP Transport**: Web-based MCP server support
2. **Media Ingestion**: Full implementation of media ingestion tool
3. **Real-time Updates**: WebSocket support for live updates
4. **Advanced RAG**: Semantic search with embeddings
5. **Batch Operations**: Bulk tools for efficiency

### Potential Additions
1. **Workflow Automation**: Chain multiple tools together
2. **Custom Tool Creation**: UI for creating custom tools
3. **Analytics**: Usage statistics and performance metrics
4. **Caching**: Intelligent caching for repeated operations
5. **Federation**: Connect multiple tldw_chatbook instances

### Integration Ideas
1. **IDE Plugins**: VS Code, IntelliJ MCP extensions
2. **CI/CD Integration**: GitHub Actions, GitLab CI
3. **Monitoring**: Prometheus metrics export
4. **Webhooks**: Event notifications
5. **GraphQL API**: Alternative query interface

## Troubleshooting

### Common Issues

1. **MCP not available error**:
   - Ensure MCP dependencies are installed: `pip install mcp[cli]`
   - Check Python version is 3.11+

2. **Server won't start**:
   - Check config.toml has `mcp.enabled = true`
   - Verify database paths are correct
   - Check logs for initialization errors

3. **Tools not appearing in Claude Desktop**:
   - Restart Claude Desktop after configuration
   - Check server is running (`ps aux | grep MCP`)
   - Verify stdio transport is selected

4. **Database errors**:
   - Ensure databases exist and are accessible
   - Check file permissions
   - Verify disk space available

### Debug Mode
Enable debug logging:
```toml
[logging]
level = "DEBUG"
```

Check logs at: `~/.local/share/tldw_cli/logs/`

## Conclusion

The MCP integration transforms tldw_chatbook into a powerful context provider for AI applications. By exposing its rich functionality through standard MCP protocols, it enables seamless integration with tools like Claude Desktop while maintaining security and modularity. The architecture supports easy extension and customization, making it suitable for both personal use and enterprise deployments.
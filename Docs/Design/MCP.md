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
- **Security**: Internal diagnostics and refusals are payload-free. Authorized
  external MCP clients can read private Library data through enabled tools,
  resources, and prompts and may send that data onward, including to cloud
  models. API keys remain internal.
- **Extensibility**: Easy to add new tools, resources, or prompts
- **Error Handling**: Comprehensive error handling with detailed logging

## Components

### Server (`server.py`)
The main MCP server implementation uses the public `mcp-unified==0.2.1`
gateway:
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

## Standalone stdio contract

Install the packaged optional extra and launch the server from the same Python
environment:

```bash
pip install "tldw_chatbook[mcp]"
python -m tldw_chatbook.MCP
```

The standalone server uses strict JSON-RPC over stdio. It supports legacy
revision `2025-03-26`, revision `2025-11-25`, and the current `2026-07-28`
profile. Batch requests are accepted only with `2025-03-26`; `2025-11-25` and
`2026-07-28` reject them.

### Standalone inventory

The retired `ingest_media` placeholder is absent. Use Library Import for
persistent URL or file ingestion.

- **Built-in tools (9):** `chat_with_llm`, `chat_with_character`, `search_rag`, `search_conversations`, `create_note`, `search_notes`, `list_characters`, `get_conversation_history`, `export_conversation`
- **Resource templates (5):** `conversation://{conversation_id}`, `note://{note_id}`, `character://{character_id}`, `media://{media_id}`, `rag-chunk://{chunk_uuid}`
- **Prompts (5):** `summarize_conversation`, `generate_document`, `analyze_media`, `search_and_synthesize`, `character_writing`
- **Library tools excluded from standalone (24):** `library_list_media`, `library_get_media`, `library_search_media`, `library_get_media_structure`, `library_get_media_chunk`, `library_list_chunk_specs`, `library_save_chunk_spec`, `library_rechunk_media`, `library_list_notes`, `library_get_note`, `library_search_notes`, `library_save_note`, `library_list_prompts`, `library_get_prompt`, `library_search_prompts`, `library_list_skills`, `library_get_skill`, `library_search_skills`, `library_list_conversations`, `library_get_conversation`, `library_search_conversations`, `library_list_collections`, `library_get_collection`, `library_search_collections`

### Standalone behavior and controls

All 24 Library tools are excluded from the standalone stdio catalog. They
remain available only through the app's gated, logged direct Library execution
path, whose raw in-app `tools/call` route is refused.

Resource reads return at most 256 KiB of UTF-8 text at a time. Follow the
opaque `nextUri` in `_meta["tldw.chatbook/continuation"]`; handler metadata,
when present, is namespaced under `_meta["tldw.chatbook/resource"]`.

Workspace-local filesystem, git, and web tools are off by default with
`[mcp] expose_local_tools = false`. When enabled, they retain workspace
confinement, consult the shared `mcp_permissions.json` permission store on
each call, and honor its kill switch. An external `ask` state is refused
because an stdio client cannot display Chatbook's operator approval card.

> [!WARNING]
> An external MCP client runs with the user's OS access. It can read private local Library content through exposed tools, resources, and prompts, and it may send that content off-device to a cloud model. Enable only the surface you intend to disclose and trust the client and its model provider.

## Portable Tool Profiles

Chatbook can export and import named tool permission profiles as deterministic
`.tldw-tool-pack` V1 archives. These are policy packages, not MCP server bundles.
They contain one flattened set of exact Allow/Ask/Deny rules, Ask/Deny
future-tool fallbacks, stable permission identities, and contract fingerprints.
The fingerprints cover raw tool name, description, input schema, and
policy-relevant risk tags without disclosing description or schema text.

The portability inventory is fail-closed and code-owned. It covers the
permission-addressable built-in agent tools, the built-in Chatbook MCP tools,
local/raw-shell/Virtual CLI tools, and local external MCP definitions available
live or from validated cache. Display-only server-source tools, Library
capabilities, managed-skill approval, runtime orchestration, and skills are
explicitly counted as excluded. Adding a permission namespace without an
inventory adapter or explicit exclusion blocks export.

Export reads one strict permission snapshot and one complete inventory. The
review is bound to the exact profile policy digest and, for imported profiles,
its lifecycle revision. Named-profile inheritance, definition-change downgrade,
and high-risk floors are resolved before serialization. Broad Allow fallbacks
are clamped to Ask. The global kill switch, runtime availability, connection
configuration, workspace/Persona gates, and project-instruction state remain
destination-local and are never serialized.

Import first validates a canonical two-member archive, then performs a
side-effect-free review against one strict local policy snapshot and one
inventory snapshot. Exact automatic matching requires authority, server key,
raw tool name, and contract fingerprint. Mapping one external MCP server to
another is manual, one-to-one, and shown in the review. A reviewed import creates
an unbound named profile only; it does not connect a server, install a tool, or
change any active workspace. Changed or missing Allow/Ask entries are omitted,
while safe Deny entries may remain pending.

The first bind is a separate workspace transaction and requires confirmation of
the exact workspace, defaults, profile revision/digest, and effective posture.
All later resolution, persistent decisions, session approvals, and tool tests
remain scoped to the captured profile. Individual policy rules are edited only
in MCP Permissions; Settings owns lifecycle actions and deep-links to that
editor.

Import receipts are private bounded evidence, never policy authority. Missing
receipt data degrades provenance but does not weaken the stored policy or clear
first-bind confirmation. Removal is allowed only for a valid imported profile
with no active/archived workspace reference and no runtime lease, and leaves a
hidden permanent-Deny tombstone. Publication, activation, binding, and removal
use stable path-free failure categories and report uncertain outcomes rather
than guessing after an ambiguous filesystem or store replacement.

Native Windows archive publication is a separate capability and currently
returns `tool_pack.export.publication_unsupported` when the required safe
publication primitives are unavailable. It is not represented as a schema or
import incompatibility.

V1 rejects executable, skill, plugin, connection, credential, command,
environment, endpoint, approval, receipt, workspace, Persona, and runtime-install
fields. A combined Tools+Skills pack or plugin installer is future work and
requires a new schema, ADR, provenance/signature policy, dependency model,
permission review, and explicit installation UX. V1 readers must never treat
unknown composition fields as installable content. See
[ADR-107](../../backlog/decisions/107-portable-tool-use-packs.md) for the complete
trust-boundary decision.

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

`use_semantic` remains a boolean compatibility switch: `false` forces media
keyword search; `true` or omission follows the active RAG profile's `plain`,
`semantic`, or `hybrid` search mode.

- **Parameters**:
  - `query`: Search query
  - `limit`: Maximum results (default: 10)
  - `media_types`: Optional media type filter
  - `use_semantic`: Enable profile-driven RAG search (default: `true`)
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

### Library Tools (read-only, descriptor-backed)

In addition to the standalone tools above, the in-process local MCP surface
exposes 18 read-only `library_*` tools — `library_list_*`, `library_get_*`, and
`library_search_*` for each of Media, Notes, Prompts, Skills, Conversations,
and Collections. The same shared service and all 18 tools are also callable by
Console agents when that conversation allows assistant Library access and its
**Direct / RAG selector** chooses Direct. They answer
factual Library questions (list, count, view, lexical search) without touching
the RAG/embedding pipeline.

- **Registration**: appended to the capability manifest from the descriptor
  table in `Library/library_tool_contract.py`
  (`_describe_local_library_tools` in `server.py`); dispatched in-process by
  `LocalMCPRuntimeDelegate` to the shared synchronous service. The standalone
  server deliberately does not consume this combined manifest.
- **Semantics**: bounded pages with exact totals; opaque stable IDs
  (`type:<base64url>`); literal keyword-only search (no semantic/embedding);
  get tools read bounded windows with revision-checked continuation cursors;
  every serialized response fits within 32 KiB.
- **Compatibility**: the standalone tools above are unchanged; the `library_*`
  namespace is additive and **independent of Console's per-conversation
  Library policy**. MCP registration and permissions remain authoritative for
  MCP calls.

See `Docs/Development/Agent-Tools/local-library-tools.md` for the full
contract (exact names, pagination, continuation, error codes, and security
boundaries).

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

The standalone gateway is stdio-only. The only `[mcp]` configuration key
consumed by the standalone gateway is `expose_local_tools`. Its default is:

```toml
[mcp]
expose_local_tools = false
```

The shipped entry point does not open an HTTP listener and does not implement
standalone authentication or client allowlisting. `mcp-unified` applies its
fixed `GatewayLimits` defaults: 600 requests per minute and 16 in-flight
requests. Those limits are not Chatbook config keys. When local tools are
enabled, workspace confinement comes from `[console] workspace_root` and each
call still uses the shared permission store and kill switch described above.
Confinement is not the only path check: credential, permission-store and
app-database paths (`Utils/sensitive_paths.py`) are refused regardless of the
configured root, enforced for these tools inside
`Tools/local_tool_impls.py`'s `resolve_workspace_path`.

## Installation and Setup

### Prerequisites
- Python 3.11 or higher
- tldw_chatbook installed
- MCP dependencies

### Installation Steps

1. **Install with MCP support**:
   ```bash
   pip install "tldw_chatbook[mcp]"
   ```
   Or for development:
   ```bash
   pip install -e ".[mcp]"
   ```

2. **Configure API keys** (if using chat tools):
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

1. **Add the packaged module command to the client configuration**:
   Edit Claude Desktop's MCP config to include:
   ```json
   {
     "mcpServers": {
       "tldw_chatbook": {
         "command": "python",
         "args": ["-m", "tldw_chatbook.MCP"]
       }
     }
   }
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
- The standalone server inherits the launching client's OS access over stdio;
  it has no network authentication or client-allowlisting layer.
- The gateway enforces its fixed request-rate and in-flight limits.
- Enabled local tools remain permission-gated and workspace-confined.

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
   - Ensure MCP dependencies are installed: `pip install "tldw_chatbook[mcp]"`
   - Check Python version is 3.11+

2. **Server won't start**:
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

# Release Recovery And Setup Guide

This guide is the release-candidate recovery reference for setup states that can block core Chatbook workflows. It mirrors the current running app labels and keeps recovery guidance source-honest.

## Local-first baseline

Chatbook's baseline install is local-first. A source checkout installed with `pip install -e .`, or a packaged install with `pip install tldw_chatbook`, should still provide the shell, Home, Console, local conversations, notes, personas, Library browsing, Chatbook artifacts, and Settings.

Missing optional features do not mean Chatbook is broken. They mean the user has opened an advanced capability that requires an optional dependency group. Recovery copy must name the missing group, identify the owning destination, and provide a safe install command without implying that local-first core workflows are unavailable.

## Advanced optional capability groups

| Capability area | Optional group | Source install | Packaged install | Owner |
| --- | --- | --- | --- | --- |
| RAG and retrieval | `embeddings_rag` | `pip install -e ".[embeddings_rag]"` | `pip install "tldw_chatbook[embeddings_rag]"` | Library Search/RAG |
| Media ingestion and transcription | `audio`, `video`, `pdf`, `ebook` | `pip install -e ".[audio,video,pdf,ebook]"` | `pip install "tldw_chatbook[audio,video,pdf,ebook]"` | Library import/media |
| MCP integration | `mcp` | `pip install -e ".[mcp]"` | `pip install "tldw_chatbook[mcp]"` | MCP destination |
| Local inference | `local_vllm`, `local_mlx`, `local_transformers` | `pip install -e ".[local_vllm]"` | `pip install "tldw_chatbook[local_vllm]"` | Console/provider setup |
| Web access | `web` | `pip install -e ".[web]"` | `pip install "tldw_chatbook[web]"` | Web/browser serving |

## Recovery Matrix

| Blocker | Visible State | Recovery | Documentation |
| --- | --- | --- | --- |
| Provider/model setup | Home shows `Model: Blocked`; Console shows provider setup copy before live model work. | Configure a provider and model in Settings. For OpenAI, set `OPENAI_API_KEY` or add the provider key under the OpenAI API settings. | See README `Configuration` and `Environment Variables`. |
| Server/local mode | Home shows server sync status such as `Configured; local mode`; server-backed work can show reconnect/auth recovery when active server state requires it. | Local mode is usable without a server. For server-backed work, configure/select the active server and reconnect or authenticate when prompted. | See README `Configuration File` and `Web Server Access`. |
| ACP runtime setup | ACP shows `Runtime not configured`, `Why: no ACP-compatible runtime is configured.`, and owner `ACP runtime`. | Configure an ACP-compatible runtime in ACP, then start or resume an ACP session before launching or following agent work. | ACP runtime setup remains owned by the ACP destination, not global Settings. |
| MCP server management | Console shows `MCP: Not wired - Manage servers in MCP.`; MCP shows server/scope inventory and prompts users to inspect runnable tools. | Open MCP to manage servers, scoped tools, permissions, and audit readiness. Install optional MCP support with `pip install -e ".[mcp]"` when MCP dependencies are needed. | See README `Model Context Protocol (MCP) Integration`. |
| Optional dependency recovery | Disabled feature states name the missing optional dependency group and owner. | Install the matching optional extra, such as `pip install -e ".[embeddings_rag]"` or `pip install "tldw_chatbook[embeddings_rag]"` for Search/RAG dependencies, or the relevant audio/video/transcription extras for media workflows. | See README `Installation with Optional Features` and `Optional Feature Groups`. |
| Missing-source recovery | Home and Console show `RAG: Missing sources`, `RAG/source: not staged`, or Library source-selection recovery copy. | Add/import Library content, select a source or RAG result, then stage it into Console before asking grounded questions. | Use Library `Sources`, `Search/RAG`, and `Import/Export Sources` modes. |

## Setup Commands

Core install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 -m tldw_chatbook.app
```

Common optional features:

```bash
pip install -e ".[embeddings_rag]"
pip install "tldw_chatbook[embeddings_rag]"
pip install -e ".[mcp]"
pip install -e ".[web]"
```

## Release Notes

- Keep local and server states visually distinct. Local mode is valid and does not imply server-backed work is ready.
- Do not route ACP runtime setup to global Settings; ACP owns ACP runtime setup.
- Do not route MCP server/tool management to Settings; MCP owns MCP server management.
- When a feature is blocked by missing dependencies, name the optional dependency group and show the owner instead of presenting a dead control.
- Do not use machine-specific absolute paths in release verification commands. Prefer `python -m pytest ...` with the virtual environment already active.

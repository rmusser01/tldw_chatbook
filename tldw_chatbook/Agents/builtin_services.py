# tldw_chatbook/Agents/builtin_services.py
"""Local service seams injected into built-in pack tools.

A separate module so ``tool_catalog`` stays dependency-light: importing the
provider must not drag in the notes, media, or RAG subsystems.

**Contract.** Every service assigned here MUST be:

1. thread-safe -- tools execute on a fresh per-call daemon thread
   (``agent_service._call_with_timeout``), so a service holding
   non-thread-safe state will corrupt under concurrent tool calls;
2. free of event-loop-bound state -- no ``httpx.AsyncClient`` or other
   object bound to the app's loop, because ``BuiltinToolProvider.invoke``
   drives async tools through ``asyncio.run`` on that fresh thread;
3. free of Textual/UI handles -- a worker thread must never touch widgets.

Violations surface as failures that are miserable to diagnose from a
worker thread, which is why the contract is stated rather than implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BuiltinToolServices:
    """Per-run handles the built-in packs operate against.

    All fields default to ``None`` so a bare instance is valid: metadata
    enumeration (TASK-656) constructs tools with ``services=None`` and only
    reads their name/description/parameters/risk_tags.

    Attributes:
        notes_library: ``Notes/Notes_Library.py`` handle.
        media_reading: ``Media/local_media_reading_service.py`` handle.
        prompt_service: ``Prompt_Management/local_prompt_service.py`` handle.
        chunk_service: ``Chunking`` entry point.
        rag_search: RAG search entry point resolved for the active profile.
    """

    notes_library: Any | None = None
    media_reading: Any | None = None
    prompt_service: Any | None = None
    chunk_service: Any | None = None
    rag_search: Any | None = None

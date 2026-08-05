"""Provider-neutral realtime voice engine package.

`import tldw_chatbook.LLM_Calls.realtime` must stay cheap: this module only
re-exports the pure-dataclass/typing shapes from `.protocol`. It must NOT
import `websockets` (or any provider transport) at package-import time --
those live in transport/session modules (e.g. a future `openai_transport.py`
or `openai_session.py`) that import `websockets` lazily, inside their own
module bodies, and are only imported when a caller actually constructs a
session. This keeps a cold `import tldw_chatbook.LLM_Calls.realtime` cheap
even when the caller never opens a realtime connection (e.g. hands-free
loop stays on the pipeline engine, or the `realtime` extra isn't installed).
"""

from __future__ import annotations

from .protocol import RealtimeCallbacks, RealtimeSession, RealtimeSessionConfig

__all__ = [
    "RealtimeCallbacks",
    "RealtimeSession",
    "RealtimeSessionConfig",
]

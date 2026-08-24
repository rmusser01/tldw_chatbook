"""Import-closure guards for the RAG stack on the boot path (TASK-21731).

Two eager module-scope imports put ~67 of this repo's modules -- the whole
``RAG_Search.simplified`` service tree, the ~15k-LOC ``Chunking`` engine it
drags through ``chunking_service``, and ``Internal_Prompts`` -- in front of
the user, twice over:

* ``Library/library_local_rag_search_service.py`` imported
  ``normalize_rag_search_mode`` from ``simplified.active_config``. That
  module is on ``import tldw_chatbook.app``'s path, so the whole tree
  executed before a single pixel was painted (703 own modules, up from 636).
* ``Event_Handlers/Chat_Events/chat_rag_events.py`` ran a module-scope
  ``try: from ...RAG_Search.simplified import ...`` feature probe. That
  module is imported during the initial **Chat screen mount** (via
  ``UI/Console_Modules/retrieval.py``), on the event loop -- so removing the
  first chain alone would only have MOVED the cost from the import phase to
  the mount phase (measured: 50 ms), leaving time-to-interactive unchanged.

Both are guarded here, because a fix for either one alone reads as a win on
the module count while the user waits exactly as long.

Subprocess-isolated for the same reason as
``test_chunking_import_closure.py`` (TASK-21102) and
``test_extras_import_closure.py`` (TASK-21104): ``sys.modules`` is
process-global, so an earlier test in the session that legitimately imported
the RAG stack would make an in-process check pass or fail for reasons that
have nothing to do with the boot path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

#: The three package prefixes this task removed from the boot path.
DEFERRED_PREFIXES = (
    "tldw_chatbook.RAG_Search.simplified",
    "tldw_chatbook.Chunking",
    "tldw_chatbook.Internal_Prompts",
)


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter with isolated config/data dirs.

    Args:
        tmp_path: Per-test scratch directory for the subprocess's HOME/XDG so
            the app import can never read or write the live user config.
        code: The Python source to execute with ``python -c``.

    Returns:
        The completed process (never raises on nonzero exit).
    """
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )


_RESIDENT_HELPER = """
import sys

DEFERRED_PREFIXES = {prefixes!r}


def resident():
    out = []
    for name in sorted(sys.modules):
        if sys.modules[name] is None:
            continue
        for prefix in DEFERRED_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                out.append(name)
                break
    return out
""".format(prefixes=DEFERRED_PREFIXES)


_APP_IMPORT_SNIPPET = _RESIDENT_HELPER + """
import tldw_chatbook.app  # noqa: F401

loaded = resident()
assert not loaded, f"deferred packages resident after app import: {loaded}"

# Anti-vacuity: the converted boot-path module must still BE in the closure
# (this guard is about what it pulls, not about whether it is reached), and
# the stdlib-only replacement for the mode vocabulary must be what got there
# in place of active_config. `chat_rag_events` is deliberately absent from
# this list -- it is a Chat-MOUNT module, not a boot module, and has its own
# test below.
for expected in (
    "tldw_chatbook.Library.library_local_rag_search_service",
    "tldw_chatbook.RAG_Search.search_modes",
):
    assert expected in sys.modules, f"expected closure member missing: {expected}"

print("APP_CLOSURE_OK")
"""


def test_app_import_does_not_execute_the_deferred_rag_packages(
    tmp_path: Path,
) -> None:
    """`import tldw_chatbook.app` resolves none of the three packages.

    The regression this pins measured 703 ``tldw_chatbook.*`` modules at app
    import (Chunking 33 + RAG_Search.simplified 24 + Internal_Prompts 10 over
    a 636 baseline).

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _APP_IMPORT_SNIPPET)
    assert result.returncode == 0, (
        "import tldw_chatbook.app must not execute the deferred RAG packages:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "APP_CLOSURE_OK" in result.stdout


_CHAT_RAG_EVENTS_SNIPPET = _RESIDENT_HELPER + """
from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events

loaded = resident()
assert not loaded, (
    f"importing chat_rag_events must not execute the RAG service tree: {loaded}"
)

# ...and the deferred probe must still RESOLVE when something asks. A guard
# that only proves absence would be satisfied by deleting the feature.
assert chat_rag_events._rag_services_available() is True, (
    "the deferred RAG availability probe did not resolve"
)
assert "tldw_chatbook.RAG_Search.simplified" in sys.modules, (
    "asking for RAG availability did not import the simplified package"
)

# The public module attribute keeps working (PEP 562 module __getattr__).
assert chat_rag_events.RAG_SERVICES_AVAILABLE is True

# ...and the real constructors it probes for are importable through it.
from tldw_chatbook.RAG_Search.simplified import (  # noqa: E402
    create_rag_service,
    create_config_for_collection,
)

assert callable(create_rag_service) and callable(create_config_for_collection)
print("CHAT_RAG_EVENTS_LAZY_OK")
"""


def test_chat_rag_events_import_is_lazy_but_the_probe_still_resolves(
    tmp_path: Path,
) -> None:
    """The Chat-mount module must not execute the RAG tree -- and must still work.

    ``chat_rag_events`` is imported while the default Chat screen mounts, on
    the event loop, before the app is interactive. Its availability probe now
    resolves on first ask. Both halves are asserted in one subprocess: the
    import is cheap, and asking the question still returns ``True`` and
    genuinely loads ``RAG_Search.simplified``.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _CHAT_RAG_EVENTS_SNIPPET)
    assert result.returncode == 0, (
        "chat_rag_events import/probe contract failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "CHAT_RAG_EVENTS_LAZY_OK" in result.stdout


_SEARCH_MODES_SNIPPET = _RESIDENT_HELPER + """
from tldw_chatbook.RAG_Search import search_modes

loaded = resident()
assert not loaded, f"search_modes must be import-cheap, pulled: {loaded}"

assert search_modes.normalize_rag_search_mode("hybrid") == "hybrid"
assert search_modes.normalize_rag_search_mode("nonsense") == "semantic"

# One object, not a copy: active_config re-imports from here, so the
# vocabulary the Library service normalizes against cannot drift from the
# one the engine's own config validates against.
from tldw_chatbook.RAG_Search.simplified import active_config  # noqa: E402
from tldw_chatbook.Library import library_local_rag_search_service as lib  # noqa: E402

assert (
    active_config.normalize_rag_search_mode
    is search_modes.normalize_rag_search_mode
), "active_config holds a different normalize_rag_search_mode object"
assert (
    lib.normalize_rag_search_mode is search_modes.normalize_rag_search_mode
), "the Library service holds a different normalize_rag_search_mode object"
assert active_config._RAG_SEARCH_MODES is search_modes.RAG_SEARCH_MODES

from tldw_chatbook.RAG_Search.simplified.search_service import (  # noqa: E402
    normalize_rag_search_mode as search_service_normalize,
)

assert search_service_normalize is search_modes.normalize_rag_search_mode
print("SEARCH_MODES_OK")
"""


_PROFILE_RAG_FROM_DEFERRED_STATE_SNIPPET = _RESIDENT_HELPER + """
import asyncio
from types import SimpleNamespace

import tldw_chatbook.app  # noqa: F401

assert not resident(), resident()

# --- Leg 1: the Library profile-mode resolution PR #2049 shipped -----------
# It must answer correctly WITHOUT the simplified stack -- that is the whole
# point of the extraction, and a fix that merely moved the import would show
# up here as a resident stack.
from tldw_chatbook.Library import library_local_rag_search_service as lib  # noqa: E402


def _runtime(mode):
    return SimpleNamespace(
        config=SimpleNamespace(search=SimpleNamespace(default_search_mode=mode))
    )


assert lib._resolve_profile_search_mode(_runtime("hybrid")) == "hybrid"
assert lib._resolve_profile_search_mode(_runtime("plain")) == "plain"
assert lib._resolve_profile_search_mode(_runtime("semantic")) == "semantic"
assert lib._resolve_profile_search_mode(_runtime("nonsense")) == "semantic"
assert lib._resolve_profile_search_mode(SimpleNamespace()) == "semantic"
assert not resident(), f"profile-mode resolution pulled the stack: {resident()}"

# --- Leg 2: the MCP profile-driven search PR #2049 shipped -----------------
# Importing it on demand must WORK, not just be absent at boot.
from tldw_chatbook.RAG_Search.simplified.search_service import (  # noqa: E402
    SimplifiedRAGSearchService,
)
import tldw_chatbook.RAG_Search.simplified.search_service as search_service  # noqa: E402

assert "tldw_chatbook.RAG_Search.simplified" in sys.modules
assert any(m.startswith("tldw_chatbook.Chunking") for m in sys.modules), (
    "the deferred stack did not actually load on demand"
)


class _StubRuntime:
    def __init__(self):
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return [
            SimpleNamespace(
                id="media-1",
                document="body",
                score=0.5,
                metadata={"title": "T", "media_type": "video"},
            )
        ]


runtime = _StubRuntime()
service = SimplifiedRAGSearchService(media_db=None)
service.rag_service = runtime
search_service.resolve_active_rag_search_mode = lambda: "hybrid"

rows = asyncio.run(service.profile_search("credential", limit=3))

assert [c["search_type"] for c in runtime.calls] == ["hybrid"], runtime.calls
assert runtime.calls[0]["top_k"] == 3
assert [r["id"] for r in rows] == ["media-1"], rows
assert rows[0]["media_type"] == "video"
print("PROFILE_RAG_FROM_DEFERRED_STATE_OK")
"""


def test_profile_driven_rag_still_works_from_the_deferred_boot_state(
    tmp_path: Path,
) -> None:
    """PR #2049's features must still work once their imports are deferred.

    Absence is not the deliverable -- a deleted feature would satisfy a
    pure closure guard. This drives both legs of the MCP profile-driven RAG
    search from exactly the state this task creates (app imported, nothing
    RAG resolved): the Library profile-mode resolution answers correctly
    without loading anything, and the MCP ``profile_search`` seam loads the
    stack on demand and routes by the active mode.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _PROFILE_RAG_FROM_DEFERRED_STATE_SNIPPET)
    assert result.returncode == 0, (
        "profile-driven RAG failed from the deferred state:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "PROFILE_RAG_FROM_DEFERRED_STATE_OK" in result.stdout


_MISSING_EXTRAS_SNIPPET = """
import asyncio
import sys
from types import SimpleNamespace

BLOCKED = "tldw_chatbook.RAG_Search.simplified"


class _BlockSimplified:
    \"\"\"Meta-path finder simulating an install without the RAG extras.\"\"\"

    def find_spec(self, name, path=None, target=None):
        if name == BLOCKED or name.startswith(BLOCKED + "."):
            raise ImportError(f"simulated missing RAG dependency for {name}")
        return None


sys.meta_path.insert(0, _BlockSimplified())

from loguru import logger

warnings = []
logger.add(lambda message: warnings.append(str(message)), level="WARNING")

from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events

# The import itself must survive an install with no RAG extras...
assert chat_rag_events._rag_services_available() is False
assert chat_rag_events.RAG_SERVICES_AVAILABLE is False
assert any("RAG services not available" in w for w in warnings), warnings

# ...and the consumer degrades the way it always did: None, not a crash.
service = asyncio.run(
    chat_rag_events.get_or_initialize_rag_service(SimpleNamespace(config={}))
)
assert service is None, service

# Cached: a second ask must not re-attempt the import (one warning, not two).
assert chat_rag_events._rag_services_available() is False
assert sum("RAG services not available" in w for w in warnings) == 1, warnings
print("MISSING_EXTRAS_OK")
"""


def test_missing_rag_extras_degrade_at_first_use_exactly_as_they_did_at_import(
    tmp_path: Path,
) -> None:
    """Moving an import moves its failure -- this pins where that failure lands.

    With ``RAG_Search.simplified`` unimportable (the plain-install case), the
    probe used to fail at ``chat_rag_events`` import time and set the flag
    False. It now fails at the first ask. The observable contract is
    unchanged: one warning, a ``False`` flag, and ``None`` from
    ``get_or_initialize_rag_service`` -- never an exception escaping into the
    Chat screen.

    Runs in a subprocess with a meta-path blocker because ``sys.modules`` is
    process-global: an earlier test that legitimately imported the RAG stack
    would give this a false pass (the aiohttp precedent in
    ``lessons-testing-evidence.md``).

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _MISSING_EXTRAS_SNIPPET)
    assert result.returncode == 0, (
        "missing-extras degradation contract failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "MISSING_EXTRAS_OK" in result.stdout


def test_search_modes_is_import_cheap_and_single_sourced(tmp_path: Path) -> None:
    """The extracted vocabulary must cost nothing and stay one object.

    ``chunking_engine_version.py`` (TASK-21102) is the exemplar: a pure value
    a boot-path module needs, lifted out of the heavy module that owned it,
    and re-imported by that module so no second copy can drift. Both
    properties are checked here, plus the two consumers that read it.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _SEARCH_MODES_SNIPPET)
    assert result.returncode == 0, (
        f"search_modes contract failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "SEARCH_MODES_OK" in result.stdout

"""Import-closure guard: `Chat_Functions` reaches `ChatPersistenceService` lazily.

TASK-23112 / ADR-097. ``Chat/chat_persistence_service.py`` grew by ~900 lines
and picked up module-scope imports of ``Chat.attachment_core``,
``Chat.console_chat_fork``, ``Chat.library_activity`` and
``Video_Generation.video_metadata``. It rode onto the boot path through one
module-scope line in ``Chat/Chat_Functions.py``, which ``app.py`` reaches via
``Library.library_local_rag_search_service -> library_rag_service ->
library_rag_state -> library_rag_answer_service``. Measured with an
import-parent tracer, that single edge put **18** ``tldw_chatbook`` modules in
the ``import tldw_chatbook.app`` closure (666 -> 648 own modules when
deferred), breaching the 660 ratchet.

``ChatPersistenceService`` is constructed in exactly one place in
``Chat_Functions`` -- ``save_chat_history_to_db_wrapper`` -- and that function
is only ever called from a user-driven save, never at import time and never
during ``TldwCli.__init__``. (The TASK-22223 trap -- "a function-scope import
is not lazy if the function runs at module import" -- is what the residency
assertion below actually rules out; the reasoning alone is not evidence.)

Scope note, so this guard is not read as more than it is: this is a removal
from the IMPORT closure, not from the whole boot. ``Chat.console_runtime``
imports the same service inside ``ensure_console_runtime``, so it is resident
again by the time the app reaches ``_ui_ready`` (it is in
``Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt`` both before
and after this change). What moved is the eager import; what this file keeps
visible is future drift back onto the import path.

Subprocess-isolated for the same reason as ``test_app_import_diet_closure.py``
(TASK-21108), whose pattern this file follows: ``sys.modules`` is
process-global, so an earlier test in the session that legitimately imported
the service would false-fail an in-process check.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# The modules the deferred edge kept off the boot import path, as measured by
# the import-parent tracer (baseline 666 -> 648). Each was FIRST imported via
# `Chat_Functions -> chat_persistence_service`; nothing else on the boot path
# reaches them.
DEFERRED_BOOT_MODULES = (
    "tldw_chatbook.Chat.chat_persistence_service",
    "tldw_chatbook.Chat.attachment_core",
    "tldw_chatbook.Chat.console_chat_fork",
    "tldw_chatbook.Chat.console_context_policy",
    "tldw_chatbook.Chat.console_context_repository",
    "tldw_chatbook.Chat.console_dispatch_checkpoint",
    "tldw_chatbook.Chat.console_dispatch_repository",
    "tldw_chatbook.Chat.console_library_policy_repository",
    "tldw_chatbook.Chat.console_prefill",
    "tldw_chatbook.Chat.console_roleplay_identity",
    "tldw_chatbook.Chat.console_roleplay_metadata",
    "tldw_chatbook.Chat.message_metadata",
    "tldw_chatbook.Event_Handlers.Chat_Events",
    "tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events",
    "tldw_chatbook.Utils.file_handlers",
    "tldw_chatbook.Video_Generation",
    "tldw_chatbook.Video_Generation.video_formats",
    "tldw_chatbook.Video_Generation.video_metadata",
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


_APP_CLOSURE_SNIPPET = f"""
import sys

import tldw_chatbook.app  # noqa: F401

forbidden = {DEFERRED_BOOT_MODULES!r}
resident = sorted(m for m in forbidden if sys.modules.get(m) is not None)
assert not resident, (
    "chat_persistence_service subtree resident after `import tldw_chatbook.app`: "
    + repr(resident)
    + " -- something took a module-scope dependency on it again. Defer it (see "
    "Chat/Chat_Functions.py:save_chat_history_to_db_wrapper) rather than raising "
    "MAX_TLDW_MODULE_COUNT (ADR-097)."
)

# Anti-vacuity. If Chat_Functions itself left the boot closure, every
# assertion above would pass without testing the deferral at all.
for expected in (
    "tldw_chatbook.Chat.Chat_Functions",
    # The seam Chat_Functions still imports eagerly from the same package, so
    # "the Chat package is simply absent" cannot masquerade as a pass.
    "tldw_chatbook.Chat.console_project_instructions",
):
    assert expected in sys.modules, "expected closure member missing: " + expected

print("CHAT_PERSISTENCE_CLOSURE_OK")
"""


_REAL_USE_PATH_SNIPPET = """
import shutil
import sys
import tempfile
from pathlib import Path

import tldw_chatbook.Chat.Chat_Functions as chat_functions

assert "tldw_chatbook.Chat.chat_persistence_service" not in sys.modules, (
    "importing Chat_Functions still executes chat_persistence_service"
)
assert not hasattr(chat_functions, "ChatPersistenceService"), (
    "ChatPersistenceService is bound at Chat_Functions module scope again"
)

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

scratch = Path(tempfile.mkdtemp(prefix="tldw_persistence_closure_"))
db = CharactersRAGDB(scratch / "chachanotes.sqlite", "task_23112_closure_client")
try:
    conversation_id, status = chat_functions.save_chat_history_to_db_wrapper(
        db=db,
        chatbot_history=[
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "General Kenobi"},
        ],
        conversation_id=None,
        media_content_for_char_assoc=None,
        character_name_for_chat=None,
    )
    assert conversation_id is not None, "real save path returned no conversation: " + status
    assert "success" in status.lower(), status
    messages = db.get_messages_for_conversation(conversation_id)
    assert len(messages) == 2, messages
finally:
    db.close_connection()
    shutil.rmtree(scratch, ignore_errors=True)

# The deferred import fired on the real use path, and resolved to the real
# module (not a stub or a shadowed name).
module = sys.modules.get("tldw_chatbook.Chat.chat_persistence_service")
assert module is not None, (
    "save_chat_history_to_db_wrapper completed without importing "
    "chat_persistence_service -- the deferred import is disconnected"
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

assert ChatPersistenceService is module.ChatPersistenceService

print("CHAT_PERSISTENCE_USE_PATH_OK")
"""


def test_app_import_does_not_execute_chat_persistence_service(tmp_path: Path) -> None:
    """`import tldw_chatbook.app` must not pull the persistence service subtree.

    Regression guard for the TASK-23112 breach: before the fix, all 18 modules
    listed in ``DEFERRED_BOOT_MODULES`` were resident after a plain app import,
    which is what took the own-module count from 648 to 666 against the 660
    ratchet.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _APP_CLOSURE_SNIPPET)
    assert result.returncode == 0, (
        "chat_persistence_service must stay off the app import closure:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "CHAT_PERSISTENCE_CLOSURE_OK" in result.stdout


def test_deferred_persistence_service_resolves_on_the_real_save_path(
    tmp_path: Path,
) -> None:
    """The deferred import still resolves where it is actually used.

    Drives ``save_chat_history_to_db_wrapper`` against a real
    ``CharactersRAGDB`` in a fresh interpreter, asserting the conversation and
    both messages land AND that the deferred module became resident as a
    result. A deferral that silently stopped resolving (renamed module,
    shadowed name, cycle) would fail here rather than at 3am.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _REAL_USE_PATH_SNIPPET)
    assert result.returncode == 0, (
        "the deferred ChatPersistenceService import failed on its real use path:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "CHAT_PERSISTENCE_USE_PATH_OK" in result.stdout

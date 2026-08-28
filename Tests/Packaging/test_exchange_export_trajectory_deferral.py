"""Per-edge guards for the exchange-export / trajectory-engine split (TASK-23020).

TASK-22213 took the trajectory family (`Chat/trajectory_export.py` and its
siblings, ~4,400 LOC) off the Chat first-paint leg. ~24 hours later #2126 put
`trajectory_export` back, through THREE module-scope edges the leg-wide
guards could only report as "something re-eagered it":

* ``Chat/console_exchange_export.py`` -> ``Chat.trajectory_export``
  (one name: ``TraceExportProfile``, a three-member enum)
* ``Widgets/Console/console_exchange_export_dialog.py`` ->
  ``Chat.trajectory_export`` (the same single name)
* ``Widgets/Console/console_exchange_export_dialog.py`` ->
  ``Widgets/Console/trace_export_dialog.py`` (three presentation names),
  whose module scope imports the whole exporter plus ``Chat.trajectory``

All three ride ``console_conversation_inspector`` ->
``UI/Screens/chat_screen.py`` onto the first paint. The fix moved the shared
vocabulary into two light leaves -- ``Chat/trace_export_profiles.py`` (the
enum, stdlib-only) and ``Widgets/Console/trace_export_profile_ui.py`` (copy,
labels, Full confirmation) -- re-imported by the heavy side so nothing can
drift.

This file guards each edge INDIVIDUALLY, so a red here names the offending
file instead of leaving the next person to re-trace the import graph the
way the leg-wide nets (``test_rag_boot_import_closure.py``, the `_ui_ready`
census) do. Subprocess-isolated for the standard reason: ``sys.modules`` is
process-global, and half this suite legitimately imports the trajectory
stack.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The deferred trajectory engine plus the dialog that eagerly imports it.
#: None of these may be resident after importing any chat-leg exchange-export
#: module. `Chat.trajectory` is NOT here -- it is a legitimate chat-leg
#: resident (see test_rag_boot_import_closure.py's comment).
FORBIDDEN_ON_CHAT_LEG = (
    "tldw_chatbook.Chat.trajectory_export",
    "tldw_chatbook.Chat.trajectory_import",
    "tldw_chatbook.UI.Screens.trajectory_screen",
    "tldw_chatbook.UI.Widgets.trajectory_timeline",
    "tldw_chatbook.UI.Widgets.trace_filter_bar",
    "tldw_chatbook.Widgets.Console.trace_export_dialog",
)

#: The light leaves that replaced the forbidden imports. Their residency is
#: the anti-vacuity half: proof the module under test still consumes the
#: vocabulary, just through the cheap seam.
ENUM_LEAF = "tldw_chatbook.Chat.trace_export_profiles"
UI_LEAF = "tldw_chatbook.Widgets.Console.trace_export_profile_ui"


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet in a fresh interpreter with isolated config/data dirs.

    Args:
        tmp_path: Per-test scratch dir for HOME/XDG so the import can never
            read or write the live user config.
        code: Python source for ``python -c``.

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
        "USERPROFILE": str(home),
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


_PER_EDGE_SNIPPET = """
import sys

MODULE = {module!r}
FORBIDDEN = {forbidden!r}
EXPECTED_LEAVES = {leaves!r}

__import__(MODULE)

resident = [
    name
    for name in FORBIDDEN
    if sys.modules.get(name) is not None
]
assert not resident, (
    f"{{MODULE}} put the trajectory engine back on the Chat first-paint "
    f"leg: {{resident}}. Its module scope (or something it imports) must "
    "take TraceExportProfile from Chat/trace_export_profiles.py and the "
    "profile copy from Widgets/Console/trace_export_profile_ui.py -- "
    "never from Chat/trajectory_export.py or trace_export_dialog.py "
    "(TASK-23020)."
)

missing = [name for name in EXPECTED_LEAVES if name not in sys.modules]
assert not missing, (
    f"anti-vacuity: {{MODULE}} no longer resolves the light leaves "
    f"{{missing}} -- this guard is no longer watching a live seam and must "
    "be re-pointed, not deleted."
)
print("EDGE_OK")
"""


@pytest.mark.parametrize(
    ("module", "leaves"),
    [
        # The Chat-layer projector needs only the enum leaf.
        ("tldw_chatbook.Chat.console_exchange_export", (ENUM_LEAF,)),
        # The dialog needs the enum AND the shared presentation.
        (
            "tldw_chatbook.Widgets.Console.console_exchange_export_dialog",
            (ENUM_LEAF, UI_LEAF),
        ),
        # The inspector is the module chat_screen actually imports; it must
        # stay clean TRANSITIVELY (this is the edge #2126 rode).
        (
            "tldw_chatbook.Widgets.Console.console_conversation_inspector",
            (ENUM_LEAF, UI_LEAF),
        ),
    ],
)
def test_chat_leg_exchange_export_module_stays_off_the_trajectory_engine(
    tmp_path: Path, module: str, leaves: tuple[str, ...]
) -> None:
    """Importing each chat-leg exchange-export module resolves no engine module.

    One test per file so the failure NAMES the offender -- the leg-wide
    guards can only say "something re-eagered trajectory_export".

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
        module: The chat-leg module to import bare.
        leaves: Light modules that must be resident afterwards (anti-vacuity).
    """
    code = _PER_EDGE_SNIPPET.format(
        module=module, forbidden=FORBIDDEN_ON_CHAT_LEG, leaves=leaves
    )
    result = _run_isolated_python(tmp_path, code)
    assert result.returncode == 0, (
        f"per-edge closure failed for {module}:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "EDGE_OK" in result.stdout


_LEAVES_ARE_LIGHT_SNIPPET = """
import sys

FORBIDDEN = {forbidden!r}

# The enum leaf must be stdlib-only: importing it (past its package inits)
# adds exactly itself to this repo's resident modules.
import tldw_chatbook.Chat  # package inits may carry their own weight

before = {{m for m in sys.modules if m.startswith("tldw_chatbook")}}
import tldw_chatbook.Chat.trace_export_profiles  # noqa: F401

added = {{m for m in sys.modules if m.startswith("tldw_chatbook")}} - before
assert added == {{"tldw_chatbook.Chat.trace_export_profiles"}}, (
    f"the enum leaf stopped being stdlib-only; it now drags {{sorted(added)}}"
)

# The UI leaf may add only itself, the confirmation dialog, and their
# support -- never anything in the forbidden engine set.
import tldw_chatbook.Widgets.Console.trace_export_profile_ui  # noqa: F401

resident = [m for m in FORBIDDEN if sys.modules.get(m) is not None]
assert not resident, (
    f"trace_export_profile_ui re-eagered the trajectory engine: {{resident}}. "
    "It must never import trajectory_export, trajectory, or "
    "trace_export_dialog (TASK-23020)."
)
print("LEAVES_LIGHT_OK")
"""


def test_the_replacement_leaves_are_themselves_light(tmp_path: Path) -> None:
    """The seam modules the fix introduced cannot re-grow the forbidden edge.

    The failure mode TASK-23020 exists to prevent is a future one-line
    import quietly re-eagering the engine; the most attractive place for
    that line is inside these two leaves (e.g. "import the copy back from
    trace_export_dialog"). Pin them shut.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    code = _LEAVES_ARE_LIGHT_SNIPPET.format(forbidden=FORBIDDEN_ON_CHAT_LEG)
    result = _run_isolated_python(tmp_path, code)
    assert result.returncode == 0, (
        f"leaf lightness contract failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "LEAVES_LIGHT_OK" in result.stdout


_ONE_OBJECT_SNIPPET = """
import sys

from tldw_chatbook.Chat import trace_export_profiles
from tldw_chatbook.Widgets.Console import trace_export_profile_ui

# Loading the heavy side ON DEMAND must hand back the SAME objects the
# chat leg already holds -- the split is a relocation, not a fork.
from tldw_chatbook.Chat import trajectory_export
from tldw_chatbook.Widgets.Console import trace_export_dialog

assert (
    trajectory_export.TraceExportProfile
    is trace_export_profiles.TraceExportProfile
), "trajectory_export forked TraceExportProfile"
assert (
    trace_export_dialog.TRACE_EXPORT_PROFILE_COPY
    is trace_export_profile_ui.TRACE_EXPORT_PROFILE_COPY
), "trace_export_dialog forked TRACE_EXPORT_PROFILE_COPY"
assert (
    trace_export_dialog.TRACE_EXPORT_PROFILE_LABELS
    is trace_export_profile_ui.TRACE_EXPORT_PROFILE_LABELS
), "trace_export_dialog forked TRACE_EXPORT_PROFILE_LABELS"
assert (
    trace_export_dialog.full_trace_confirmation
    is trace_export_profile_ui.full_trace_confirmation
), "trace_export_dialog forked full_trace_confirmation"

# ...and the deferred dialog class still resolves (absence-only guards are
# satisfied by deleting the feature).
from textual.screen import ModalScreen

assert issubclass(trace_export_dialog.TraceExportDialog, ModalScreen)
print("ONE_OBJECT_OK")
"""


def test_profile_vocabulary_is_one_object_across_both_sides(
    tmp_path: Path,
) -> None:
    """Both export stacks share the leaf objects; neither side forked them.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _ONE_OBJECT_SNIPPET)
    assert result.returncode == 0, (
        f"one-object contract failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "ONE_OBJECT_OK" in result.stdout


_DIALOG_FROM_DEFERRED_STATE_SNIPPET = """
import asyncio
import json
import sys

FORBIDDEN = {forbidden!r}


def assert_engine_absent(moment):
    resident = [m for m in FORBIDDEN if sys.modules.get(m) is not None]
    assert not resident, f"trajectory engine resident {{moment}}: {{resident}}"


from tldw_chatbook.Widgets.Console.console_exchange_export_dialog import (
    ConsoleExchangeExportDialog,
)

assert_engine_absent("after importing the exchange export dialog")

from textual.widgets import Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    ExchangeCapture,
)
from tldw_chatbook.Chat.trace_export_profiles import TraceExportProfile

capture = ExchangeCapture(
    run_tag="run",
    seq=1,
    created_at="2026-08-27T00:00:00Z",
    provider="anthropic",
    model="claude-test",
    endpoint="https://example.test/v1",
    request={{"messages_payload": [{{"role": "user", "content": "hello"}}]}},
    response={{"content": "deferred-state answer", "tool_calls": []}},
    status="complete",
    usage_json=None,
    omitted_keys=(),
    capture_detail=CaptureDetail.FULL,
)


class Harness(ConsolidatedCSSApp):
    def __init__(self):
        super().__init__()
        self.clipboard_items = []

    def compose(self):
        yield Static("background")

    def copy_to_clipboard(self, text):
        self.clipboard_items.append(text)


async def main():
    app = Harness()
    async with app.run_test(size=(80, 24)) as pilot:
        dialog = ConsoleExchangeExportDialog(
            capture,
            expected_capture_revision=1,
            capture_revision_provider=lambda: 1,
        )
        await app.push_screen(dialog)
        await pilot.pause()
        assert_engine_absent("after mounting the dialog")

        # The default profile's copy renders from the light UI leaf.
        rendered = str(
            dialog.query_one("#exchange-export-profile-copy", Static).render()
        )
        assert "Redacted diagnostic" in rendered, rendered

        # Full profile: the shared confirmation still gates the disclosure.
        await dialog.select_profile(TraceExportProfile.FULL_TRACE)
        confirmations = []

        async def confirm():
            confirmations.append(True)
            return True

        dialog._confirm_full_export = confirm
        assert await dialog.export_selected() is True
        assert confirmations == [True], confirmations
        assert len(app.clipboard_items) == 1

        payload = json.loads(app.clipboard_items[0])
        assert payload["provider"] == "anthropic"
        assert payload["capture_detail"] == "full"
        assert payload["response"]["content"] == "deferred-state answer"


asyncio.run(main())

# The ENTIRE flow -- open, profile switch, confirm, project, disclose --
# ran without ever resolving the trajectory engine. That is the deliverable:
# the exchange export surface no longer needs it at all.
assert_engine_absent("after the full export flow")
print("DIALOG_DEFERRED_OK")
"""


def test_exchange_export_dialog_works_from_the_deferred_state(
    tmp_path: Path,
) -> None:
    """The dialog opens, confirms, and discloses with the engine never loaded.

    Absence is not the deliverable -- a deleted feature would satisfy a pure
    closure guard. This drives the real dialog (mount, profile copy render,
    Full-trace confirmation via the relocated ``full_trace_confirmation``,
    projection, clipboard disclosure) in a subprocess starting from exactly
    the state the Chat leg boots into, and asserts the trajectory engine is
    absent before, during, and after.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    code = _DIALOG_FROM_DEFERRED_STATE_SNIPPET.format(
        forbidden=FORBIDDEN_ON_CHAT_LEG
    )
    result = _run_isolated_python(tmp_path, code)
    assert result.returncode == 0, (
        f"deferred-state dialog flow failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "DIALOG_DEFERRED_OK" in result.stdout

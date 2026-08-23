"""Import-closure guard: the TASK-21108 app.py import diet stays done.

``import tldw_chatbook.app`` used to execute three things it does not need
before first paint, measured on dev at 36 modules / ~36 ms of import
self-time:

1. ``Widgets/Settings_Widgets/speech_tts_settings_panel`` -- a 5,600-line
   Textual widget module, imported for the single
   ``SpeechTTSPanelDraftSnapshot`` payload class used in two ``type(x) is``
   checks (20 modules, ~13 ms). It dragged the whole
   ``Third_Party/textual_fspicker`` tree (and with it ``rich._emoji_codes``),
   ``UI/Lab_Modules/lab_speech_status``, ``UI/Speech/speech_runtime_status``
   and ``Chat/console_voice_input``. The payload now lives in the pure
   ``speech_tts_panel_types`` module, which the panel re-imports, so the
   class stays one object across both paths.
2. ``TTS/voice_bundle_service`` (1,857 lines, ~2 ms) -- pulled in not by
   ``app.py`` but by the eager ``TTS/__init__``, so ``from tldw_chatbook.TTS
   import TTSProfileService`` executed it. Its five exports are now served by
   a PEP 562 ``__getattr__`` on the package.
3. The ``Notes/notes_sync_runtime`` chain plus ``Notes/notes_sync_legacy``
   (the TASK-21112 start gate's evidence source) -- 15 modules and ~21 ms,
   imported because ``TldwCli.__init__`` built the lasting-sync runtime
   eagerly. The owner is now built by the lazy ``notes_sync_runtime_owner``
   property, whose first production reader is ``on_mount``.

Subprocess-isolated for the same reason as
``test_chunking_import_closure.py`` (TASK-21102) and
``test_persona_buddy_import_closure.py`` (TASK-21103), whose pattern this
file follows: ``sys.modules`` is process-global, so an earlier test in the
session that legitimately imported any of these would false-fail (or a
pre-imported app would false-pass) an in-process check.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


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
        timeout=120,
    )


_APP_DIET_SNIPPET = """
import sys

import tldw_chatbook.app  # noqa: F401

forbidden = (
    # The 5,600-line panel and the payload-only seam that replaced it.
    "tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel",
    # Reached only through the eager TTS package init before the facade.
    "tldw_chatbook.TTS.voice_bundle_service",
    # The lasting-sync chain, built eagerly in TldwCli.__init__ before.
    "tldw_chatbook.Notes.notes_sync_runtime",
    "tldw_chatbook.Notes.notes_sync_legacy",
    "tldw_chatbook.Notes.notes_sync_authority",
    "tldw_chatbook.Notes.notes_sync_coordinator",
    "tldw_chatbook.Notes.notes_sync_executor",
    "tldw_chatbook.Notes.notes_sync_filesystem",
    "tldw_chatbook.Notes.notes_sync_models",
    "tldw_chatbook.Notes.notes_sync_reconciler",
    "tldw_chatbook.Notes.notes_sync_watcher",
    "tldw_chatbook.Notes.notes_device_state_schema",
    "tldw_chatbook.Notes.notes_device_state_store",
    "tldw_chatbook.Notes.note_import_discovery",
    "tldw_chatbook.Notes.note_import_plan_models",
    "tldw_chatbook.Notes.note_import_windows_fs",
    "tldw_chatbook.Notes.sync_paths",
)
resident = sorted(m for m in forbidden if sys.modules.get(m) is not None)
assert not resident, (
    "deferred modules resident after app import: " + repr(resident) + " -- if a "
    "new boot-path consumer genuinely needs one of these, that is the decision "
    "to review, not this list"
)

# The panel's own subtree, which came off the boot path with it. Kept as a
# named group so a re-import of the panel is reported as the subtree
# regression it is.
panel_subtree = tuple(
    m for m in sys.modules
    if sys.modules[m] is not None
    and (
        m == "tldw_chatbook.Third_Party.textual_fspicker"
        or m.startswith("tldw_chatbook.Third_Party.textual_fspicker.")
        or m in (
            "tldw_chatbook.UI.Lab_Modules.lab_speech_status",
            "tldw_chatbook.UI.Speech.speech_runtime_status",
            "tldw_chatbook.Chat.console_voice_input",
        )
    )
)
assert not panel_subtree, (
    "Speech/TTS panel subtree resident after app import: " + repr(sorted(panel_subtree))
)

# Anti-vacuity: the converted seams must still be part of the app's import
# closure. If one leaves entirely, the assertions above would pass without
# testing the conversion at all.
for expected in (
    # The payload seam app.py now imports instead of the panel.
    "tldw_chatbook.Widgets.Settings_Widgets.speech_tts_panel_types",
    # The pure settings model the payload seam depends on -- already on the
    # boot path via Event_Handlers/STTS_Events/stts_events, which is why the
    # seam costs nothing. If this leaves, the seam starts ADDING boot cost.
    "tldw_chatbook.UI.Screens.settings_speech_tts",
    # The package whose init the voice-bundle facade defers out of.
    "tldw_chatbook.TTS",
    # app.py still wires notes scopes eagerly; only the sync runtime moved.
    "tldw_chatbook.Notes.notes_scope_service",
):
    assert expected in sys.modules, f"expected closure member missing: {expected}"

print("APP_IMPORT_DIET_OK")
"""


def test_app_import_does_not_execute_the_deferred_boot_modules(tmp_path: Path) -> None:
    """None of the TASK-21108 deferrals is resident after ``import tldw_chatbook.app``.

    Regression guard for the TASK-21108 defect: before the fix, the Speech/TTS
    settings panel, ``TTS/voice_bundle_service`` and the 15-module
    lasting-sync chain all executed during ``import tldw_chatbook.app``, so
    this subprocess failed on the residency assertion.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _APP_DIET_SNIPPET)
    assert result.returncode == 0, (
        "import tldw_chatbook.app must not execute the deferred boot modules:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "APP_IMPORT_DIET_OK" in result.stdout


_DEFERRED_SEAMS_SNIPPET = """
import sys

# 1. The payload seam imports without executing the panel.
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_panel_types import (
    SpeechTTSPanelDraftSnapshot,
    _RealtimeSettingsDraft,
)

assert "tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel" not in sys.modules, (
    "importing the payload seam executed the 5,600-line panel"
)
assert "tldw_chatbook.Third_Party.textual_fspicker" not in sys.modules

# ...and the panel re-exports the SAME objects, so `type(x) is Snapshot`
# holds whichever module a caller imported it from.
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
    SpeechTTSPanelDraftSnapshot as PanelSnapshot,
    _RealtimeSettingsDraft as PanelRealtimeDraft,
)

assert PanelSnapshot is SpeechTTSPanelDraftSnapshot
assert PanelRealtimeDraft is _RealtimeSettingsDraft

# 2. The TTS package facade: importing the package must not execute
# voice_bundle_service, but the package must still serve its five exports,
# and they must be the submodule's own objects.
import tldw_chatbook.TTS as tts_pkg

assert "tldw_chatbook.TTS.voice_bundle_service" not in sys.modules, (
    "the TTS package init still executes voice_bundle_service"
)

from tldw_chatbook.TTS.voice_bundle_service import (
    TTSVoiceBundleHandle,
    TTSVoiceBundleImportChoice,
    TTSVoiceBundleImportResult,
    TTSVoiceBundlePortabilityService,
    TTSVoiceBundleReview,
)

for name, obj in (
    ("TTSVoiceBundleHandle", TTSVoiceBundleHandle),
    ("TTSVoiceBundleImportChoice", TTSVoiceBundleImportChoice),
    ("TTSVoiceBundleImportResult", TTSVoiceBundleImportResult),
    ("TTSVoiceBundlePortabilityService", TTSVoiceBundlePortabilityService),
    ("TTSVoiceBundleReview", TTSVoiceBundleReview),
):
    assert getattr(tts_pkg, name) is obj, name
    assert name in dir(tts_pkg), name

# The __getattr__ hook must not swallow unknown names, or `from pkg import
# <submodule>` would stop falling through to the import machinery.
try:
    tts_pkg.definitely_not_a_tts_export
except AttributeError:
    pass
else:  # pragma: no cover - only reached on a broken hook
    raise AssertionError("TTS.__getattr__ served a name it does not own")

from tldw_chatbook.TTS import profile_schema  # noqa: F401

print("DEFERRED_SEAMS_OK")
"""


def test_deferred_seams_resolve_and_stay_single_sourced(tmp_path: Path) -> None:
    """Each deferral still serves its callers, with one object per name.

    Three properties in one subprocess: the payload seam imports without the
    panel and the panel re-exports the same class objects (so the ``type(x)
    is SpeechTTSPanelDraftSnapshot`` checks in ``app.py`` keep matching); the
    TTS package facade defers ``voice_bundle_service`` but still serves its
    five exports as the submodule's own objects; and the facade raises
    ``AttributeError`` for names it does not own, so ``from tldw_chatbook.TTS
    import <submodule>`` still works.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _DEFERRED_SEAMS_SNIPPET)
    assert result.returncode == 0, (
        f"deferred seam check failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "DEFERRED_SEAMS_OK" in result.stdout

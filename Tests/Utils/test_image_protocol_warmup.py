"""Graphics-protocol warm-up must run before Textual owns the terminal.

task-1650: textual_image selects its rendering protocol ONCE, at import
time, by querying the terminal (see textual_image/renderable/__init__.py).
Every app-side import was lazy -- nested in functions that run inside the
live app -- so the query could never get a reply and selection silently
fell back to half-cell rendering. Symptom: pixelated avatars and inline
images in Kitty/iTerm2, with no exception and no log line.
"""

import re
import sys
from pathlib import Path

import pytest

APP_SOURCE = (
    Path(__file__).resolve().parents[2] / "tldw_chatbook" / "app.py"
).read_text(encoding="utf-8")


def test_warm_up_loads_the_protocol_selecting_submodule():
    """The helper must import the submodule where selection actually happens.

    Importing the top-level ``textual_image`` package is NOT enough: the
    protocol choice lives in ``textual_image.renderable``, which a plain
    ``import textual_image`` never loads.
    """
    pytest.importorskip("textual_image")
    from tldw_chatbook.Utils.terminal_utils import warm_up_image_protocol

    sys.modules.pop("textual_image.renderable", None)
    sys.modules.pop("textual_image.widget", None)

    assert warm_up_image_protocol() is True
    assert "textual_image.renderable" in sys.modules


def test_warm_up_is_safe_without_the_optional_dependency(monkeypatch):
    """A missing optional dependency must not break startup.

    Args:
        monkeypatch: pytest fixture used to simulate the absent package.
    """
    from tldw_chatbook.Utils import terminal_utils

    def _raise(name, *args, **kwargs):
        raise ImportError("simulated missing textual_image")

    monkeypatch.setattr(terminal_utils.importlib, "import_module", _raise)
    assert terminal_utils.warm_up_image_protocol() is False


@pytest.mark.parametrize(
    "entry",
    [
        # The two real entry paths: `python -m tldw_chatbook.app` runs the
        # module-level __main__ block; the installed console script calls
        # main_cli_runner(). There is no `def main()`.
        'if __name__ == "__main__":',
        "def main_cli_runner(",
    ],
)
def test_every_entry_point_warms_up_before_running_the_app(entry):
    """Each entry point must warm up BEFORE handing the terminal to Textual.

    Args:
        entry: Source marker opening the entry-point block to inspect.
    """
    body = APP_SOURCE.split(entry, 1)[1]
    warm = body.find("warm_up_image_protocol(")
    run = body.find("app_instance.run()")
    assert warm != -1, f"{entry} never warms up the image protocol"
    assert run != -1, f"{entry} has no app_instance.run() to guard"
    assert warm < run, f"{entry} warms up AFTER app.run() -- too late to query"


def test_no_lazy_first_import_of_textual_image_widget_without_warmup():
    """Lazy import sites are fine, but only because startup warmed up first.

    Guards the invariant rather than the call sites: if the warm-up is ever
    removed from terminal_utils, the lazy imports silently regress to
    half-cell rendering again.
    """
    helper = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook" / "Utils" / "terminal_utils.py"
    ).read_text(encoding="utf-8")
    assert "def warm_up_image_protocol" in helper
    assert re.search(r"import_module\(\s*[\"']textual_image\.widget[\"']", helper)


def test_missing_dependency_is_distinguished_from_a_failed_query(monkeypatch):
    """ImportError and terminal-query failure must be told apart.

    Qodo PR #1150: a blanket ``except Exception`` returning False masked
    unexpected regressions. An absent optional dependency is expected
    (debug); a terminal that refuses the capability query is not (warning).

    Args:
        monkeypatch: pytest fixture used to force each failure mode.
    """
    from tldw_chatbook.Utils import terminal_utils

    seen: list[dict] = []
    monkeypatch.setattr(
        terminal_utils,
        "log_counter",
        lambda name, labels=None: seen.append(labels or {}),
    )

    monkeypatch.setattr(
        terminal_utils.importlib,
        "import_module",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("absent")),
    )
    assert terminal_utils.warm_up_image_protocol() is False
    assert seen[-1]["result"] == "missing_dependency"

    class _TerminalError(Exception):
        pass

    monkeypatch.setattr(
        terminal_utils.importlib,
        "import_module",
        lambda *a, **k: (_ for _ in ()).throw(_TerminalError("no reply")),
    )
    assert terminal_utils.warm_up_image_protocol() is False
    assert seen[-1]["result"] == "query_failed"


def test_successful_warm_up_reports_the_selected_protocol(monkeypatch):
    """A successful warm-up names the protocol it resolved.

    Qodo PR #1150: both entry points discard the bool, so the helper
    itself must carry the diagnostic -- otherwise a degraded render still
    has no log line, which is the symptom this fix exists to remove.

    Args:
        monkeypatch: pytest fixture capturing emitted metrics.
    """
    pytest.importorskip("textual_image")
    from tldw_chatbook.Utils import terminal_utils

    seen: list[dict] = []
    monkeypatch.setattr(
        terminal_utils,
        "log_counter",
        lambda name, labels=None: seen.append(labels or {}),
    )

    assert terminal_utils.warm_up_image_protocol() is True
    assert seen[-1]["result"] == "ok"
    # tgp / sixel / halfcell / unicode depending on the host terminal.
    assert seen[-1]["protocol"] in {"tgp", "sixel", "halfcell", "unicode"}

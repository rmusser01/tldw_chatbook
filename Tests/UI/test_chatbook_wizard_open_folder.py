"""Opening the export folder must not block the app (TASK-1373).

`ProgressStep.on_button_pressed` shelled out with `subprocess.run(["open"/
"xdg-open"/"explorer", folder])` and no timeout. Textual runs message handlers on
a serialized pump, so for as long as that child process ran, the app processed no
clicks, keys or navigation.

On macOS `open` returns promptly, which is why this was never noticed here. But
`xdg-open` is a shell script that in several desktop environments does not return
until the launched file manager exits -- so on Linux a button press could freeze
the app for the rest of the session. The fix must not wait on the child at all:
opening a folder is fire-and-forget.
"""

from __future__ import annotations

import io
import textwrap
import tokenize

import pytest

from tldw_chatbook.UI.Wizards import ChatbookCreationWizard as wizard_module


def _code_only(source: str) -> str:
    """Strip comments and string literals, leaving executable tokens.

    Load-bearing, not tidiness: the fix's own comment *explains* that
    ``subprocess.run`` waits for the child, so a plain substring check matched
    the explanation and reported the bug as unfixed. The same trap is documented
    in ``Tests/test_call_from_thread_guard.py`` -- a guard that reads prose is a
    guard that can be fooled in both directions.
    """
    ignored = {tokenize.COMMENT, tokenize.STRING}
    readline = io.StringIO(textwrap.dedent(source)).readline
    return " ".join(
        tok.string
        for tok in tokenize.generate_tokens(readline)
        if tok.type not in ignored
    )


@pytest.mark.unit
def test_open_folder_does_not_wait_on_the_child_process():
    """The folder-open path must never call a blocking `subprocess.run`.

    Asserted against the module source rather than by driving the widget,
    because the defect is precisely that the call blocks: a behavioural test
    would have to actually launch a file manager to observe it.
    """
    source = _code_only(_handler_source())

    assert "subprocess . run" not in source, (
        "the folder-open handler still calls subprocess.run, which waits for the "
        "child process on Textual's message pump -- xdg-open does not always "
        "return promptly, so this can freeze the whole app on a button press"
    )


@pytest.mark.unit
def test_open_folder_still_launches_the_platform_handler():
    """Not blocking must not mean not working."""
    source = _handler_source()

    # The launcher names are string literals, so this half reads the raw source.
    for expected in ("open", "xdg-open", "explorer"):
        assert expected in source, (
            f"the {expected!r} platform branch disappeared: the handler must "
            "still open the folder, just without waiting for it"
        )
    assert "Popen" in _code_only(source), (
        "expected a non-waiting launch (Popen) after dropping subprocess.run"
    )


def _handler_source() -> str:
    """Return the source of the completion-button handler."""
    import inspect

    return inspect.getsource(wizard_module.ProgressStep.on_button_pressed)

"""No Speech mixin may carry an `@on` handler.

Textual registers decorated handlers in its metaclass, scanning each class's
own namespace. A mixin that is not itself a MessagePump contributes nothing,
so an `@on` method defined there is registered nowhere: no error, no
warning, the handler never runs.

This has now bitten twice. First `on_tts_provider_select_changed` in the
catalog mixin, where provider switching silently stopped; then
`on_default_selects_changed` in the settings mixin, where changing the
default provider stopped repopulating the model and voice lists.

The second one got through a pre-move check for exactly this, because the
check ran over a smaller closure than was then extracted. Scanning the
modules themselves cannot go stale that way.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SPEECH = pathlib.Path(__file__).resolve().parents[2] / "tldw_chatbook" / "UI" / "Speech"
MIXINS = sorted(SPEECH.glob("*_mixin.py"))


@pytest.mark.unit
def test_there_are_mixins_to_check():
    """A glob that matches nothing would make every case below vacuous."""
    assert MIXINS, f"no *_mixin.py under {SPEECH}"


@pytest.mark.unit
@pytest.mark.parametrize("path", MIXINS, ids=lambda p: p.name)
def test_no_mixin_declares_an_on_handler(path):
    """`@on` here is dead code that looks alive."""
    tree = ast.parse(path.read_text())
    offenders = []
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in fn.decorator_list:
                name = ast.unparse(dec).split("(")[0]
                if name == "on":
                    offenders.append(f"{cls.name}.{fn.name}")

    assert not offenders, (
        f"@on in a mixin is never dispatched: {offenders}. Declare the "
        "handler on each host class and delegate to a plain method here."
    )

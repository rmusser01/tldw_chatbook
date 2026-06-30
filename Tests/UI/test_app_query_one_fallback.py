"""App-level query fallback tests."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.css.query import NoMatches

from tldw_chatbook.app import TldwCli


def test_query_one_reraises_screen_access_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken active screen lookup should not be hidden by the first miss."""

    def _raise_no_matches(
        self: App[None],
        selector: object,
        expect_type: object | None = None,
    ) -> object:
        raise NoMatches(f"missing {selector!r}")

    class BrokenScreenApp(TldwCli):
        @property
        def screen(self):  # type: ignore[override]
            raise RuntimeError("screen lookup failed")

    monkeypatch.setattr(App, "query_one", _raise_no_matches)
    app = BrokenScreenApp.__new__(BrokenScreenApp)

    with pytest.raises(RuntimeError, match="screen lookup failed") as exc_info:
        app.query_one("#missing")

    assert isinstance(exc_info.value.__cause__, NoMatches)

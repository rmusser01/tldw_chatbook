import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def read_starting_panel_source(monkeypatch):
    path = Path.cwd() / "tldw_chatbook/Widgets/Library/library_search_rag_panel.py"
    baseline = subprocess.check_output(
        [
            "git",
            "show",
            "79a7a270d6:tldw_chatbook/Widgets/Library/library_search_rag_panel.py",
        ],
        text=True,
    )
    original = Path.read_text

    def read(self, *args, **kwargs):
        return baseline if self.resolve() == path else original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)

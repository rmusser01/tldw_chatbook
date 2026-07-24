"""TASK-334: no dead `chat_with_provider` imports from LLM_API_Calls remain."""

import importlib
import pathlib
import re

import pytest

REPO_SRC = pathlib.Path(__file__).resolve().parents[2] / "tldw_chatbook"
DEAD_IMPORT = re.compile(
    r"import\s+chat_with_provider|from\s+[.\w]*LLM_API_Calls\s+import\s+chat_with_provider"
)


def test_no_dead_chat_with_provider_import_anywhere():
    offenders = []
    for py in REPO_SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if DEAD_IMPORT.search(text):
            offenders.append(str(py.relative_to(REPO_SRC)))
    # MCP/tools.py defines a LOCAL stub (def chat_with_provider) — that is NOT an
    # import and must not match; if it does, tighten the regex.
    assert offenders == [], f"dead chat_with_provider imports remain: {offenders}"


@pytest.mark.parametrize(
    "module",
    [
        "tldw_chatbook.MCP.server",
        "tldw_chatbook.Tools.code_audit_tool",
        "tldw_chatbook.UI.Tools_Settings_Window",
    ],
)
def test_touched_modules_import_clean(module):
    assert importlib.import_module(module) is not None

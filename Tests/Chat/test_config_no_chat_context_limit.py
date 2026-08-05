from pathlib import Path

import tldw_chatbook.config as config_mod
from tldw_chatbook.config import DEFAULT_RAG_SEARCH_CONFIG


def test_chat_context_limit_removed_from_defaults():
    assert "chat_context_limit" not in DEFAULT_RAG_SEARCH_CONFIG


def test_chat_context_limit_absent_from_config_source():
    # 325 AC#2: no references remain (dict default + sample TOML both gone).
    source = Path(config_mod.__file__).read_text(encoding="utf-8")
    assert "chat_context_limit" not in source

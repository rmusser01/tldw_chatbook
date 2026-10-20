"""Protected path lookup must not start the retrieval engine (TASK-34406.31)."""

from Tests.Packaging.test_chunking_import_closure import _run_isolated_python

_SNIPPET = """
import sys
from tldw_chatbook import config
from tldw_chatbook.Utils.sensitive_paths import (
    resolve_sensitive_context, is_sensitive_path,
)

context = resolve_sensitive_context()
root = config.get_user_data_dir()
for name in ("chromadb", "rag_profiles"):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    assert directory.resolve() in context.direct_child_denied_dirs
    assert is_sensitive_path(directory / "private.json", context)
    assert not is_sensitive_path(directory, context)
assert is_sensitive_path(config._get_effective_config_path(), context)
for prefix in (
    "tldw_chatbook.RAG_Search.simplified",
    "tldw_chatbook.RAG_Search.config_profiles",
    "tldw_chatbook.RAG_Search.reranker",
    "tldw_chatbook.RAG_Search.parallel_processor",
    "numpy",
):
    loaded = [name for name in sys.modules
              if name == prefix or name.startswith(prefix + ".")]
    assert not loaded, f"path lookup executed retrieval dependency: {loaded}"
# The actual metadata route and its parent admission must still be reached.
assert "tldw_chatbook.RAG_Search" in sys.modules
assert "tldw_chatbook.RAG_Search.storage_paths" in sys.modules
assert "tldw_chatbook.Backup_Recovery.storage_admission" in sys.modules
print("RAG_STORAGE_PATHS_OK")
"""


def test_sensitive_context_does_not_import_retrieval_engine(tmp_path):
    result = _run_isolated_python(tmp_path, _SNIPPET)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr[-5000:]}"
    assert "RAG_STORAGE_PATHS_OK" in result.stdout

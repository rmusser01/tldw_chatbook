"""Read tools bound the file first; the three editors write atomically.

TASK-32806.6.

`ReadFileTool` returned the entire file as `content` with no size cap (a
150 MB single-line file cost ~400 MB RSS), and `local_tool_impls.read_file`
capped its OUTPUT at 32 KB but still `read_text`'d the whole file first.
Both now check `st_size` before materialising.

`fs_edit`, `fs_patch` and the legacy `write_file` wrote in place
(truncate-then-write, no fsync, following a symlink at the target) while
`fs_write` used O_EXCL temp + fsync + `os.replace`. The three now route
through that same path.

These are structural pins over the source. The runtime paths sit behind
config reads and path resolution that this worktree's storage-admission
gate (`RecoveryRequired`) cannot satisfy, so a runtime assertion here would
fail for an unrelated reason; each pin fails loudly if a site regresses.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _function_source(module: str, name: str, must_contain: str | None = None) -> str:
    source = (REPO / module).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            body = ast.get_source_segment(source, node) or ""
            if must_contain is None or must_contain in body:
                return body
    raise AssertionError(f"could not find {name} in {module}")


@pytest.mark.parametrize(
    ("module", "function", "st_size_guard"),
    [
        (
            "tldw_chatbook/Tools/file_operation_tools.py",
            "execute",  # ReadFileTool.execute references READ_FILE_MAX_BYTES
            "READ_FILE_MAX_BYTES",
        ),
        (
            "tldw_chatbook/Tools/local_tool_impls.py",
            "_read_relative_file",
            "MAX_READ_FILE_BYTES",
        ),
    ],
)
def test_reads_are_bounded_before_the_file_is_materialised(
    module, function, st_size_guard
):
    body = _function_source(module, function, must_contain=st_size_guard)
    assert ".st_size" in body, f"{function} does not check st_size"
    # The size guard must appear before the whole-file read in source order.
    read_call = "read_text(" if "read_text(" in body else "read_bytes("
    assert body.index(st_size_guard) < body.index(read_call), (
        f"{function} reads the whole file before checking its size"
    )


@pytest.mark.parametrize(
    ("module", "function", "writer"),
    [
        ("tldw_chatbook/Tools/local_tool_impls.py", "_edit_relative_file", "_atomic_write_target"),
        ("tldw_chatbook/Tools/patch_tool_impls.py", "_patch_relative_file", "_atomic_write_target"),
        ("tldw_chatbook/Tools/file_operation_tools.py", "execute", "atomic_write_text"),
    ],
)
def test_the_editors_no_longer_write_in_place(module, function, writer):
    body = _function_source(module, function, must_contain=writer)
    assert ".write_bytes(" not in body, f"{function} still writes in place"
    assert 'open(path, "w", ' not in body and 'open(path, "a", ' not in body, (
        f"{function} still opens the target for in-place write"
    )
    assert writer in body, f"{function} does not route through {writer}"

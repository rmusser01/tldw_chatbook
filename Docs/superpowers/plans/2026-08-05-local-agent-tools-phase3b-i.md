# Local Agent Tools — Phase 3b-i (fs_patch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `fs_patch` — unified-diff apply across workspace files with dry-run — ported from tldw_server's self-contained `filesystem_diff.py`.

**Architecture:** New core module `Tools/patch_tool_impls.py`: a near-mechanical port of the parser/applier (stdlib-only, ~290 lines) plus a workspace wrapper enforcing ADR-032 confinement and phase-2 write discipline (encode-before-write, newline preservation). One new `LocalToolSpec` (`mutates` tag).

**Spec:** `Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md` §2.4 · **ADRs:** 032
**Reference source:** tldw_server @ `5605b9d9906322c2e6b5342b48c391ae674d315e`, `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py`. Clone may exist at `/tmp/tldw_server_mcp/tldw_server`; if missing, re-clone per the phase-3a plan's instructions. **Attribution header with repo + path + SHA is binding (re-plan §5).**

## Verified facts (do not re-derive)

- The reference module is fully self-contained stdlib Python: `parse_unified_diff(diff_text, *, max_files, max_hunks, max_bytes) -> tuple[PatchFile, ...]` and `apply_patch_to_text(original, patch_file) -> str`, raising `FilesystemPatchError(reason_code)`. It handles: `a/`/`b/` prefixes, `/dev/null` create sentinel, tab/timestamp header metadata, paths with spaces, `\ No newline at end of file`, hunk line-count validation, CRLF/CR/LF detection and preservation, context-mismatch refusal. It REJECTS deletes (`delete_not_supported`) and renames (`rename_not_supported`) — keep that.
- The reference is pure text-in/text-out — no filesystem I/O. The workspace wrapper is ours to write.
- Phase-2 discipline that MUST carry over: encode-before-write (`write_bytes(updated.encode("utf-8"))`, wrap UnicodeEncodeError in LocalToolError), `newline=""` reads, non-UTF-8 read wrapped in LocalToolError (`local_tool_impls.py` edit_file is the model).
- `resolve_workspace_path` confines each target file; LocalToolError is the shared error type.
- Provider: `LocalToolSpec` in `_default_specs`, `tags=("mutates",)`, catalog exact-id test in `Tests/Agents/test_local_tool_provider.py` gets extended (now ends `...web_fetch, web_search`; todo_write stays conditionally registered).
- Tests: `Tests/Tools/test_patch_tool_impls.py` (new); `ws = tmp_path/"ws"` fixture pattern; run with the repo venv python from the worktree. Known pre-existing failures to deselect: the anthropic native-tools and github-api-client tests.

---

## Task 0: Backlog task

- [ ] Create "Local agent tools phase 3b-i: fs_patch (unified-diff apply)" via CLI; ACs:
  1. fs_patch applies multi-file multi-hunk unified diffs confined to the workspace root
  2. Context mismatches, deletes, renames, and malformed diffs return model-actionable errors without writing
  3. dry_run returns the would-be result and writes nothing
  4. Diff size/file/hunk limits enforced; writes are encode-before-write and newline-preserving
  5. All new tests pass
  Commit: `docs: create phase-3b-i backlog task`

---

## Task 1: Port parser/applier + workspace wrapper

**Files:**
- Create: `tldw_chatbook/Tools/patch_tool_impls.py`
- Test: `Tests/Tools/test_patch_tool_impls.py`

- [ ] **Step 1: Failing tests** (write first; sample diffs as fixtures):

```python
MODIFY_DIFF = """\
--- a/notes.txt
+++ b/notes.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+BETA
 gamma
"""

CREATE_DIFF = """\
--- /dev/null
+++ b/new.txt
@@ -0,0 +1,2 @@
+one
+two
"""

def test_apply_modify(tmp_path): ...        # notes.txt patched, "patched notes.txt" in result
def test_apply_create(tmp_path): ...        # new.txt created
def test_dry_run_writes_nothing(tmp_path): ...  # files untouched, result describes changes
def test_context_mismatch(tmp_path): ...    # LocalToolError "patch_context_mismatch"; file untouched
def test_delete_and_rename_refused(tmp_path): ...  # "delete_not_supported" / "rename_not_supported"
def test_malformed_diff(tmp_path): ...      # "invalid_diff"
def test_limits(tmp_path): ...              # max_bytes / max_files / max_hunks exceeded -> reason codes
def test_confinement(tmp_path): ...         # diff targeting ../evil.txt -> LocalToolError, nothing written
def test_crlf_preserved(tmp_path): ...      # CRLF file stays CRLF byte-exact outside the edited line
def test_multi_file_atomicity_note(tmp_path): ...  # documents behavior: per-file sequential apply; a failing LATER file leaves earlier files patched (see Step 3 note)
```

- [ ] **Step 2: Verify failure** (ModuleNotFoundError)

- [ ] **Step 3: Implement** `patch_tool_impls.py`:

1. Port `parse_unified_diff` + `apply_patch_to_text` + all helpers/dataclasses from the reference, near-verbatim, with the attribution header. Rename `FilesystemPatchError` usage at the wrapper boundary (see 3).
2. Add limits constants: `PATCH_MAX_BYTES = 256 * 1024`, `PATCH_MAX_FILES = 20`, `PATCH_MAX_HUNKS = 200`.
3. Add the workspace wrapper:

```python
def patch_files(diff_text: str, *, workspace_root: Path, dry_run: bool = False) -> str:
    """Parse and apply a unified diff to workspace files.

    Every target is confined via resolve_workspace_path. Modify targets must
    exist; create targets must not. dry_run validates and reports without
    writing. Returns a per-file summary ("patched X", "would patch X").
    Files are applied sequentially; if a later file fails, earlier files stay
    patched — the error names the failed file so the model can recover
    (atomic multi-file apply is a documented non-goal for this phase).
    """
```

   Wrapper rules: translate `FilesystemPatchError` to `LocalToolError` keeping the reason code in the message (`fs_patch failed [patch_context_mismatch]: <file>`); read modify-targets with `open(newline="")` + non-UTF-8 wrap; write with encode-before-write (`write_bytes`), wrapping UnicodeEncodeError; create-target parent must exist (fs_write parity).

- [ ] **Step 4:** tests pass
- [ ] **Step 5:** `git commit -m "feat: fs_patch core (ported unified-diff parser/applier + workspace wrapper)"`

---

## Task 2: `fs_patch` spec + provider test

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Test: `Tests/Agents/test_local_tool_provider.py`

- [ ] **Step 1: Failing tests** — catalog includes `local:fs_patch`; `tags == ("mutates",)`; schema requires `diff` (string), optional `dry_run` (bool, default false); handler smoke test on tmp workspace (a create diff lands the file).
- [ ] **Step 2: Implement** — spec in `_default_specs`; description should tell the model the format (unified diff, `a/`/`b/` prefixes optional, no deletes/renames, dry_run to preview) since models hallucinate diff formats.
- [ ] **Step 3:** extend the catalog exact-id test; run provider + integration suites
- [ ] **Step 4:** `git commit -m "feat: fs_patch tool spec"`

---

## Task 3: Close-out (controller-led)

- [ ] Backlog task: ACs checked, Implementation Notes, Done.
- [ ] Final review subagent (diff + ACs + test run).
- [ ] superpowers:finishing-a-development-branch.

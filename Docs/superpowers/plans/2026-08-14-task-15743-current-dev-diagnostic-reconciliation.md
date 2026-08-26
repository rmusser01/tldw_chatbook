# TASK-15743 Current-Dev Diagnostic Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the ADR-029 production-diagnostic boundary on the final TASK-3070.2/TASK-16001 stack without blessing unsafe current-dev drift.

**Architecture:** Reuse the existing diagnostic extractor, reviewed-shape registry, and generated manifest. First make the audited new call shapes fail the architecture evidence, then minimally replace private/dynamic fields and exception capture with fixed labels plus bounded exception class names, and regenerate the manifest once.

**Tech Stack:** Python 3.11, pytest, AST-based architecture checks, Loguru, Ruff, Bandit.

**Interpreter:** Every command below first resolves the shared repository venv
without embedding a private checkout path:
`project_python="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/python"`.

**Final-rebase amendment:** The mandatory final rebase imported 17 additional
unsafe call sites across 11 owners. The design records their exact disposition.
`test_task_15743_final_rebase_diagnostics_are_metadata_only` must fail before
the repairs and pass afterward. These repairs join the final source/manifest
commit and all affected focused tests below; no sink or metadata policy changes.

**Verification amendment:** The owner explicitly directed on 2026-08-14 not to
run the repository-wide suite. The focused architecture, security boundary,
and directly affected feature/static gates are the completion boundary; the
checkpointed full-suite program below is retained only as superseded history
and must not be executed for this task.

---

### Task 1: Freeze governance and the audited delta

**Files:**
- Create: `backlog/tasks/task-15743 - Reconcile-current-dev-diagnostic-inventory-drift.md`
- Create: `Docs/superpowers/specs/2026-08-14-task-15743-current-dev-diagnostic-reconciliation-design.md`
- Create: `Docs/superpowers/plans/2026-08-14-task-15743-current-dev-diagnostic-reconciliation.md`

- [ ] Record TASK-3070.2 and TASK-16001 as dependencies and link ADR-029.
- [ ] Record the design's exact 31 unsafe call identities, nine reviewed-safe surviving shapes, 19 Moonshot deletions, 13 Z.AI deletions, and unchanged six-file sink topology.
- [ ] Run `git diff --check -- Docs/superpowers/specs/2026-08-14-task-15743-current-dev-diagnostic-reconciliation-design.md Docs/superpowers/plans/2026-08-14-task-15743-current-dev-diagnostic-reconciliation.md 'backlog/tasks/task-15743 - Reconcile-current-dev-diagnostic-inventory-drift.md'`; expect exit 0.
- [ ] Commit only the governance artifacts with `docs: plan current-dev diagnostic reconciliation`.

### Task 2: Add failing reviewed-shape evidence

**Files:**
- Modify: `Tests/Architecture/test_persistent_diagnostic_inventory.py`

- [ ] Extend `REVIEWED_METADATA_ONLY_DIAGNOSTICS` with every surviving new or moved fixed event and its exact allowed field set.
- [ ] Add `test_task_15743_reviewed_delta_is_complete`, whose expected set is the design's exact 31 repair rows plus nine reviewed-safe rows; reject a missing, duplicate, renamed, or extra disposition.
- [ ] Resolve `project_python` as documented above, then run `"$project_python" -B -m pytest -q Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only`; expect RED with one expected zero-match/private-field/exception-capture disposition per unsafe row and no no-edit-row failure.
- [ ] Run `"$project_python" -B -m pytest -q Tests/Architecture/test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged`; expect exactly one stale-manifest assertion failure.
- [ ] Commit the failing evidence only if the repository permits an intentional RED commit; otherwise retain the verified RED and continue without committing.

### Task 3: Repair Agents, Character, and fleet diagnostics

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_fleet_attention.py`
- Modify: `tldw_chatbook/Chat/console_fleet_wake.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

- [ ] Replace only the reviewed unsafe call shapes with fixed events and, where useful, `exception_type=type(exc).__name__`.
- [ ] Preserve each catch/retry/cancel/control-flow path exactly.
- [ ] Run `"$project_python" -B -m pytest -q Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only Tests/Architecture/test_persistent_diagnostic_inventory.py::test_task_15743_reviewed_delta_is_complete Tests/Agents/test_agent_service.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_fleet_wake.py Tests/Chat/test_console_fleet_wake_safety.py Tests/Chat/test_fleet_attention.py Tests/Chat/test_fleet_usage_reattach.py Tests/Character_Chat/test_character_file_operations.py Tests/UI/test_console_fleet_wake_wiring.py`; expect zero failures while the separate inventory-equality node intentionally remains red until Task 5.
- [ ] Temporarily restore one exception-capturing/private-field shape and prove the named evidence fails, then restore the fix.
- [ ] Commit the architecture evidence and these production repairs together with `fix: redact fleet diagnostic metadata`.

### Task 4: Repair image, Library, and CSS diagnostics

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/image.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/app.py`

- [ ] Remove URL, session, path, dynamic reason, exception text, and exception-capture fields from the reviewed calls.
- [ ] Retain only fixed labels, generated-sheet count, and bounded exception class names permitted by the evidence.
- [ ] Run `"$project_python" -B -m pytest -q Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only Tests/Architecture/test_persistent_diagnostic_inventory.py::test_task_15743_reviewed_delta_is_complete Tests/Architecture/test_console_wave6_inventory.py Tests/Chat/test_console_generate_image.py Tests/Chat/test_console_generation_actions.py Tests/Chat/test_console_generation_card.py Tests/Chat/test_console_h3_image_edit.py Tests/Chat/test_console_image_controller.py Tests/Chat/test_console_remote_images.py Tests/UI/test_console_citation_sources.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_pending_attachment_stash.py Tests/Library/test_library_conversations_state.py Tests/Widgets/Library/test_library_conversations_canvas.py Tests/UI/test_library_shell.py Tests/UI/test_css_build_integrity.py Tests/UI/test_css_bundle_sync_guard.py Tests/UI/test_widget_css_consolidation.py Tests/App/test_submit_library_ingest_job.py`; expect zero failures while the separate inventory-equality node intentionally remains red until Task 5.
- [ ] Mutate one call per family back to its unsafe shape and prove the evidence fails, restoring after each mutation.
- [ ] Commit with `fix: redact UI diagnostic metadata`.

### Task 5: Regenerate once and verify the stack

**Files:**
- Modify: `Docs/security/production-diagnostic-inventory.json`

- [ ] Run `"$project_python" scripts/check_persistent_diagnostic_inventory.py --write`; expect exactly `wrote Docs/security/production-diagnostic-inventory.json` and exit 0.
- [ ] Review the complete stored-versus-generated delta: exact owners, reviewed-safe additions, 32 reviewed deletions, and unchanged six-file sink topology.
- [ ] Run `"$project_python" -B -m pytest -q Tests/Architecture/test_persistent_diagnostic_inventory.py`; expect all architecture nodes green, then run the two exact focused pytest commands from Tasks 3 and 4 again; expect zero failures.
- [ ] Run `"$project_python" -m ruff check --no-cache Tests/Architecture/test_persistent_diagnostic_inventory.py tldw_chatbook/Agents/agent_service.py tldw_chatbook/Character_Chat/Character_Chat_Lib.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_fleet_attention.py tldw_chatbook/Chat/console_fleet_wake.py tldw_chatbook/UI/Console_Modules/image.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/app.py`; expect `All checks passed!`.
- [ ] Run `"$project_python" -m ruff format --check --no-cache Tests/Architecture/test_persistent_diagnostic_inventory.py tldw_chatbook/Agents/agent_service.py tldw_chatbook/Character_Chat/Character_Chat_Lib.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_fleet_attention.py tldw_chatbook/Chat/console_fleet_wake.py tldw_chatbook/UI/Console_Modules/image.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/app.py`; expect all files already formatted.
- [ ] Run `"$project_python" -m bandit -q -ll tldw_chatbook/Agents/agent_service.py tldw_chatbook/Character_Chat/Character_Chat_Lib.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_fleet_attention.py tldw_chatbook/Chat/console_fleet_wake.py tldw_chatbook/UI/Console_Modules/image.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/app.py`; expect exit 0.
- [ ] Run `"$project_python" -c "from pathlib import Path; paths=('tldw_chatbook/Agents/agent_service.py','tldw_chatbook/Character_Chat/Character_Chat_Lib.py','tldw_chatbook/Chat/console_agent_bridge.py','tldw_chatbook/Chat/console_chat_controller.py','tldw_chatbook/Chat/console_fleet_attention.py','tldw_chatbook/Chat/console_fleet_wake.py','tldw_chatbook/UI/Console_Modules/image.py','tldw_chatbook/UI/Screens/chat_screen.py','tldw_chatbook/UI/Screens/library_screen.py','tldw_chatbook/app.py'); [compile(Path(path).read_text(encoding='utf-8'), path, 'exec') for path in paths]"`; expect exit 0 and no bytecode artifact.
- [ ] Run `"$project_python" -B -m pytest -q Tests/test_persistent_diagnostic_boundary.py Tests/test_persistent_diagnostic_sentinel_matrix.py`; expect zero failures, then run `git diff --check`; expect exit 0.
- [ ] Commit the reviewed generated manifest with `docs: reconcile production diagnostic inventory`.
- [ ] Run `git status --short` and `git ls-files | rg -n '(^|/)__pycache__/|\.(pyc|mp4|webm|avi)$'`; expect empty status and no tracked artifact match (the `rg` command exits 1).
- [ ] With a clean frozen worktree, run the checkpointed full-suite program below. It partitions the discovered test files into resumable 25-file chunks beneath a HEAD-specific private `/private/tmp` root, writes a success marker only after a chunk exits 0, prints the failed chunk log before stopping, skips successful chunks on relaunch, and prints per-chunk plus aggregate pass/skip/warning counts. Expect every chunk marked successful, aggregate failures 0, and aggregate counts recorded in Implementation Notes.

```bash
project_python="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/python"
head_short="$(git rev-parse --short=12 HEAD)"
checkpoint_root="/private/tmp/task-15743-full-suite-$head_short"
"$project_python" -B - "$checkpoint_root" <<'PY'
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

root = Path(sys.argv[1])
if root.is_symlink():
    raise SystemExit("CHECKPOINT_REFUSED")
if root.exists():
    if not root.is_dir() or root.stat().st_uid != os.geteuid():
        raise SystemExit("CHECKPOINT_REFUSED")
else:
    root.mkdir(mode=0o700)
if root.stat().st_mode & 0o077:
    raise SystemExit("CHECKPOINT_REFUSED")
if subprocess.run(
    ["git", "status", "--porcelain=v1"],
    check=True,
    capture_output=True,
    text=True,
).stdout:
    raise SystemExit("WORKTREE_NOT_FROZEN")

files = sorted(str(path) for path in Path("Tests").rglob("test_*.py"))
chunks = [files[index : index + 25] for index in range(0, len(files), 25)]
totals: Counter[str] = Counter()

def open_owned(path: Path, flags: int, mode: int = 0o600) -> int:
    fd = os.open(path, flags | os.O_NOFOLLOW, mode)
    details = os.fstat(fd)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_uid != os.geteuid()
        or details.st_mode & 0o077
    ):
        os.close(fd)
        raise SystemExit("CHECKPOINT_REFUSED")
    return fd

def read_owned(path: Path) -> str:
    fd = open_owned(path, os.O_RDONLY)
    with os.fdopen(fd, encoding="utf-8") as handle:
        return handle.read()

for index, chunk in enumerate(chunks):
    marker = root / f"chunk-{index:03d}.ok"
    log_path = root / f"chunk-{index:03d}.log"
    if not marker.exists():
        log_fd = open_owned(
            log_path,
            os.O_WRONLY | os.O_CREAT,
        )
        os.ftruncate(log_fd, 0)
        with os.fdopen(log_fd, "w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-B", "-m", "pytest", "-q", *chunk],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode:
            sys.stdout.write(read_owned(log_path))
            raise SystemExit(result.returncode)
        summaries = [
            line
            for line in read_owned(log_path).splitlines()
            if re.search(r"\b(passed|skipped|warning|warnings)\b", line)
        ]
        if not summaries:
            raise SystemExit(f"SUMMARY_MISSING:{index}")
        marker_fd = open_owned(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        )
        with os.fdopen(marker_fd, "w", encoding="utf-8") as marker_file:
            marker_file.write(summaries[-1] + "\n")
    summary = read_owned(marker).strip()
    print(f"chunk-{index:03d}: {summary}")
    for count, label in re.findall(
        r"(\d+) (passed|skipped|warning|warnings|xfailed|xpassed)", summary
    ):
        totals[label.rstrip("s")] += int(count)
if len(list(root.glob("chunk-*.ok"))) != len(chunks):
    raise SystemExit("CHECKPOINT_INCOMPLETE")
print("FULL_SUITE_TOTALS " + " ".join(f"{key}={totals[key]}" for key in sorted(totals)))
PY
```

### Task 6: Review, close, and integrate

**Files:**
- Modify: `backlog/tasks/task-15743 - Reconcile-current-dev-diagnostic-inventory-drift.md`

- [ ] Obtain independent spec review, fix every finding, and repeat until approved.
- [ ] Obtain independent code-quality/security review, fix every finding, and repeat until approved.
- [ ] Check TASK-15743 acceptance criteria and add concise implementation notes and DoD evidence.
- [ ] Rebase TASK-15743 after its dependencies land, rerun its exact gates, address posted review comments, and require green checks before merge.

ADR required: no

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: this implements the existing ADR-029 metadata-only boundary without changing architecture or persistence policy.

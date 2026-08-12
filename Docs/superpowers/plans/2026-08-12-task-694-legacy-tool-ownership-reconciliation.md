# TASK-694 Legacy Tool Ownership Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close TASK-694 by pinning the current provider ownership of the four legacy capabilities and correcting stale governance without adding, deleting, or behaviorally changing runtime tools.

**Architecture:** Keep the live runtime exactly as it is: `LocalToolProvider` owns `web_search`, the mutually exclusive Library providers own current retrieval, and no provider owns `code_audit`. Add one read-only ownership/import ratchet, then amend the records that still promise a four-tool built-in port; preserve the legacy Python import surface and defer the complete audit-subsystem decision to the expanded TASK-743.

**Tech Stack:** Python 3.11+, pytest, stdlib `subprocess`/`json`, Backlog.md CLI, Markdown governance records, Ruff, mypy, Bandit.

---

## Scope and file map

No production Python file is modified. The relevant production files are read-only mutation targets used to prove the new tests are discriminating.

- Create: `Tests/Agents/test_legacy_tool_ownership.py` — one ownership inventory test and one fresh-process compatibility-import test.
- Modify: `backlog/tasks/task-694 - Reconcile-legacy-tool-ownership-after-System-A-retirement.md` — canonical current task contract, plan link, acceptance criteria, and closeout evidence.
- Modify: `backlog/tasks/task-545 - Wire-built-in-tool-executor-into-MCP-permission-gate.md` — historical scope note records the final replacement ownership instead of an open four-tool port.
- Modify: `backlog/tasks/task-743 - Rehome-file-operation-auditing-off-the-deleted-Settings-side-effect.md` — owns the full audit subsystem and every live file-mutation seam.
- Modify: `backlog/tasks/task-3500 - Align-MCP-perform_rag_search-and-agent-RAGSearchTool-with-profile-driven-retrieval.md` — MCP-only retrieval parity after the agent-side premise is retired.
- Modify: `backlog/tasks/task-1354 - Complete-web_search-and-web_fetch-Console-and-MCP-exposure.md` — truthfully limits public-only target validation to `web_fetch`.
- Modify: `backlog/decisions/032-local-agent-tool-permission-boundary.md` — same `web_fetch`/configured-backend egress correction in the canonical decision.
- Modify: `Docs/superpowers/specs/2026-07-26-retire-system-a-design.md` — append the final audit ownership outcome while retaining the historical observation.
- Modify: `Docs/superpowers/plans/2026-07-26-retire-system-a.md` — correct the completed plan's current follow-up disposition.
- Modify: `Docs/superpowers/specs/2026-08-07-rag-port-p0-foundations-design.md` — distinguish the still-open MCP gap from the already profile-driven agent provider.
- Modify: `Docs/superpowers/plans/2026-08-07-rag-port-p0-foundations.md` — record the corrected TASK-3500 scope without rewriting history.
- Modify: `Docs/Development/Agent-Tools/Claude_Code_File_Audit_System.md` — prominent current-state warning that the described audit is unwired and is not enforcement.
- Read-only mutation targets: `tldw_chatbook/Agents/tool_catalog.py`, `local_tool_provider.py`, `library_tool_provider.py`, `library_rag_tool_provider.py`, and `tldw_chatbook/Tools/__init__.py`.

ADR required: no

ADR path: `backlog/decisions/030-local-library-agent-tool-boundary.md`; `backlog/decisions/032-local-agent-tool-permission-boundary.md`

Reason: TASK-694 introduces no new runtime, storage, provider, permission, egress, or security boundary. It records the provider boundaries already accepted in ADR-030/032. TASK-743 must perform its own ADR check if it keeps or redesigns the audit subsystem.

### Non-negotiable minimality guard

The implementation diff must not modify any file under `tldw_chatbook/`. Do not add `_GATEABLE_BUILTINS`, `[tools]` flags, risk tags, `BUILTIN_HIGH_RISK_TAGS`, `_SHADOWED_BUILTIN_NAMES`, runtime warnings, aliases, or replacement handlers. Do not delete `WebSearchTool`, `RAGSearchTool`, `SearchNotesTool`, `CodeAuditTool`, or the audit hook files in this task.

---

### Task 1: Add the ownership and compatibility ratchet

**Files:**
- Create: `Tests/Agents/test_legacy_tool_ownership.py`
- Read: `tldw_chatbook/Agents/tool_catalog.py:491-525`
- Read: `tldw_chatbook/Agents/local_tool_provider.py:142-258`
- Read: `tldw_chatbook/Agents/library_tool_provider.py:40-78`
- Read: `tldw_chatbook/Agents/library_rag_tool_provider.py:119-153`
- Read: `tldw_chatbook/Tools/__init__.py:38-98`

- [ ] **Step 1: Capture the missing-test RED boundary**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Agents/test_legacy_tool_ownership.py::test_legacy_names_are_absent_and_replacements_have_current_owners \
  Tests/Agents/test_legacy_tool_ownership.py::test_legacy_compatibility_classes_resolve_in_a_fresh_process -q
```

Expected: pytest exits 4 because the new module/nodes do not exist. Record this as coverage RED, not a runtime defect.

- [ ] **Step 2: Write the smallest read-only ownership tests**

Create `Tests/Agents/test_legacy_tool_ownership.py` with this structure:

```python
from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - fixed current-interpreter compatibility probe
import sys
from pathlib import Path

import pytest

from tldw_chatbook.Agents.library_rag_tool_provider import LibraryRagToolProvider
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
    ToolProvider,
    gateable_builtin_tools,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_NAMES = frozenset({"rag_search", "web_search", "search_notes", "code_audit"})


def _catalog_names(provider: ToolProvider) -> set[str]:
    return {entry.name for entry in provider.list_catalog()}


def test_legacy_names_are_absent_and_replacements_have_current_owners(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Agents.local_tool_provider.get_cli_setting",
        lambda *_args, **_kwargs: False,
    )

    assert LEGACY_NAMES.isdisjoint(
        {entry.tool_name for entry in gateable_builtin_tools()}
    )
    assert _catalog_names(BuiltinToolProvider()) == {
        "calculator",
        "get_current_datetime",
    }
    assert "web_search" in _catalog_names(
        LocalToolProvider(workspace_root=tmp_path)
    )
    assert "library_search_notes" in _catalog_names(LibraryToolProvider(object()))
    assert _catalog_names(LibraryRagToolProvider(object())) == {
        "search_library_rag"
    }


def test_legacy_compatibility_classes_resolve_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    code = """
import json
import sys


def deny_external_io(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo", "sqlite3.connect"}:
        raise RuntimeError(f"forbidden import-time I/O: {event}")


sys.addaudithook(deny_external_io)
import tldw_chatbook.Tools as Tools

names = ("WebSearchTool", "RAGSearchTool", "SearchNotesTool")
print(json.dumps({
    name: [getattr(Tools, name).__module__, getattr(Tools, name).__name__]
    for name in names
}, sort_keys=True))
"""
    home = tmp_path / "home"
    home.mkdir()
    env = {
        **os.environ,
        "HOME": str(home),
        "TLDW_CONFIG_PATH": str(tmp_path / "config.toml"),
        "TLDW_TEST_MODE": "1",
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)

    result = subprocess.run(  # nosec B603 - fixed executable and local source
        [sys.executable, "-B", "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "RAGSearchTool": [
            "tldw_chatbook.Tools.rag_search_tool",
            "RAGSearchTool",
        ],
        "SearchNotesTool": [
            "tldw_chatbook.Tools.note_management_tools",
            "SearchNotesTool",
        ],
        "WebSearchTool": [
            "tldw_chatbook.Tools.web_search_tool",
            "WebSearchTool",
        ],
    }
```

The test must list catalogs only. It must not call `invoke` or instantiate a
legacy class. The child audit hook is mandatory because the parent pytest
network guard does not cover a subprocess: any socket resolution/connection or
SQLite connection during the imports must make the child fail. `-B` prevents
bytecode writes; config-file reads remain allowed.

- [ ] **Step 3: Run the new tests GREEN**

Run the exact two-node command from Step 1.

Expected: 2 passed.

- [ ] **Step 4: Prove the tests are discriminating**

Apply each temporary mutation separately, run only the affected node, and immediately restore with the inverse patch before moving on:

1. Add a `GateableTool(..., tool_name="rag_search")` row to `_GATEABLE_BUILTINS` in `Agents/tool_catalog.py`; the inventory node must fail.
2. Temporarily omit `web_search` from `LocalToolProvider.list_catalog`; the inventory node must fail.
3. Temporarily omit `library_search_notes` from `LibraryToolProvider.list_catalog`; the inventory node must fail.
4. Temporarily return no row from `LibraryRagToolProvider.list_catalog`; the inventory node must fail.
5. Remove each of the three relevant `_SUBMODULE_BY_NAME` mappings from `Tools/__init__.py`, one at a time; the fresh-process compatibility node must fail each time.

After restoring every mutation, run:

```bash
git diff --check
git status --short
../../.venv/bin/python -m pytest Tests/Agents/test_legacy_tool_ownership.py -q
```

Expected: only the new test file is modified/untracked; 2 passed.

- [ ] **Step 5: Run the focused provider/compatibility suite**

```bash
../../.venv/bin/python -m pytest \
  Tests/Agents/test_legacy_tool_ownership.py \
  Tests/Agents/test_builtin_file_tools.py \
  Tests/Agents/test_library_tool_provider.py \
  Tests/Agents/test_local_tool_provider.py \
  Tests/Utils/test_optional_import_deferral.py -q
```

Expected: baseline 203 tests plus the two new nodes pass; only characterized dependency warnings are acceptable.

- [ ] **Step 6: Commit the test ratchet**

```bash
git add Tests/Agents/test_legacy_tool_ownership.py
git commit -m "test(tools): pin legacy capability ownership"
```

---

### Task 2: Reconcile the authoritative task and ADR records

**Files:**
- Modify: `backlog/tasks/task-545 - Wire-built-in-tool-executor-into-MCP-permission-gate.md:21,48,68`
- Modify: `backlog/tasks/task-743 - Rehome-file-operation-auditing-off-the-deleted-Settings-side-effect.md:19-31`
- Modify: `backlog/tasks/task-3500 - Align-MCP-perform_rag_search-and-agent-RAGSearchTool-with-profile-driven-retrieval.md:1-32` (Backlog CLI title edit will rename the file)
- Modify: `backlog/tasks/task-1354 - Complete-web_search-and-web_fetch-Console-and-MCP-exposure.md:20-56`
- Modify: `backlog/decisions/032-local-agent-tool-permission-boundary.md:77-90`

- [ ] **Step 1: Capture the stale-contract RED scan**

Run:

```bash
rg -n \
  'risk-tag decision|public-only|agent runtime.s RAGSearchTool|MCP/RAGSearchTool|code_audit itself is covered by TASK-694' \
  'backlog/tasks/task-545 - Wire-built-in-tool-executor-into-MCP-permission-gate.md' \
  'backlog/tasks/task-743 - Rehome-file-operation-auditing-off-the-deleted-Settings-side-effect.md' \
  'backlog/tasks/task-3500 - Align-MCP-perform_rag_search-and-agent-RAGSearchTool-with-profile-driven-retrieval.md' \
  'backlog/tasks/task-1354 - Complete-web_search-and-web_fetch-Console-and-MCP-exposure.md' \
  backlog/decisions/032-local-agent-tool-permission-boundary.md
```

Expected: stale current claims are present. Save the exact matching lines as RED evidence.

- [ ] **Step 2: Record the final TASK-545 outcome**

Amend only the current scope-note/closeout sentences. Preserve what P2 historically did, but replace the open “TASK-694 owns the port/risk tags” conclusion with:

- `web_search` is live under `LocalToolProvider`/ADR-032;
- direct/fallback Library providers own current note/RAG retrieval under ADR-030;
- `code_audit` never became a live System B tool and TASK-743 owns the complete rehome-or-delete decision;
- no four-name built-in port, shadow list, or new risk vocabulary is pending.

- [ ] **Step 3: Expand TASK-743 with Backlog CLI plus a bounded patch**

Keep it To Do. Update its title/description/ACs so the decision covers `CodeAuditTool`, `FileAuditSystem`, `file_operation_hooks.py`, the demo, feature tests, and live docs.

If retained, require coverage of built-in `write_file` and local `fs_write`, `fs_edit`, `fs_patch`; bounded state ownership; provider/model selection; payload-free diagnostics; prompt/content privacy; and proof that observation cannot bypass permissions or workspace confinement. If deleted, require removing implementation, hook, demo, docs, feature tests, and stale references. Require its own ADR check before a retained redesign.

- [ ] **Step 4: Narrow TASK-3500 to MCP only**

Use `backlog task edit 3500 --title 'Align MCP perform_rag_search with profile-driven retrieval'` so the filename and title stay synchronized. Rewrite its description and ACs around MCP `perform_rag_search`, active-profile mode, match semantics, reranking degradation, compatibility, and `_ScoredRow.score_kind`. Remove the agent `RAGSearchTool` ACs because `LibraryRagToolProvider` already delegates to the profile-driven Library service.

- [ ] **Step 5: Correct the web egress statements**

In ADR-032 and TASK-1354, make these exact distinctions:

- both `web_search` and `web_fetch` remain ordinary permission-gated local tools;
- `web_fetch` alone enforces public HTTP(S) target and redirect-hop validation;
- for each `web_search` invocation, the caller/model selects one allowlisted search engine, which determines the destination; the operator supplies supported per-engine credentials and configurable endpoints where available; fixed-endpoint engines remain implementation-defined; a configured Searx endpoint may be local; and `web_search` does not apply public-target validation;
- this is documentation truthfulness only, not an egress-policy change.

Do not alter TASK-1354's Done status or checked ACs.

- [ ] **Step 6: Render and scan the corrected records**

```bash
backlog task 545 --plain
backlog task 743 --plain
backlog task 1354 --plain
backlog task 3500 --plain
rg -n 'web_fetch.*public|configured.*search backend|Searx|perform_rag_search|fs_write|fs_edit|fs_patch' \
  backlog/decisions/032-local-agent-tool-permission-boundary.md \
  backlog/tasks/task-*.md
git diff --check
```

Expected: all tasks render; TASK-743 remains To Do; TASK-1354 remains Done; TASK-3500 is MCP-only; no current record promises a four-tool built-in port or public-only `web_search` transport.

- [ ] **Step 7: Commit the authoritative governance correction**

Stage exactly the five modified governing files (including both sides of
TASK-3500's delete/add rename) and commit:

```bash
git add -- \
  'backlog/tasks/task-545 - Wire-built-in-tool-executor-into-MCP-permission-gate.md' \
  'backlog/tasks/task-743 - Rehome-file-operation-auditing-off-the-deleted-Settings-side-effect.md' \
  'backlog/tasks/task-3500 - Align-MCP-perform_rag_search-and-agent-RAGSearchTool-with-profile-driven-retrieval.md' \
  'backlog/tasks/task-3500 - Align-MCP-perform_rag_search-with-profile-driven-retrieval.md' \
  'backlog/tasks/task-1354 - Complete-web_search-and-web_fetch-Console-and-MCP-exposure.md' \
  backlog/decisions/032-local-agent-tool-permission-boundary.md
git commit -m "docs(tools): reconcile legacy capability ownership"
```

---

### Task 3: Correct historical records and the audit guide

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-26-retire-system-a-design.md:147-155`
- Modify: `Docs/superpowers/plans/2026-07-26-retire-system-a.md:923-931`
- Modify: `Docs/superpowers/specs/2026-08-07-rag-port-p0-foundations-design.md:167-175`
- Modify: `Docs/superpowers/plans/2026-08-07-rag-port-p0-foundations.md:522-532`
- Modify: `Docs/Development/Agent-Tools/Claude_Code_File_Audit_System.md:1-14,124-137`

- [ ] **Step 1: Capture the historical/current-state RED scan**

```bash
rg -n \
  'Porting it properly belongs to TASK-694|code_audit itself is covered by TASK-694|agent-side `RAGSearchTool`|MCP/agent divergence|automatically hooks|Real-time Monitoring' \
  Docs/superpowers/specs/2026-07-26-retire-system-a-design.md \
  Docs/superpowers/plans/2026-07-26-retire-system-a.md \
  Docs/superpowers/specs/2026-08-07-rag-port-p0-foundations-design.md \
  Docs/superpowers/plans/2026-08-07-rag-port-p0-foundations.md \
  Docs/Development/Agent-Tools/Claude_Code_File_Audit_System.md
```

Expected: the stale ownership and wired-audit claims are present.

- [ ] **Step 2: Preserve history while appending the current outcome**

Do not rewrite the original observations. Add explicit current-state amendments:

- System A records: TASK-694 did not port `code_audit`; TASK-743 now owns the entire audit subsystem decision, including local file mutations.
- RAG P0 records: the agent side is already profile-driven through `LibraryRagToolProvider`; TASK-3500 now tracks MCP `perform_rag_search` only.

- [ ] **Step 3: Add the audit guide warning**

Immediately below the title/overview heading, add a prominent note that:

- the described audit subsystem is not wired into the Console agent runtime;
- it does not monitor built-in or local file tools and must not be treated as enforcement or a security control;
- TASK-743 owns the keep/redesign/delete decision;
- the remaining document is retained as historical design/reference.

Change “Automatic Integration” prose so it cannot still promise automatic hooks. Do not delete the detailed reference material in TASK-694.

- [ ] **Step 4: Verify the corrected documentation**

```bash
rg -n 'TASK-743|not wired|not.*security control|LibraryRagToolProvider|MCP.*perform_rag_search' \
  Docs/Development/Agent-Tools/Claude_Code_File_Audit_System.md \
  Docs/superpowers/specs/2026-07-26-retire-system-a-design.md \
  Docs/superpowers/plans/2026-07-26-retire-system-a.md \
  Docs/superpowers/specs/2026-08-07-rag-port-p0-foundations-design.md \
  Docs/superpowers/plans/2026-08-07-rag-port-p0-foundations.md
git diff --check
```

Expected: current-state notices are explicit; historical context remains; no live guide claims automatic auditing.

- [ ] **Step 5: Commit the documentation correction**

```bash
git add \
  Docs/Development/Agent-Tools/Claude_Code_File_Audit_System.md \
  Docs/superpowers/specs/2026-07-26-retire-system-a-design.md \
  Docs/superpowers/plans/2026-07-26-retire-system-a.md \
  Docs/superpowers/specs/2026-08-07-rag-port-p0-foundations-design.md \
  Docs/superpowers/plans/2026-08-07-rag-port-p0-foundations.md
git commit -m "docs(tools): mark legacy audit and rag ownership"
```

---

### Task 4: Final verification and TASK-694 closeout

**Files:**
- Modify: `backlog/tasks/task-694 - Reconcile-legacy-tool-ownership-after-System-A-retirement.md`
- Optional, only if a genuinely new reusable incident occurred: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-backlog-hygiene.md`

- [ ] **Step 1: Fetch and integrate latest dev once**

Require a clean tree, then run:

```bash
git fetch origin dev
git diff --name-only HEAD...origin/dev
git rebase origin/dev
git status --short --branch
```

Inspect any overlap before resolving it. Preserve upstream changes and the reviewed ownership contract; rerun every affected gate after a conflict. Do not perform the rebase while any task mutation is applied.

- [ ] **Step 2: Prove runtime scope stayed empty**

```bash
git diff --name-only origin/dev...HEAD -- 'tldw_chatbook/**/*.py'
```

Expected: no output. Any production Python path is a scope violation and must be removed or brought back through design review.

- [ ] **Step 3: Run behavioral verification**

```bash
../../.venv/bin/python -m pytest \
  Tests/Agents/test_legacy_tool_ownership.py \
  Tests/Agents/test_builtin_file_tools.py \
  Tests/Agents/test_library_tool_provider.py \
  Tests/Agents/test_local_tool_provider.py \
  Tests/Utils/test_optional_import_deferral.py -q
```

Expected: all tests pass. Do not invoke web, RAG, notes, or audit tools merely to test catalog ownership.

- [ ] **Step 4: Run test/static/security checks**

```bash
../../.venv/bin/python -m ruff format --check Tests/Agents/test_legacy_tool_ownership.py
../../.venv/bin/python -m ruff check Tests/Agents/test_legacy_tool_ownership.py
../../.venv/bin/python -m mypy Tests/Agents/test_legacy_tool_ownership.py
../../.venv/bin/python -m bandit -q -s B101 Tests/Agents/test_legacy_tool_ownership.py
../../.venv/bin/python -m compileall -q Tests/Agents/test_legacy_tool_ownership.py
git diff --check origin/dev...HEAD
git diff --check
```

Expected: all changed-code checks are green. If a broad repository baseline is non-green, reproduce the exact node/finding on clean `origin/dev`; never label a baseline failure green.

- [ ] **Step 5: Run the final stale-claim scans**

Search current task/ADR/docs separately from historical records. Confirm:

- no current record promises the four-name built-in port;
- no current record calls `web_search` public-target-only;
- no current record calls legacy `RAGSearchTool` the agent retrieval owner;
- the audit guide says unwired/not enforcement;
- compatibility mappings remain present;
- no successful legacy invocation test was added.

Use:

```bash
rg -n 'rag_search|web_search|search_notes|code_audit|RAGSearchTool|SearchNotesTool|WebSearchTool' \
  backlog Docs Tests tldw_chatbook/Tools/__init__.py
```

Classify every remaining hit; do not blindly delete valid historical or compatibility references.

- [ ] **Step 6: Self-review the exact range**

```bash
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD
```

Check every changed file against the approved design, ADR-030/032, and the five TASK-694 ACs. Confirm no test is vacuous, no provider implementation changed, and no historical record was rewritten as if it had always known the current outcome.

- [ ] **Step 7: Close TASK-694 truthfully**

Only after all gates above are complete:

1. Check all five TASK-694 acceptance criteria with `backlog task edit 694 --check-ac <n>`.
2. Add concise Implementation Notes: ownership map, preserved compatibility imports, expanded TASK-743, narrowed TASK-3500, egress/doc corrections, exact test/static evidence, and the no-production-diff result.
3. Record `ADR required: no` and links to ADR-030/032 in the notes.
4. Add/update a lesson only if execution produced a new, evidence-backed reusable incident; do not invent or duplicate one.
5. Set TASK-694 Done via `backlog task edit 694 -s Done`.
6. Render `backlog task 694 --plain` and confirm every AC checked, plan/notes present, and status Done.

- [ ] **Step 8: Commit closeout**

```bash
git add 'backlog/tasks/task-694 - Reconcile-legacy-tool-ownership-after-System-A-retirement.md'
git commit -m "docs(tools): close legacy ownership reconciliation"
git status --short --branch
```

Expected: clean worktree. Do not push, open a PR, or merge unless separately requested.

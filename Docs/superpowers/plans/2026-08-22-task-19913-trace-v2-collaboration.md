# Trace v2 Collaboration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export and import exhaustive causal traces through an explicit, privacy-safe collaboration workflow.

**Architecture:** Extend the pure export/import seam with a normalized v2 bundle and retain the v1 validator. A dedicated preflight model computes privacy counts and profiles before any file is written; a small Textual dialog selects the profile and confirms explicit full export. Imported snapshots remain ephemeral and read-only.

**Tech Stack:** Python stdlib `json`/`hashlib`, atomic existing writer, Textual dialog patterns, pytest.

**Spec:** `Docs/superpowers/specs/2026-08-22-task-19907-trace-v2-exhaustive-collaboration-design.md`
**ADR required:** yes
**ADR path:** `backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md`
**Reason:** v2 is a portable data/security contract and changes explicit local-data egress behavior.

---

### Task 1: TASK-19913 — V2 bundle, profiles, integrity, and compatibility

**Files:**
- Modify: `tldw_chatbook/Chat/trajectory_export.py`
- Modify: `tldw_chatbook/Chat/trajectory_import.py`
- Test: `Tests/Chat/test_trajectory_export.py`
- Test: `Tests/Chat/test_trajectory_import.py`
- Create: `Tests/Chat/test_trace_v2_collaboration.py`

- [ ] **Step 1: Write failing profile and manifest tests**

Assert safe-summary omission, redacted-diagnostic defaults, full-trace explicit opt-in,
credential prohibition in all profiles, redaction/truncation/missing counts and
per-field states, a synthetic `trace_export` operation event, causal lineage round-trip,
canonical SHA-256 digest, digest mismatch failure, and v1 import.

```python
def test_redacted_diagnostic_is_default() -> None:
    preflight = preflight_trace_export(snapshot)
    payload = build_trace_export(snapshot, preflight=preflight)
    assert payload["format"] == "tldw-trace"
    assert payload["version"] == 2
    assert payload["manifest"]["profile"] == "redacted_diagnostic"
    assert "secret-value" not in json.dumps(payload)
```

- [ ] **Step 2: Confirm failures.**

- [ ] **Step 3: Add pure profile/preflight types**

Use dataclasses/enums in `trajectory_export.py`: `TraceExportProfile` and
`TraceExportPreflight`. One traversal classifies included/redacted/omitted/truncated/
sensitive fields; the writer consumes that result rather than rescanning differently.

- [ ] **Step 4: Build canonical v2 payload and digest**

Serialize normalized Trace events, append the export-operation event, manifest,
lineage, source/missing metadata, and privacy counts. Compute SHA-256 over sorted compact
JSON with the digest field omitted,
then insert it. Keep the existing atomic temp+replace writer.

- [ ] **Step 5: Dispatch import by format/version**

Retain the current v1 path untouched. Add `ImportedTrace` containing snapshot,
manifest, integrity verdict, privacy inventory, and an ephemeral `trace_import`
operation event. Validate v2 structure, digest, profiles, and event references before
building it. Reject higher versions and dangling causal IDs with actionable
`TrajectoryImportError` messages.

- [ ] **Step 6: Verify and mutation-test**

Run: `.venv/bin/pytest -q Tests/Chat/test_trajectory_export.py Tests/Chat/test_trajectory_import.py Tests/Chat/test_trace_v2_collaboration.py`

Delete the digest comparison temporarily and confirm the tamper test fails.

- [ ] **Step 7: Commit**

`git commit -m "feat(trace): add privacy-governed v2 collaboration bundle"`

### Task 2: TASK-19913 — Export preflight and read-only shared Trace UI

**Files:**
- Create: `tldw_chatbook/Widgets/Console/trace_export_dialog.py`
- Modify: `tldw_chatbook/UI/Screens/trajectory_screen.py`
- Modify: `Docs/Features/Trajectory-View.md` or canonical Trace guide
- Create: `Tests/UI/test_trace_export_ui.py`
- Modify: `Tests/UI/test_trajectory_import_ui.py`

- [ ] **Step 1: Write failing dialog tests**

Assert preflight counts, default redacted profile, safe-summary selection, full-profile
warning/confirmation, cancel/no-write, successful atomic file write, and actionable
write failure.

- [ ] **Step 2: Write failing shared-import state tests**

Assert title/state says `READ-ONLY SHARED TRACE`, profile/redaction/integrity metadata is
inspectable, export/import does not mutate any conversation, message, trajectory,
annotation, or agent-run row, and unsupported/tampered bundles show errors.

- [ ] **Step 3: Implement the smallest dialog**

Reuse existing modal/radio/button patterns. Show event count and privacy inventory,
three profiles, destination, Export, and Cancel. Full mode requires a second explicit
confirmation; credentials remain excluded regardless.

- [ ] **Step 4: Add ADR-031 bindings**

Use `x export trace` and rename `o open` to `o import trace`; keep the custom hint line
exactly aligned with implemented non-Escape bindings.

- [ ] **Step 5: Preserve ephemeral import**

Continue constructing `TrajectoryScreen` with no conversation ID/revision providers.
Pass `ImportedTrace` into the screen so shared/read-only state, manifest, integrity,
privacy inventory, and import operation remain inspectable; do not add a DB repository
to the import path.

- [ ] **Step 6: Verify**

Run: `.venv/bin/pytest -q Tests/UI/test_trace_export_ui.py Tests/UI/test_trajectory_import_ui.py Tests/UI/test_trajectory_screen.py Tests/Chat/test_trace_v2_collaboration.py`

- [ ] **Step 7: Commit**

`git commit -m "feat(trace): add export preflight and shared trace UI"`

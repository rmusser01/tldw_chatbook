# Personal Context Quarantine Signal Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user retained native execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent unsupported-record quarantine state from adding model-visible metadata or consuming context budget when the permitted records are unchanged.

**Architecture:** Remove the native hint at the existing context serializer boundary. Whole-record packing depends only on permitted ordered records and current byte/token limits. Preserve the internal authorized-view field, native authority fingerprint, quarantine maintenance and all caller ownership checks; this is not V2 disclosure enforcement.

**Tech Stack:** Python 3.12+, existing Personal Context context service, V1 shared-core models, offline pytest and Console request preparation.

**Spec:** [Accepted disclosure design: Next Send and removal of global quarantine signal](../specs/2026-09-25-personal-context-provider-disclosure-controls-design.md#noninterference-and-user-facing-explanation)

**Backlog:** TASK-25907.11 — Done; user approved native implementation on 2026-09-25.

ADR required: no new ADR; existing ADR-203/182 apply.
ADR path: backlog/decisions/203-personal-context-provider-disclosure-authority.md; backlog/decisions/182-personal-context-memory-evolution.md.
Reason: directly implement the accepted existing serializer privacy correction; no new storage, source, provider, custody, permission or service boundary.

## Global Constraints

- Preserve V1 schemas/canonical bytes and repository/Sync behavior. Do not modify the shared package, database schema, grants, current device-only disclosure behavior, source owners or private quarantine maintenance.
- Remove `unsupported_records_present` from ordinary model serialization; do not replace it with another count, unsupported-kind list, error or existence hint. Existing Next Send selected/omitted rows remain authorized and use the same pass.
- Keep the 12 KiB and ten-percent input-token limits, whole-record packing, existing hard priorities and workspace overrides. Empty selected records produce an empty block and zero profile tokens, including unsupported-only state.
- Preserve native authority fingerprints and freshness/expiry/lock/context-switch revalidation. Matched privacy comparisons concern payload, selected versions, estimated tokens and permitted selection rows; they do not claim invariant internal control revisions or constant-time repository reads.
- Use synthetic profiles and offline request preparation only. No provider/network/model calls, real-profile reads, app launch, UI/styles, migrations, automatic jobs or dependency installs.
- Preserve the frozen 24-case `Tests/Personal_Context/fixtures/memory_baseline_v1.json`, `baseline-v1.json`, `lexical-v1.json` and independent TASK-25907.10/roadmap additions. Write separate create-only evidence. Keep the existing detector for `unscoped_quarantine_signal`; removing detection would hide a regression.
- Run targeted tests only; a full suite requires explicit user opt-in. Retain native execution in the existing isolated worktree/branch and stage only this task's owned files. No push, PR or merge.

## Native preflight and file map

Use `/Users/macbook-dev/.codex/worktrees/personal-context-memory-baseline/tldw_chatbook`, branch `codex/personal-context-memory-baseline`. The shared checkout is not this task's edit target. Confirm native imports before testing:

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -c 'from pathlib import Path; import tldw_chatbook.Personal_Context.context_service as m; assert Path(m.__file__).resolve().is_relative_to(Path.cwd().resolve()); print(m.__file__)'
```

| File | Responsibility |
| --- | --- |
| `tldw_chatbook/Personal_Context/context_service.py` | Stop forwarding/serializing the hint and remove hint-dependent packing/empty-block behavior |
| `Tests/Personal_Context/test_context_service.py` | Paired positive/negative production snapshot and selection contracts |
| `Tests/Chat/test_console_personal_context_snapshot.py` | Real first-request preparation consumes the resulting empty/allowed block without provider execution |
| `Tests/Personal_Context/test_memory_baseline.py` | Real encrypted synthetic h11 quarantine regression and preserved allowed control |
| `Docs/superpowers/reviews/evidence/personal-context-memory/quarantine-signal-v1.json` | New reproducible post-fix report; no historical overwrite |
| `Docs/superpowers/reviews/2026-09-25-personal-context-quarantine-signal-removal-review.md` | Actual RED/GREEN, targeted receipts, review findings, acceptance and limits |
| `backlog/docs/personal-context-memory-evaluation.md`, roadmap and TASK-25907.11 | Measured one-gap improvement and unchanged broader guarantees |

Existing inspected interfaces: `ProfileContextService.build_snapshot(request) -> ProfileContextSnapshot`, `build_explained_snapshot(request) -> ProfileContextBuildResult`, `build_console_first_request_plan(...)` through Chat test `_plan`, and offline `run_suite(manifest_path, root)` / `write_report(report, output)`. The fake authorized-view hint comes from the existing `_view(..., unsupported=...)` helper; do not remove its ability to exercise the boundary.

## Unit 1: Freeze the privacy behavior and correct the native serializer

**Files:** Modify context service and its test module. No new abstraction, dependency, repository read or service API is needed.

**Interfaces:** `_render_json(records: list[dict[str, object]]) -> str` renders only `records`; `_serialize_whole_records` retains every existing argument except its private `unsupported_records_present` parameter. Public snapshot/explanation types and `AuthorizedProfileContextView.unsupported_records_present` remain unchanged.

- [x] **Step 1: Replace the old expected-leak test and add paired failing controls.** In `test_context_service.py`, replace `test_unknown_newer_records_add_only_an_opaque_indicator` with the empty-case assertion below. Add the paired test using its existing `_record`, `_view`, `_ViewService`, `NOW` and request imports:

```python
@pytest.mark.parametrize("available", [100, 8_000])
def test_unsupported_only_does_not_create_model_context(available, monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Personal_Context.context_service.estimate_tokens",
        lambda text, **_kwargs: 0 if not text else (len(text) + 3) // 4,
    )
    source = _ViewService(_view((), unsupported=True))
    result = ProfileContextService(source, clock=lambda: NOW).build_explained_snapshot(
        ProfileContextRequest(current_user_text="x", available_input_tokens=available)
    )
    assert len(source.calls) == 1
    assert result.snapshot.serialized_block == ""
    assert result.snapshot.source_version_ids == ()
    assert result.snapshot.estimated_tokens == 0
    assert result.explanation is not None and result.explanation.rows == ()
    assert result.explanation.state == "empty"

@pytest.mark.parametrize("available", [0, 100, 400, 20_000])
def test_quarantine_hint_cannot_change_permitted_packing(available, monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Personal_Context.context_service.estimate_tokens",
        lambda text, **_kwargs: 0 if not text else (len(text) + 3) // 4,
    )
    records = (
        _record(record_id="allowed-small", subject="brief", value="concise"),
        _record(record_id="allowed-large", subject="memo", value="abc" * 500),
        _record(record_id="private", visibility=AgentVisibility.USER_ONLY,
                subject="private", value="PRIVATE_CANARY"),
    )
    request = ProfileContextRequest(current_user_text="brief", available_input_tokens=available)
    results = []
    for unsupported in (False, True):
        source = _ViewService(_view(records, unsupported=unsupported))
        builder = ProfileContextService(source, clock=lambda: NOW)
        result = builder.build_explained_snapshot(request)
        assert len(source.calls) == 1
        assert result.snapshot == builder.build_snapshot(request)
        results.append(result)
    assert results[0] == results[1]
    assert results[0].explanation is not None
    for result in results:
        assert result.explanation.state != "unavailable"
        assert "unsupported_records_present" not in result.snapshot.serialized_block
        assert "PRIVATE_CANARY" not in result.snapshot.serialized_block
        assert all(r.record_id != "private" for r in result.explanation.rows)
    if available == 20_000:
        assert "concise" in results[0].snapshot.serialized_block
        assert "version-allowed-small" in results[0].snapshot.source_version_ids
```

The hint-dependent overhead previously affects available context budget as well as the obvious field. The 20,000-token control must produce an actual allowed record; a generic failed/disabled builder cannot satisfy these tests. Fake internal revisions are fixed for the pair; native revision changes are not being asserted invariant. The deterministic estimator keeps these privacy controls offline; existing provider-aware budget tests remain in the targeted module.

- [x] **Step 2: Run the new controls RED.**

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest Tests/Personal_Context/test_context_service.py -k 'unsupported_only or quarantine_hint' -q
```

Expected: failures from emitted metadata/nonempty unsupported-only block or changed packing; not fixture/schema/import failures. Record the exact failures before editing production code.

- [x] **Step 3: Remove hint forwarding and serialization only.** Delete `unsupported_records_present=view.unsupported_records_present` from `_snapshot_from_view`'s private packing call. Replace `_render_json` with:

```python
@staticmethod
def _render_json(records: list[dict[str, object]]) -> str:
    return _CONTEXT_HEADER + json.dumps(
        {"records": records},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
```

Indent the decorator and signature normally within the class. In `_serialize_whole_records`, remove the private hint parameter; use `empty = cls._render_json([])`, `candidate = cls._render_json([*selected, candidate_record])`, `if not selected: return "", ()`, and `return cls._render_json(selected), tuple(versions)`. Leave eligibility/order/query compilation, both budget checks, decision rows and token estimation unchanged. No change to service.py, private quarantine data or native authority hash.

- [x] **Step 4: Run the entire context test module GREEN**, then confirm `rg -n unsupported_records_present tldw_chatbook/Personal_Context/context_service.py` has no production serializer reference. Inspect authorized-view consumers to ensure no compatibility field was removed.

## Unit 2: Verify real native consumers and frozen synthetic evidence

**Files:** Modify the Chat snapshot and baseline test modules. These tests exercise the production entry points without executing a provider.

- [x] **Step 1: Add the real offline h11 control.** In `test_memory_baseline.py`, reuse its existing `completed_report` and `row` helper:

```python
def test_quarantine_case_has_no_unscoped_signal(completed_report):
    case = row(completed_report, "h11")
    assert case["positive_control_passed"] is True
    assert case["selected_labels"] == ["control"]
    assert case["harness_errors"] == case["authority_failures"] == []
    assert "unscoped_quarantine_signal" not in case["policy_gaps"]
    assert all(gap["code"] != "unscoped_quarantine_signal"
               for gap in completed_report["policy_gaps"])
```

This report seeds a real encrypted synthetic repository and calls the real service/tool/snapshot paths. Keep the baseline detector and case fixture unchanged. Its `device_only_in_provider_context` observation remains honest.

- [x] **Step 2: Add the Console consumer regression.** In `test_console_personal_context_snapshot.py`, reuse `_plan` and imported `SimpleNamespace`; import `ProfileContextService` and `AuthorizedProfileContextView`. Build an actual context service over an authorized empty view:

```python
def test_first_request_omits_unsupported_only_profile(monkeypatch):
    from tldw_chatbook.Personal_Context.context_service import ProfileContextService
    from tldw_chatbook.Personal_Context.service import AuthorizedProfileContextView
    monkeypatch.setattr(
        "tldw_chatbook.Personal_Context.context_service.estimate_tokens",
        lambda text, **_kwargs: 0 if not text else (len(text) + 3) // 4,
    )
    view = AuthorizedProfileContextView(
        generation=1, record_set_revision="manifest-v1", workspace_scope_id=None,
        authority_revision="authority-v1", records=(), unsupported_records_present=True,
    )
    source = SimpleNamespace(authorized_context_view=lambda **_kwargs: view)
    plan = _plan(ProfileContextService(source))
    assert plan.profile_context_snapshot.serialized_block == ""
    assert plan.config.personal_context_block == ""
    from tldw_chatbook.Agents.agent_service import append_personal_context
    assert append_personal_context("BASE", plan.config.personal_context_block) == "BASE"
    assert append_personal_context("BASE", PROFILE_BLOCK) == "BASE\n\n" + PROFILE_BLOCK
```

Pair this with the existing `test_first_request_plan_builds_one_snapshot_and_pins_exact_block` positive control and context Unit 1's actual allowed record. The two assertions call the production system-message append seam used by both AgentService request paths: an empty real Console block preserves base bytes, while the allowed block is appended. No provider is dispatched.

- [x] **Step 3: Run the affected target set once after changes.**

```bash
TLDW_MEMORY_BASELINE_REPORT=Docs/superpowers/reviews/evidence/personal-context-memory/quarantine-signal-v1.json PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest Tests/Personal_Context/test_context_service.py Tests/Personal_Context/test_memory_baseline.py Tests/Chat/test_console_personal_context_snapshot.py Tests/Chat/test_console_memory_selection.py Tests/Agents/test_profile_tool_provider.py -q
```

Preserve existing scope/private/expiry/conflict, provider-aware token bounds, root snapshot and explanation freshness checks. Add another module only if an actual new dependency/failure warrants it; do not run a full sweep.

- [x] **Step 4: Verify the new create-only report and independent synthetic-root reproduction.** Step 3 uses the existing pytest export test; its existing reproducibility test runs the same suite on a second root and compares the full report. Retain pytest config/keyring/network isolation rather than importing the harness in a bare interpreter. Then inspect the exported JSON:

```python
import json
from pathlib import Path
previous = json.loads(Path("Docs/superpowers/reviews/evidence/personal-context-memory/lexical-v1.json").read_text())
a = json.loads(Path("Docs/superpowers/reviews/evidence/personal-context-memory/quarantine-signal-v1.json").read_text())
assert a["fixture_sha256"] == previous["fixture_sha256"]
assert a["summaries"] == previous["summaries"]
assert a["harness_errors"] == a["authority_failures"] == a["context_checks"] == []
assert not any(gap["code"] == "unscoped_quarantine_signal" for gap in a["policy_gaps"])
assert any(gap["code"] == "device_only_in_provider_context" for gap in a["policy_gaps"])
assert a["disclosure_checks_passed"] is False
```

If retrieval summaries differ unexpectedly, investigate instead of blessing new labels/fixtures. If the report path already exists, inspect its provenance and use another explicitly named create-only output; never overwrite historical evidence. The source fixture and previous report hashes must remain unchanged.

## Unit 3: Static checks, review and concrete closeout

**Files:** New execution review and evidence report; update evaluation docs, roadmap, this plan and task .11. No UI or ADR schema edits.

- [x] **Step 1: Check formatting and changed-line lint.** From the worktree, run:

```bash
.venv/bin/python -m ruff check tldw_chatbook/Personal_Context/context_service.py Tests/Personal_Context/test_context_service.py Tests/Personal_Context/test_memory_baseline.py
.venv/bin/python -m ruff format --check tldw_chatbook/Personal_Context/context_service.py Tests/Personal_Context/test_context_service.py Tests/Personal_Context/test_memory_baseline.py
.venv/bin/python -m ruff check --output-format json Tests/Chat/test_console_personal_context_snapshot.py
.venv/bin/python -m ruff format --diff Tests/Chat/test_console_personal_context_snapshot.py
git diff --unified=0 -- Tests/Chat/test_console_personal_context_snapshot.py
git diff --check
```

Require whole-file Ruff/format success for the three Personal Context files. The older Chat module has pre-existing formatting/lint debt: compare its JSON diagnostics and formatting diff with those obtained by piping its unchanged pre-implementation Git blob into Ruff's stdin modes, using the exact original path as --stdin-filename. Require zero diagnostics or formatting changes intersecting added/modified lines, and no new diagnostic outside them; preserve unrelated baseline debt. Record the comparison's exact starting commit, commands, exit codes and excluded baseline findings. Do not silently treat a nonzero whole-file Chat result as a pass.
- [x] **Step 2: Self-review and dispatch one bounded read-only fix review** under requesting-code-review. Check model/ordinary-preview payload, unsupported-only emptiness, same-pass packing, successful native controls, preserved native authority checks and V1 schema/fixture boundaries. Resolve verified Critical/Important findings with targeted RED/GREEN tests; do not launch a recurring audit loop.
- [x] **Step 3: Record actual test counts, separate deterministic report and remaining limits.** Explain that one unscoped metadata leak is closed; device-only model egress, V2 source evidence, suppression, destination enrollment, source-derived Notes custody and background recovery are still unimplemented future controls. Do not describe the fix as complete disclosure/privacy enforcement. Preserve original baseline reports and independent evaluation additions.
- [x] **Step 4: Verify all six task criteria and original 59 criteria, native import/worktree boundary, links, unique family IDs and backward dependencies.** Use Backlog CLI for checks/notes/status. Only then mark .11 Done, update the roadmap and plan checkboxes, stage explicit owned paths (only the roadmap prefix) and commit locally. No push/PR/merge is included.

## Planning checkpoint and execution scope

This is the concrete next native implementation proposal, after the original four implemented slices and five accepted design contracts. No production code or runtime test was changed/run while preparing this plan. The prior native execution preference persists; there is no need to choose an execution mechanism again. The user subsequently approved this bounded model-payload fix. That execution approval does not activate the broader V2 workflow.

## Plan self-review

- The accepted ADR-203 removal requirement is covered by Units 1–2; broader disclosure, enrollment and V2 requirements remain explicitly outside this independently testable slice.
- Native signatures, Console helper, system-message append seam, zero-token request validation, frozen h11 fixture and create-only report writer were inspected. The spec anchor was corrected and repr-only privacy assertions replaced with actual payload/selection assertions.
- Unit 3 distinguishes full-file static checks for the Personal Context files from the older Chat module's baseline debt; it cannot certify that debt as fixed.
- This checkpoint changes documentation/tracker files only. Execution receipts and the new JSON report are future deliverables, not planning evidence.

## Native planning validation

Scoped checks passed across 24 owned documents: 171 local Markdown links,
60 task reference/documentation paths, 12 unique family IDs and all 59
unchanged original child criteria. TASK-25907.11 retains six unchecked criteria;
TASK-25907.9 is Done with its six approved design criteria checked. All five
Python example blocks parse; this is syntax inspection, not execution evidence.

Allocation was checked against 559 available refs and 77 worktrees, without
fetching or checking open PRs; integration must recheck concurrent allocation.
The independent TASK-25907.10 file/roadmap suffix and frozen fixture/historical
reports retain their original bytes. Production, test and shared-core files
have no diff. Whitespace passes. No runtime tests were run at this checkpoint.

## Execution closeout

The approved native implementation is complete. The affected five-module run
passed 172 cases; the three existing root/child prompt controls passed separately
(175 distinct targeted cases). A reviewer-requested empty-state assertion passed
a focused 3-case repeat without changing production code. The separate
24-case report reproduced byte-for-byte, with unchanged retrieval/selections
and the remaining device-only disclosure gap reported honestly.

The report-export example was corrected to use the existing isolated pytest
hook/reproduction test, and root/child verification was added to cover criterion
4 explicitly. All four changed files pass formatting/changed-line lint;
three Personal Context files pass whole-file Ruff. The older Chat module's two
baseline findings remain outside modified lines, documented in the
[execution review](../reviews/2026-09-25-personal-context-quarantine-signal-removal-review.md).
All six task criteria are checked and implementation notes are recorded. No
additional lesson was filed: the isolation and successful-control traps are
already recorded in the repository's testing/live-verification lessons.

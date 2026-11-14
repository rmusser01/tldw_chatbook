# Personal Context Next Send Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user retained native execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain why agent-eligible Personal Context records appear in, or are omitted from, one captured Next Send preview without exposing ineligible records or changing the model request.

**Architecture:** The existing `ProfileContextService` builds one snapshot and a separate, ephemeral explanation during the same eligible-candidate ordering and whole-record packing pass. The Console preview passes that explanation through a request-local callback to the inspector, while its existing `ConsoleContextSnapshot` and provider payload remain unchanged. The inspector validates ownership before publication, clears results on replacement, and uses one expiry timer instead of repository polling.

**Tech Stack:** Python 3.12+, Textual 8.x, existing encrypted Personal Context repository, frozen in-memory projection types, and the existing Console preview path.

**Spec:** [Approved memory evolution design, section C](../specs/2026-09-25-personal-context-memory-evolution-design.md#c-explain-selection-in-next-send)

**Backlog:** TASK-25907.3 — Done. Seven criteria checked after evidence and review.

**Status:** Native implementation complete. [Execution review](../reviews/2026-09-25-personal-context-next-send-selection.md) records evidence and limits.

ADR required: yes — existing decisions apply; no new ADR is needed.
ADR path: backlog/decisions/182-personal-context-memory-evolution.md; backlog/decisions/088-console-lightweight-next-send-history-projection.md; backlog/decisions/150-design-token-system-and-design-language.md
Reason: ADR-182 owns the explanation's privacy and selection contract; ADR-088 governs read-only Next Send presentation; ADR-150 governs the existing inspector. No canonical schema, provider or sync contract changes.

## Global constraints

- Preserve the public `ProfileContextSnapshot`, `ConsoleContextSnapshot`, root-run pinning and provider injection contracts. New diagnostics never enter `next_send_payload`, a general snapshot dataclass, run logs, chat persistence, debug representations, Sync or exports.
- Explain only candidates that have already passed agent visibility, scope, state, conflict and expiry checks. Do not echo `unsupported_records_present`, quarantine counts or the existence of ineligible records.
- Preserve the 12 KiB UTF-8 block ceiling, ten-percent token ceiling, existing five priority groups, keyed workspace overrides and greedy whole-record packing; a later small candidate can still fit after an oversized one.
- Disabled/locked states require the exact typed authority outcome of the same build. An empty snapshot or unknown failure yields empty/insufficient-budget or unavailable as appropriate, never a guessed authority state.
- Bind the sidecar to one captured session/workspace, draft, attachments/staged sources, resolved provider/model, reserved input budget, profile identity/revisions and eligibility time. The four-part snapshot `cache_key` is insufficient alone. Do not persist a hash of prompt text.
- Clear the previous sidecar before replacement work, on modal suspension/dismissal and on owner change. Reject late result publication. One timer clears at the earliest eligible expiry boundary; no continuous repository-decryption polling.
- Use the existing inspector layout and design tokens. Read `backlog/docs/design-language.md` before modifying UI or styles. Imported identifiers render literally and boundedly.
- Use synthetic repositories and mounted tests; targeted runs only. No full suite without user opt-in. Preserve the isolated worktree and unrelated shared checkout.

## Considered approaches

1. **Recommended: request-local sidecar from the existing selection pass.** One service call returns the unchanged snapshot and a separately owned explanation. The bridge's preview-only wrapper captures it, and the inspector alone renders it. Cost: additive callback plumbing across the preview path, offset by strict separation from generic snapshot serialization.
2. **Re-run selection in the UI.** Simpler wiring, but requests can differ by clock, budget or authority between passes; it cannot truthfully explain the snapshot. Rejected.
3. **Add diagnostic fields to `ProfileContextSnapshot` or `ConsoleContextSnapshot`.** Convenient transport, but generic dataclass serialization and copied payload/debug paths could retain sensitive candidate identifiers. Rejected.

## Task 1: One service-owned snapshot and explanation pass

**Files:** `tldw_chatbook/Personal_Context/context_service.py`, `tldw_chatbook/Personal_Context/service.py`, `Tests/Personal_Context/test_context_service.py`; a new focused service test file is allowed if the existing file becomes unwieldy.

**Interfaces:** `ProfileContextService.build_explained_snapshot(request: ProfileContextRequest) -> ProfileContextBuildResult`; `ProfileContextBuildResult.snapshot` is the existing `ProfileContextSnapshot`; `.explanation` is a frozen, repr-suppressed `ProfileContextSelectionExplanation` with state, eligible candidate rows, a captured owner token and earliest eligible expiry. `build_snapshot()` still returns only `ProfileContextSnapshot` through the same internal build. `ProfileContextService.explanation_is_current(explanation, request) -> bool` checks captured profile identity/authority/revisions and time through the authorized owner, off the UI thread and only at publication. A typed locked/disabled outcome is rechecked against the same owning service's current status; unknown failures can only remain unavailable. `AuthorizedProfileContextView` gains an internal, defaulted profile identity field; its existing callers and public serialized block remain compatible.

- [x] **Step 1: Write failing service tests.** Make a synthetic authorized view with a private canary, an expired canary, workspace/global keyed pair, an oversized eligible record and a later small record. Assert exactly selected, overridden and budget-omitted eligible IDs and priority groups; absent private/expired/conflicted IDs and counts; identical snapshot bytes and `source_version_ids` to `build_snapshot()` for a fixed clock. Include Unicode byte and provider-token boundaries, zero/header-only budget, unavailable versus typed locked/disabled, and record/profile revision changes without reuse of the old `cache_key` alone.

```python
result = ProfileContextService(source, clock=lambda: NOW).build_explained_snapshot(request)
assert result.snapshot == ProfileContextService(source, clock=lambda: NOW).build_snapshot(request)
assert {row.record_id for row in result.explanation.rows}.isdisjoint(hidden_ids)
assert "unsupported_records_present" not in repr(result.explanation)
```

- [x] **Step 2: Run the new focused tests RED** with `.venv/bin/python -m pytest Tests/Personal_Context/test_context_service.py -q -k 'explanation or diagnostic'`; require missing-interface/behavior failures, not fixture errors.
- [x] **Step 3: Implement the minimal shared traversal.** Split ordering into the same selected order plus eligible override rows. Let the existing serializer optionally collect per-candidate selected/byte-budget/token-budget decisions while it packs; never select from a second pass. Capture the typed build outcome at the authorized-view boundary. Keep IDs, owner tokens and timestamps out of `repr`, and no canonical or model-field changes.

```python
def build_snapshot(self, request: ProfileContextRequest) -> ProfileContextSnapshot:
    return self._build(request, explain=False).snapshot

def build_explained_snapshot(self, request: ProfileContextRequest) -> ProfileContextBuildResult:
    return self._build(request, explain=True)
```

- [x] **Step 4: Run the full `Tests/Personal_Context/test_context_service.py` selection GREEN** and relevant `Tests/Personal_Context/test_service.py` cases. Check service-side Ruff, formatter, and whitespace. Commit this unit with only owned files.

## Task 2: Carry the preview-only sidecar without changing model snapshots

**Files:** `tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/Chat/test_console_personal_context_snapshot.py`, targeted new/extended screen-factory tests.

**Interfaces:** The bridge's `build_personal_context_preview_snapshot` accepts an optional `selection_sink: Callable[[ProfileContextService, ProfileContextRequest, ProfileContextSelectionExplanation], None]`, wraps its `profile_context_service` only for that call, and still returns a plain `ProfileContextSnapshot`. The controller's `build_context_snapshot` accepts an optional sink and forwards it only through the existing preview builder. ChatScreen's Next Send factory returns an inspector-only wrapper containing the unchanged `ConsoleContextSnapshot`, the explanation and a one-shot async validation callback; ordinary callers continue receiving the old snapshot. The wrapper has no generic serializer or useful `repr`.

- [x] **Step 1: Write failing integration tests** at the actual `build_console_first_request_plan` and `build_context_snapshot` entry points. Pin one builder invocation and exact block parity for identical inputs/clock. Assert diagnostics never enter the model messages, `next_send_payload`, `dataclasses.asdict(ConsoleContextSnapshot)`, `_format_export_text`, logs or Sync; fakes may stand in only for unrelated provider/network work. Export and lifecycle assertions are completed in Task 3, where the inspector owns those paths.
- [x] **Step 2: Run those focused tests RED**, requiring the missing request-local sidecar interface.
- [x] **Step 3: Implement the preview-only adapter** and capture one immutable owner token at the Screen factory. The sink carries the exact `ProfileContextRequest`, including resolved provider/model and reserved input budget, plus the owning builder; the factory compares the live session/workspace, draft, attachments/staged source identity and resolved target against that capture. Changes to the request's budget sources invalidate the captured request through the Console context/session revision boundary; no cache-key-only reuse is allowed. Call the builder's current-check off the UI thread. A mismatch produces content-free unavailable and never reuses the prior sidecar. Do not re-run profile selection to make an explanation.
- [x] **Step 4: Run the focused Chat and Screen factory tests GREEN**, plus existing root-run pinning and non-agent-preview tests. Check Ruff/format/whitespace, then commit only the owned adapter files.

## Task 3: Mount the disposable Next Send explanation

**Files:** `tldw_chatbook/Widgets/Console/console_conversation_inspector.py`, `Tests/UI/test_console_conversation_inspector.py`; touch a source CSS module and rebuild the bundle only if an existing token-backed pattern cannot express the panel.

**Interfaces:** The inspector accepts its existing snapshot factory result or the inspector-only wrapper. It stores the sidecar in an instance variable, never on `ConsoleContextSnapshot`; `_load_snapshot()` clears it before awaiting replacement and publishes it only when the modal, captured session and load generation remain current. One-shot expiry clears it. Copy, Save to File, Raw JSON and provider payload continue to read the unchanged snapshot.

- [x] **Step 1: Write failing mounted tests** for selected priority labels, workspace override, byte/token omission, empty/insufficient-budget/typed disabled/locked/unavailable states, literal bounded identifiers and narrow keyboard access. Gate an old worker, start Refresh or dismiss/suspend, then release the old result and prove it cannot repopulate details. Advance a controlled clock through expiry without changing manifest revision and prove the explanation clears. Verify export, raw JSON and debug `repr` omit diagnostic canaries.
- [x] **Step 2: Run those tests RED** and confirm absence of the UI and lifecycle behavior, not an unrelated mount failure.
- [x] **Step 3: Add one collapsed “Personal Context selection” section** to the existing Next Send pane, with labels from application code and literal bounded values. Start/refresh, invalidation, dismissal and expiry all clear the sidecar before replacement or exit. Keep the existing payload sections and actions intact; the explanation never appears on Costs, Exchange or Current.
- [x] **Step 4: Run mounted inspector and design-token/CSS consistency tests GREEN.** Use production stylesheet hosts at wide and 60-column narrow sizes, inspect synthetic renders, and check changed-file Ruff/format/whitespace. Commit only the owned UI/CSS/test files. A narrow-layout scroll fix and duplicate-callback fail-closed guard also touch the Screen factory and its targeted test.

## Task 4: Integrated evidence, review and tracker closeout

**Files:** This plan, TASK-25907.3, `backlog/docs/personal-context-memory-roadmap.md`, and a new execution review in `Docs/superpowers/reviews/`. Synthetic renders, if retained, belong under `Docs/superpowers/reviews/evidence/personal-context-memory/`.

- [x] **Step 1: Run affected service, Chat and mounted UI tests** after the last integration change, including preview/prepared-request parity, zero/header budgets, Unicode, overrides, later-fitting record, stale owner, lock/removal and expiry. Avoid a redundant unchanged rerun.
- [x] **Step 2: Review the full branch once with a fresh reviewer** against privacy, sidecar serialization, exact request ownership, no second selection, deadline/late-result handling and existing provider/preview behavior. Re-grade findings by user effect; fix blocking issues with RED→GREEN tests, record deferred minors and reviewer exclusions.
- [x] **Step 3: Run final changed-file static checks, generated CSS guard if CSS changed, `git diff --check`, local Markdown links, ten task-family IDs, 59 unchanged child acceptance texts and backward dependencies.** Record the exact evidence and limitations. Keep the existing `unsupported_records_present` model-block gap visible as ADR-182 follow-up, without claiming this task repairs it.
- [x] **Step 4: Check all seven criteria against results, add implementation notes and ADR links through Backlog CLI, mark Done only when its definition of done is met, update the roadmap and commit owned documentation.** Keep the branch local; no push, PR or merge is part of this task.

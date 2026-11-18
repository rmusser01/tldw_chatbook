# Personal Context Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Native execution is the user's preserved choice. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user inspect the provenance actually retained for a profile record or pending proposal, including explicit limits and metadata-only deleted records.

**Architecture:** The existing PersonalContextService produces immutable Settings-only projections from authenticated current objects. A shared, disposable provenance section renders those projections in My Profile and the existing proposal review. Its read lifecycle is separate from proposal approval and existing record mutations.

**Tech Stack:** Python 3.12+, existing Textual 8.x, frozen dataclasses, standard-library hashlib/time/unicodedata, existing encrypted SQLite repository and shared canonical serialization; no dependency or schema change.

**Spec:** [Approved design, section B](../specs/2026-09-25-personal-context-memory-evolution-design.md#b-inspect-existing-provenance)

**Backlog:** TASK-25907.2 — Done. All seven acceptance criteria verified.

**Status:** Native implementation complete on 2026-09-25. The [execution review](../reviews/2026-09-25-personal-context-provenance-execution-review.md) records 246 distinct targeted tests, five repeated full-CSS checks, independent review, decisions and one deferred minor recovery issue.

ADR required: yes — existing decisions apply; no new ADR needed for this slice.
ADR path: backlog/decisions/182-personal-context-memory-evolution.md; backlog/decisions/150-design-token-system-and-design-language.md
Reason: ADR-182 already approves Settings-owned provenance projections, unknown-history labels and lifecycle fences. ADR-150 governs the existing Settings and modal surfaces. This plan introduces no canonical storage, permission, source-resolution or approval contract.

## Global Constraints

- Use existing canonical fields; canonical compatibility fixtures must remain unchanged.
- No source/history resolution, Undo lookup, model call, agent tool, persistent derivative, export field or logging of provenance.
- Settings may inspect user-owned private records. Existing agent search/get, context, privacy controls and proposal acceptance retain their behavior.
- Exact copy: **No source reference retained**; **Legacy source reference — quotation not verified**; **Record changed; reload details**; **Edit history not recorded**.
- Imported metadata is bounded literal text: no Rich/Markdown interpretation, terminal controls or automatic links.
- Work stays in the existing isolated worktree. Preserve the shared checkout and unrelated changes. Native execution remains selected.
- Read `backlog/docs/design-language.md` before UI work. New visual values use existing `$ds-*` tokens. Rebuild generated CSS from its source modules.
- Use synthetic encrypted repositories and offline mounted UI fixtures. Run targeted checks only; no full suite or real profile.

## Review Focus

1. Accept-edited and later Settings edits preserve ambiguous provenance: Unit 1 proves the view never claims unchanged acceptance, inferred authorship, semantic support or durable confidence.
2. Profile replacement can reuse selected IDs, and proposals have no canonical revision field: Units 1–2 test profile identity, purge generation and a whole-proposal fingerprint, not only target/base IDs.
3. Worker cancellation does not stop a thread: Unit 2 tests old completions after selection, lock, removal, replacement and unmount, including a stalled reader.
4. A tombstone can retain provenance while its content has been retired: Units 1–2 prove metadata is inspectable without loading Undo, old bodies or source text, and cannot enter edit/restore actions.
5. Imported IDs can contain markup, links or misleading directional/control characters: Unit 2 tests literal rendering, explicit truncation, Unicode text and keyboard access at narrow sizes.

## UI contract

- Job: answer “what was recorded about where this item came from?” while viewing My Profile or deciding a pending proposal.
- Keep the existing dense, keyboard-driven Settings layout. Add one collapsed **Recorded provenance** section after the selected record list and inside the proposal review's existing scroll area. Expanding it starts inspection; collapsing it disposes the detail.
- My Profile gets a separate collapsed **Deleted record metadata** list. Labels use only a bounded ID and recorded update time. Selecting one follows the existing editor's draft-discard behavior, clears the live-record selection and disables record mutation actions. It does not offer Restore or Undo, and must not silently discard a draft if the current editor requires confirmation.
- Order: recorded source/actor/reason; timestamps and current state; object/version/promotion IDs; evidence availability and bounded references/hashes; unknown edit/inference history. Pending proposal metadata is labelled as proposal metadata, never as an approved record.
- A changed record offers **Reload details**, which reloads the parent Settings selection. A changed/resolved proposal requires closing and reopening its review; never replace the proposal or discard typed edits underneath the user.
- No new hotkeys. Sections and buttons remain reachable by Tab, Enter and existing scrolling. Wrap long literal values; do not widen the screen to fit IDs.

## Interfaces and ownership

Create `tldw_chatbook/Personal_Context/settings_provenance.py` for frozen, slotted, in-memory types and pure projection helpers. Keep all IDs, hashes and values out of dataclass reprs.

| Type | Fields / purpose |
| --- | --- |
| `SettingsProfileIdentity` | `profile_id: str`, `purge_generation: int`; authenticated manifest identity, not a permission grant |
| `SettingsProvenanceSubject` | `profile: SettingsProfileIdentity`, `object_type: Literal["record", "proposal"]`, `object_id: str`, `version_token: str`; captured selection identity |
| `SettingsDeletedRecord` | `subject: SettingsProvenanceSubject`, `scope_id: str`, `updated_at: datetime`; no payload, previous title or controls |
| `SettingsProvenanceField` | `label: str`, `value: str`; labels come from application code |
| `SettingsProvenanceProjection` | `subject: SettingsProvenanceSubject`, `fields: tuple[SettingsProvenanceField, ...]`, `reference_status: str`, `source_references: tuple[str, ...]`, `source_hashes: tuple[str, ...]`; metadata only |
| `SettingsProvenanceResult` | `state: Literal["available", "changed", "unavailable"]`, `projection: SettingsProvenanceProjection | None = None`; non-available results contain no previous detail |

Pure helpers:

- `provenance_subject(profile: SettingsProfileIdentity, value: ProfileRecord | ProfileProposal) -> SettingsProvenanceSubject`
- `project_provenance(subject: SettingsProvenanceSubject, value: ProfileRecord | ProfileProposal) -> SettingsProvenanceProjection`

For records, the token is the current `version_id`. For proposals, use SHA-256 of existing `canonical_bytes(proposal)`: `base_version_id` describes its target, not the proposal's revision. The fingerprint stays internal, is never persisted/displayed/logged, and does not establish evidence authenticity.

Extend `PersonalContextSettingsSnapshot` additively with defaulted fields:

- `profile_identity: SettingsProfileIdentity | None = None`
- `provenance_subjects: tuple[SettingsProvenanceSubject, ...] = ()`
- `deleted_records: tuple[SettingsDeletedRecord, ...] = ()`

Preserve the existing non-deleted `records` and pending `proposals` contracts. Generate subjects in the owning service, not by guessing metadata in UI. Capture/check manifest identity around snapshot collection; return unavailable or raise the existing conflict error if it changes. Individual stale object tokens are rejected during detail inspection.

Add `PersonalContextService.settings_provenance(subject: SettingsProvenanceSubject) -> SettingsProvenanceResult`. It uses existing repository getters through the service, never SQL in the UI. READY and DISABLED allow user inspection; ABSENT, REMOVED and LOCKED do not. Authenticate status/manifest identity before and after the read, and recheck the current selected object token. Check proposal state and expiry with the injected clock; do not call the proposal helper that runs an expiry sweep. A missing, quarantined, resolved or mismatched object returns no projection. Existing integrity quarantine behavior of authenticated getters is unchanged.

Projection rules:

- Record fields: source, actor, reason, created/updated timestamps, state, record ID, current/parent version IDs and `derived_from_record_id` when retained. Absent parent/origin says **Not recorded**.
- Proposal fields: proposal ID, operation, state, created/expiry timestamps, target/base version IDs when retained, and the proposal envelope's provenance. Do not substitute the proposed record's provenance. Say **Proposal revision not retained as a canonical field**; do not label its fingerprint a stored version.
- Always show **Edit history not recorded** and **Inference classification not recorded**. Approval is a recorded event, not proof about the authorship/support of the current wording. No record confidence field is invented; proposal confidence is outside this display.
- No refs means **No source reference retained**, even when independent hashes remain. Retained refs use the exact legacy warning. Display hashes separately as unverified metadata; never pair unequal lists into fabricated reference/hash relationships.
- Tombstone projections use only the authenticated current tombstone. No history, payload, semantic key or Undo dependency.

## Read lifecycle

Create `tldw_chatbook/Widgets/Settings_Widgets/personal_context_provenance.py` with:

- `literal_metadata(value: str, *, limit: int = 160) -> str`
- `PersonalContextProvenanceDetails(subject: SettingsProvenanceSubject, service_loader: Callable[[], PersonalContextService], *, reload_details: Callable[[], None] | None = None, **kwargs: Any)`
- `PersonalContextProvenanceDetails.invalidate() -> None`

The section owns only inspection state. It receives the owning service through the panel; it does not reach into `ProfileProposalService._service`, app private fields, or encrypted tables.

1. Start reads with `call_after_refresh` after mount/expansion. Capture a local request generation, immutable subject and service instance. Invalidate on selection/reload, collapse, screen suspension, unmount and known profile lifecycle transitions before starting replacement work. Cancelled read threads may finish, but their results cannot publish.
2. Resolve the panel's service factory on each new read and Settings reload, off the UI thread, so a replacement app-owned service is observed. Do not overwrite its cached service from a stale worker. A different service instance invalidates the captured owner even if IDs were reused. Parent `load_records()` invalidates both its section and an open proposal section before refreshing; known mutations invalidate the detail at their existing entry point. Do not cancel an acceptance/irreversible commit worker.
3. To detect changes made by another owner while a detail stays open, recheck only the selected subject once per second while the section is expanded on the active screen. Never reload/decrypt the complete Settings inventory in that check. A nonblocking, per-service inspection lock prevents overlapping provenance reads; contention returns unavailable without queueing readers.
4. A successful detail has a two-second display lease measured from the start of its read with `time.monotonic()`. A UI deadline timer clears it if renewal stalls. Reject an already-expired result rather than starting its lease at callback delivery. Known local invalidations clear immediately; external changes are observed by this bounded refresh, not an unsupported claim of instantaneous cross-process notification.
5. Apply only to a mounted, expanded, active section with matching request generation/subject/owner and an unexpired lease. A changed or unavailable result clears fields immediately. A changed state is latched until the user reloads/reopens; do not silently adopt a new version. Pause timers and discard content offscreen. No permanent job or new persisted freshness field.

Render with literal `rich.text.Text`/`Static(markup=False)`. Replace Unicode `Cc`, `Cf`, `Zl`, `Zp` characters with spaces before clipping; preserve other Unicode. Cap each value at 160 characters including an ellipsis, display at most eight refs and eight hashes with explicit “showing N of M” labels. Do not add Rich link spans, Markdown links, or click handlers. Parent scroll containers own scrolling; new layout classes use existing design tokens.

## Task 1: Service-owned metadata and selection tokens

**Files:** Create `tldw_chatbook/Personal_Context/settings_provenance.py` and `Tests/Personal_Context/test_settings_provenance.py`; modify `tldw_chatbook/Personal_Context/service.py` and `Tests/Personal_Context/test_service.py`.

**Interfaces:** Produce the frozen types/helpers, additive snapshot fields and `settings_provenance()` above. No new repository, shared-core or agent API.

- [x] **Step 1: Add failing service tests using the existing real encrypted repository fixtures.** Pin these assertions, including successful authorized controls:

```python
def test_approval_and_subsequent_edit_do_not_invent_history():
    # Accept unchanged, accept edited, then edit in Settings in separate cases.
    assert detail.reference_status == "Legacy source reference — quotation not verified"
    assert displayed_history == "Edit history not recorded"
    assert displayed_inference == "Inference classification not recorded"
    assert "confidence" not in field_labels

def test_proposal_changes_without_base_version_change_are_rejected():
    assert before.base_version_id == after.base_version_id
    assert provenance_subject(identity, before) != provenance_subject(identity, after)
    assert service.settings_provenance(captured_subject).state == "changed"

def test_deleted_metadata_never_loads_retired_content():
    # Delete through the service, then forbid Undo/history/source access.
    assert snapshot.records == ()
    assert len(snapshot.deleted_records) == 1
    assert result.state == "available"
    assert deleted_payload_marker not in repr(result.projection.fields)
    assert deleted_payload_marker not in tuple(f.value for f in result.projection.fields)
```

Also parameterize manual/migration/promotion provenance; empty refs with and without hashes; unequal ref/hash lists; exact envelope-vs-proposed-record fields; missing/quarantined objects; expiry without mutation; READY/DISABLED versus unavailable statuses; profile/purge/version changes between reads; repr privacy; unchanged canonical bytes on clean inspection; and nonblocking concurrent readers. Do not treat a repr-only assertion as payload exclusion evidence.

- [x] **Step 2: Run `.venv/bin/python -m pytest Tests/Personal_Context/test_settings_provenance.py -q` and record the intended missing-interface/behavior failures.**
- [x] **Step 3: Implement the specified projection and owning service path, preserving snapshot compatibility and existing agent eligibility.**
- [x] **Step 4: Run the new file plus `Tests/Personal_Context/test_service.py` and `Tests/Personal_Context/test_proposal_service.py`; require passing targeted tests. Run Ruff on changed Python files and `git diff --check`.**
- [x] **Step 5: Commit only this unit's four owned files as `feat(memory): expose retained Settings provenance metadata`.**

## Task 2: Mounted provenance inspection and stale-read handling

**Files:** Create the provenance widget and `Tests/UI/test_personal_context_provenance.py`; modify `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py`, `tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py`, `tldw_chatbook/css/components/_settings_personal_context.tcss`, `Tests/UI/test_settings_personal_context.py`, and `Tests/UI/test_personal_context_proposal_review.py`. Rebuild `tldw_chatbook/css/tldw_cli_modular.tcss`.

**Interfaces:** Consume Unit 1 subjects/results. Proposal modal gains optional keyword-only `provenance_subject: SettingsProvenanceSubject | None = None` and `provenance_service_loader: Callable[[], PersonalContextService] | None = None`; the production panel always supplies both. Existing callers without them show unavailable provenance, never synthesize a trusted projection. Acceptance signatures and `ProposalReviewResult` stay unchanged.

- [x] **Step 1: Add failing literal-rendering and mounted lifecycle tests.** Use controlled worker barriers and an injected/patchable monotonic clock, not timing-dependent sleeps:

```python
def test_metadata_is_bounded_literal_unicode():
    assert literal_metadata("[link=https://example.test]x[/link]\x1b\u202ey") == "[link=https://example.test]x[/link]  y"
    assert literal_metadata("東京 café") == "東京 café"
    assert len(literal_metadata("x" * 1000)) == 160

async def test_old_worker_cannot_repopulate_invalidated_details():
    # Mount, release a valid read, start a delayed renewal, invalidate/change owner.
    assert visible_metadata == ()
    # Complete the old thread after the replacement read has completed.
    assert old_source_marker not in rendered_plain_text

async def test_stalled_read_does_not_keep_old_metadata_visible():
    # Expire the two-second lease while renewal is blocked.
    assert visible_metadata == ()
    assert active_inspection_reads <= 1
```

Parameterize invalidation for selection, record version, proposal contents with unchanged base version, profile ID/purge generation, locked/removal/replacement, collapse, screen suspension and unmount. Test results delayed past their lease, no offscreen polling, no full snapshot reload on renewal, and a yielding inherited mount handler. Confirm literal renderables have no style/link spans and no action opens a reference.

- [x] **Step 2: Run `.venv/bin/python -m pytest Tests/UI/test_personal_context_provenance.py -q`; require intended failures before implementation.**
- [x] **Step 3: Implement the section and lifecycle above, wire both production callers, and expose separate tombstone selection.** Keep proposal input state, acceptance worker groups, unknown-outcome handling and rejection behavior unchanged. Only read-inspection workers are disposable.
- [x] **Step 4: Add/complete production-host tests at `(100, 35)` and `(60, 24)`.** Expand both sections by keyboard; scroll through long values; reload a changed record; select deleted metadata and verify Edit/Archive/Delete cannot mutate it; confirm proposal Accept, Accept edited and Reject still reach the same owning service with the same payload/controls. No edit fields are replaced by a renewal. Assert actionable buttons and explanatory text are reachable, not merely present in a widget tree.
- [x] **Step 5: Rebuild CSS with `.venv/bin/python tldw_chatbook/css/build_css.py`; run the new UI file, the two modified UI test files, existing `Tests/UI/test_personal_context_review_modal.py` and `Tests/UI/test_design_token_governance.py`.** Require passing checks and changed-file Ruff/format/whitespace checks. Do not run the whole UI suite.
- [x] **Step 6: Commit only this unit's owned files and generated bundle as `feat(memory): inspect provenance in My Profile and proposal review`.**

## Task 3: Verify the integrated path and close the tracker

**Files:** Update this plan, TASK-25907.2 and `backlog/docs/personal-context-memory-roadmap.md`; create `Docs/superpowers/reviews/2026-09-25-personal-context-provenance-execution-review.md`. Any synthetic screenshots belong under `Docs/superpowers/reviews/evidence/personal-context-memory/`.

**Interfaces:** Consume the actual production Settings panel/modal and measured test receipts. This task adds no product interface.

- [x] **Step 1: Run the complete targeted selection from Units 1–2 if integration changes followed their last passing run.** Add `Tests/Agents/test_profile_tool_provider.py` and `Tests/Chat/test_console_personal_context_snapshot.py` as regression checks for the shared service change. Do not repeat a passing selection without intervening changes or unresolved concerns.
- [x] **Step 2: Inspect the mounted production Settings surface and proposal modal with real CSS, synthetic records and provider/keyring/network isolation.** Capture wide/narrow renders and interaction results. Verify private record provenance is user-visible while the existing agent regression control still denies that record. If a full-app manual launch cannot be isolated safely, record that limit and use the production Settings host harness; do not claim a full-app live check.
- [x] **Step 3: Run changed-file Ruff check and format check, generated CSS consistency/governance, `git diff --check`, and scoped Backlog/link/AC validation.** Record exact commands/counts and any pre-existing failures separately; no “all tests pass” claim from an incomplete shell chain.
- [x] **Step 4: Self-review the branch and obtain the native execution workflow's single fresh final review.** Fix blocking findings, rerun only affected checks, and preserve review decisions/evidence in the execution review. No per-unit agent delegation.
- [x] **Step 5: Check each of the seven Backlog criteria against evidence, add implementation notes and ADR links through the CLI, and mark Done only if the full task definition is met.** Update the roadmap with the actual outcome, remaining limitations and next task; commit only owned documentation. No PR, push or merge is part of this plan.

## Self-review and approval checkpoint

All seven criteria map to Units 1–2 and integrated evidence in Unit 3. The review found and addressed four plan hazards: proposals lack a revision field; existing Settings omits tombstones; cached service factories and late workers can retain a previous owner; and identical approval metadata cannot establish editing/history or confidence. The freshness lease is an explicit proposed implementation choice for bounded external-change detection, not an existing capability or a permanent memory-maintenance job.

No API signatures refer to a future Backlog task. No change to Next Send diagnostics, retrieval ranking, forgetting or provider disclosure is included. The baseline's known policy failures remain open in their existing tasks.

At the planning checkpoint, validation passed for 20 local Markdown links across three changed documents, both ADR paths, the scoped guard for all ten task IDs, and preservation of 59 child criteria. TASK-25907.2 then had seven open criteria; no product tests were claimed at that documentation-only checkpoint. Implementation and verification are now complete, with evidence and limitations in the linked execution review.

The user approved this execution plan on 2026-09-25 and requested native continuation. The plan-review checkpoint is complete.

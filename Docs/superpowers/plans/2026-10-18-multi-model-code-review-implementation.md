# Multi-Model Code Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Console primary coordinate independent reviews through saved provider/model presets, present their original reports, and optionally obtain a fresh, evidence-based verification and unified assessment.

**Architecture:** Keep orchestration in the primary agent and execution in the existing fleet. Add a bounded, session-owned review service for source snapshots, role admission, structured reports, blinded verification and storage accounting. Extend the existing automatic-work ledger with typed review-boundary eligibility; retain its acceptance, causal-chain and recovery rules.

**Tech Stack:** Python ≥3.12, existing Pydantic 2, stdlib filesystem/JSON/hash/locking utilities, Textual 8.x, existing SQLite AgentRuns owner, existing tool providers and model dispatch. No new dependency.

**Spec:** [Primary-agent multi-model code review design](../specs/2026-10-18-multi-model-code-review-design.md)

**Task:** [TASK-35582](../../../backlog/tasks/task-35582%20-%20Implement-primary-agent-multi-model-code-review-recipe.md)

**Status:** Implementation authorized on 2026-10-18 and in progress in the isolated TASK-35582 worktree. Follow the 11 sequential slices and their review gates; no completed behavior or acceptance is implied by this status. TASK-35582 is In Progress and criteria remain unchecked until supported by evidence.

```text
ADR required: yes
ADR path: backlog/decisions/281-primary-agent-multi-model-code-review.md
Reason: Existing ADR-281 governs the primary recipe, provider/source/report ownership,
role capabilities, cross-run verification and the ADR-134/135 continuation amendment.
```

## Global constraints

- Read the spec and [ADR-281](../../../backlog/decisions/281-primary-agent-multi-model-code-review.md) together. The plan implements their revised contracts; it does not replace them.
- Preserve Console's default **two total children per primary turn**. Concurrency is a separate limit. Children cannot delegate further.
- Preserve existing routing, destination consent, permissions, exclusions, sensitive-path checks, project-instruction handling, stop/close fences and actual worker settlement.
- The default team starts empty. Store ordered stable preset IDs and optional focuses in `[agents.code_review]`; never select paid destinations automatically.
- Run-only provider/model overrides are admitted during preparation through ADR-147, frozen into slots, and never written into saved teams or presets. Scoped spawning accepts only those slot references.
- `[tools] code_review_enabled = true` gates the entire feature. Guide lookup starts no capture, model call or child.
- One shared general brief, with optional additional focus. No peer reports or primary-agent suspected findings in reviewer tasks.
- Present individual reports before verification. Verification needs an explicit user stage grant and one fresh configured verifier; model arguments cannot create grants.
- Use structured schema-version-1 originals, not a prose parser. Records ≤8 KiB; submission chunks ≤8 records. Unknown fields/versions refuse.
- Guide result, including JSON envelope, ≤12 KiB UTF-8. Non-model projections omit guide bodies, focus text and source/report bodies.
- Verification batches ≤8 units and ≤4,096 estimated claim tokens, further ≤half remaining admissible input. Evidence pages ≤1,024 estimated tokens, further bounded by actual remaining space. Preserve the final full-request admission check.
- Every complete model-bound review response also fits the live run's `max_tool_result_chars` and the existing 16,000-character native-continuation ceiling. Apply that ceiling conservatively to both protocols, including JSON metadata/cursors/revisions; paginate before the runtime's generic truncation seam.
- A verifier child uses explicit checkpoint-backed request-history reduction, independent of primary-only run-log eviction. Keep its task, admitted guidance and live protocol state; remove only recoverable completed rounds. No additional child/model budget or history-reading capability is granted.
- Reviewer/verifier submissions use an independent revision per bound slot attempt. Context-control revisions and raw-capture publication cannot invalidate another participant's append; authority and terminal-state checks still apply to every write.
- Threaded executable-check proposals yield the primary's existing fleet wait; they never execute themselves. Task 9 adds a primary-only claim/decline stream and bounded host-result wait for the same live verifier, without another child, automatic turn or budget reset. Inline verifiers retain source-based verification; check proposals resolve immediately as unavailable, without blocking their synchronous parent.
- Review source reads participate in existing nested project-instruction activation through provider-validated original binding paths. Opaque refs/cursors cannot select arbitrary guidance or reveal physical capture paths, and instructions grant no tool authority.
- Fixed storage caps: source/check content 64 MiB; reports/raw/manifests/index/ledger 16 MiB; participant raw plus submitted output 1 MiB; total per review 80 MiB; total regular files per review 2,048; retained plus reserved per tab 160 MiB; retained contexts per tab four. Count staging, generated files and physical copies. Existing narrower limits still apply.
- Keep source/report state session-local under scratch ownership. No new review database or durable review-job recovery. Any existing DB change uses its normal versioned migration.
- No new workflow engine, provider adapters, named/per-workspace teams, automatic fixes, comments/publication or merge actions.
- Use targeted tests only. A full-suite sweep requires user opt-in. Live provider qualification is a bounded, separately recorded execution activity, not something performed while writing this plan.
- UI changes follow [design language](../../../backlog/docs/design-language.md) and [component patterns](../../../backlog/docs/component-patterns.md). Edit source CSS and rebuild the bundle; do not edit generated CSS directly.

## Baseline and reuse map

Planning inspected checkout `b07fc0425288fb51d17cde5f745051720a7c9db6`. Re-read the named functions at execution time rather than relying on line numbers after intervening changes.

| Existing owner | Reuse or required change |
| --- | --- |
| `Agents/agent_models.py::RunBudget`, `AgentConfig`, `AgentDefinition` | Keep `max_subagents=2`; add a narrowly typed optional review binding, without changing ordinary runs. Catalog allowlists alone do not restrict runtime tools. |
| `Agents/agent_routing.py::resolve_spawn_target`, `freeze_spawn_target` | Resolve explicit preset destinations, sampling and fallbacks; capture snapshots; recheck current authority before dispatch. |
| `Agents/agent_service.py` spawn closure, `wait_agents` and provider-call boundary | Bind slots to actual runs, enforce child scopes, capture raw output before bounded summaries, attribute actual destinations and reduce verifier request history from its checkpoint. Yield review-bound waits on pending check requests without collecting the live verifier. Existing run-log eviction is gated to primaries; it cannot supply this child behavior. |
| `Agents/agent_runtime.py::LoopDeps`, spawn dispatch | Accept the mutually exclusive scoped spawn form; close runtime gates and reject guessed forbidden calls. |
| `Agents/tool_catalog.py::build_spawn_schema`, provider registry | Add review schemas/discovery and role-aware projections. |
| `Agents/project_instruction_runtime.py::InstructionActivationLedger.prepare`, `tool_catalog.py::PathAwareToolProvider` | Implement validated source-reference/query scope mapping in the review provider. Reuse the existing ledger's per-chain delivery, deferral and consent; do not create another instruction loader. |
| `Chat/console_agent_bridge.py` per-run composition | Compose the review provider into live and preview inputs; pass trusted session/actor/authority, not model-supplied identity. |
| `Chat/console_scratch_space.py::ConsoleScratchSpaceManager` | Retain generation-aware scratch leases; tombstone before draining writers. |
| `Tools/git_tool_impls.py::prepare_repository`, `run_git` | Reuse repository confinement and bounded process execution. Existing string-returning Git tools do not expose an immutable blob reader; add narrow internal byte-safe helpers, without exposing arbitrary Git arguments. |
| `Tools/patch_tool_impls.py::parse_patch_targets`, `apply_patch_to_text` | Reuse bounded unified-diff parsing and in-memory application. Do not run `git apply` against the checkout. |
| `DB/automatic_work.py::AutomaticWorkLedger`, `DB/AgentRuns_DB.py` | Extend the existing body-free wake claims; use real SQLite transactions and existing acceptance states. Current schema version is 24. |
| `Chat/console_fleet_wake.py::AgentWakeCoordinator` | Add review-boundary eligibility beside completion/progress eligibility; keep queue priority, runtime capacity, acceptance and uncertain-attempt handling. |
| `Chat/console_chat_controller.py`, `Chat/console_interrupt_rounds.py` | Accepted user provenance, approval/acceptance callbacks, session close/stop and source-epoch fences. Keep substantial review logic out of the large controller. |
| `Widgets/settings_agents_panel.py::AgentsSettingsPanel` | Ordered preset/focus/verifier controls beside preset editing; preserve off-loop DB reads. |
| `Packaging/check_manifest.py`, `pyproject.toml`, `MANIFEST.in` | Explicitly ship runtime guides, with installed-artifact tests. |

Read [testing evidence lessons](../../../backlog/docs/lessons-testing-evidence.md), [live verification lessons](../../../backlog/docs/lessons-live-verification.md), and [backlog hygiene](../../../backlog/docs/lessons-backlog-hygiene.md) before execution. In particular, prove a named model at the actual provider boundary; configuration objects alone have previously concealed parent-model dispatch.

## File boundaries

These are proposed files, not files created by this planning change. Use the existing flat `Agents/` convention; do not introduce a generic artifact or workflow framework.

| Create | Responsibility |
| --- | --- |
| `tldw_chatbook/Agents/code_review_models.py` | Strict wire models, enums, constants, immutable scope/receipt values and pure launch arithmetic. |
| `tldw_chatbook/Agents/code_review_store.py` | Session bundle storage, reservations, revision/idempotency publication, leases and cleanup accounting. |
| `tldw_chatbook/Agents/code_review_source.py` | Admitted target resolution, immutable capture and same-view source reads. |
| `tldw_chatbook/Agents/code_review_service.py` | Prepare/read/submit/release, slot lifecycle, terminal sealing and host projections. |
| `tldw_chatbook/Agents/code_review_verification.py` | Substantive projection, opaque IDs, batching, disposition coverage and deterministic expansion. |
| `tldw_chatbook/Agents/code_review_history.py` | Verifier-only checkpoint-backed request projection using existing protocol-round and continuation ownership utilities. |
| `tldw_chatbook/Agents/code_review_tool_provider.py` | Tool schemas, guide loading, role-aware dispatch and safe metadata projections. |
| `tldw_chatbook/Agents/code_review_guides/review.md`, `verify.md` | Versioned on-demand recipes. |
| `tldw_chatbook/Chat/console_code_review.py` | Native session composition, trusted intent/grants, boundary receipts and presentation/check coordination. |
| `tldw_chatbook/Widgets/Console/console_code_review_modal.py` | Bounded report/index reader, explicit export/release choices and expired-state feedback. |
| `Docs/Examples/skills/multi-model-review/SKILL.md` | Optional trusted inline entry point. |
| `Tests/Agents/code_review_helpers.py` | Small fixture builders and integration harness described below; no production abstraction. |

Existing files change only at their owned seams in the table above. In task file lists, `Agents/`, `Chat/`, `Tools/`, `Widgets/`, `DB/`, `UI/`, `Utils/`, `css/` and `config.py` are relative to `tldw_chatbook/`; `Tests/`, `Docs/` and `Packaging/` are relative to the repository root. Config defaults and the generated template are both owned by `tldw_chatbook/config.py::CONFIG_TOML_CONTENT`; there is no `Config_Files/config.txt` template to change. Settings styles are in `tldw_chatbook/css/features/_settings.tcss`; Console styles are in `tldw_chatbook/css/features/_console.tcss` and `_console_panels.tcss`. Reuse them or add a lazily loaded `tldw_chatbook/css/screen_modal_console_code_review.tcss` for the new modal, following existing modal ownership.

## Shared contracts and test vocabulary

Task 1 defines the wire types; Tasks 2–6 define their owners. All production methods derive caller scope from a trusted bound actor. A wire `review_id`, source reference, cursor, slot ID or receipt ID is only a lookup key. It is never proof of authority.

Use `ReviewRefused(reason: str)` for stable feature refusals, preserving the underlying permission/routing owner's reason. Use strict Pydantic models with `extra="forbid"`; do not deserialize untrusted objects with pickle. Collections below are immutable tuples internally and JSON arrays on the wire.

| Type | Required fields / meaning |
| --- | --- |
| `ReviewTarget` | Discriminated `kind`: `pr`, `commit_range`, `patch`, `working_tree`; binding reference, relative scope and explicit `context_paths` to capture before launch. Kind-specific exact revisions/comparison, patch reference/preimage, staged/unstaged/untracked selection. Reject unused extra fields. |
| `ReviewerChoice`, `ReviewTeam` | Saved `preset_id: str`, `focus: str = ""`; ordered reviewers and optional `verifier_preset_id`. IDs refer to existing DB rows, not display names. Saved choices contain no run-only destination overrides. |
| `ReviewDestinationOverride`, `ReviewRunChoice` | Run-only override has optional nonempty `provider` and/or `model`, at least one supplied. A run choice has `preset_id`, `focus` and optional `destination`. Reviewer focus supplements the shared brief; verifier choices require empty focus. Saved defaults are converted to run choices without destination overrides. |
| `ReviewRole` | `primary`, `reviewer`, `verifier`; host-only role association. |
| `ReviewRunBinding` | Immutable review ID, slot ID, attempt ID, role, target digest and source epoch; constructed by the host and attached to an actual run. A verifier also has host-only `check_feedback_mode` primary_handoff/unavailable_inline, derived from actual launch mode rather than tool JSON or a preparation-time guess; other roles leave it unset. |
| `SourceLocation` | `source_ref`, `start_line >= 1`, `end_line >= start_line`; source manifest establishes valid line range. |
| `SourceRange` | Strict 1-based integer `start_line`, `end_line >= start_line`, rejecting booleans. Optional source-read selection, validated against the captured document's line count; no path or revision field. |
| `EvidenceRef` | Discriminated captured-source location/digest or host check receipt; no arbitrary host path or participant-owned read receipt. |
| `FindingRecord` | `schema_version=1`, `local_key`, `title`, nonempty `locations`, `trigger`, `impact`, `evidence`, `severity` in critical/high/medium/low, `uncertainty`. |
| `CoverageRecord` | `schema_version=1`, inspected refs, gaps with stable reason and optional safe ref, explicit `empty_findings: bool`; empty findings is valid only with successful sealed completion. |
| `DispositionRecord` | `schema_version=1`, `verification_unit_id`, verdict confirmed/refuted/unresolved, reason, evidence, optional duplicate unit; nonempty `next_check` required for unresolved, optional for confirmed/refuted. |
| `CheckProposal`, `CheckReceipt` | Proposed argv/check intent and referenced opaque units; host assigns a proposal ID bound to the live verifier attempt/target. At most one outstanding proposal per verifier. A result receipt identifies executed/declined/unavailable/cancelled/expired/uncertain, bounded evidence and, only for actual execution, the native execution receipt. A proposal alone never proves execution. |
| `CheckActionRecord`, `ReviewCheckClaim` | Strict v1 primary action with `proposal_id`, `action` claim/decline and optional reason. A host-only claim binds the live primary/verifier attempt, target, admitted disposable executor/specification and acceptance identity. Claiming is separate from execution and grants no permission. |
| `ReviewContext` | Full spec record: owner/session/conversation/epoch, context-control `revision`/state, publication sequence, source/guide/preset/override snapshots, grants, slots/actual receipts, per-attempt submission revisions, primary check-action streams, proposal/claim/result state, quotas, original index, private unit map, batch cursor and dispositions. Credentials excluded. |
| `PreparedReview`, `ReviewPage`, `SubmissionAck`, `ReleaseReceipt` | Prepared review ID/context-control revision/slots/small summary; page `text`, role-filtered `records`, next `cursor`, context-control revision and the caller's own `submission_revision` when applicable; ack accepted IDs plus `submission_revision`/cursor; tombstone/cleanup status respectively. Pack the complete canonical response, with final actor-specific metadata, before advancing its delivered cursor or committing a mutation that needs an acknowledgment. |
| `ReviewBoundaryReceipt` | Private immutable review ID/context-control revision, exact next slot/slot-attempt/stage, predecessor settlement IDs, stage grant ID, epoch, original chain and runtime owner. Only a live host registry can validate it. |
| `SourceQuery` | Strict discriminated union of `ListSourceQuery(op="list", relative_dir, side)`, `SearchSourceQuery(op="search", text, relative_dir, side)` and `ResolveSourceQuery(op="resolve", relative_path, side)`. `side` is one of the captured target's base/head/staged/worktree views, never an arbitrary revision. Relative paths remain subject to the admitted binding/exclusions. |
| `ReviewHistoryCheckpoint` | Host-only verifier attempt ID, accepted submission revision, current batch/cursor, pending opaque unit IDs and bounded evidence/check refs. No original IDs, reviewer attribution, report counts, private map or raw report prose. |
| `ReviewLimits` | Frozen internal storage-cap value object using the exact production constants below; unit fixtures may inject smaller limits. It is not user configuration or a model/tool argument. |

Implement these public Python interfaces; tool adapters translate JSON into them. Code samples use imports from their task's declared modules plus stdlib `json`, pytest and the existing `RunBudget`; include those imports in the actual test files. Task steps include focused implementation excerpts/algorithms rather than a second complete copy of the production code:

```python
def minimum_continuations(reviewers: int, verification: bool, remaining: int) -> int:
    additional_children = max(reviewers + int(verification) - remaining, 0)
    return (additional_children + 1) // 2

def canonical_record_bytes(record: dict) -> bytes:
    return json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
```

Validate nonnegative integer counts, rejecting booleans, before launch arithmetic. Serialization for record limits includes every wire field and uses actual UTF-8 bytes, not Python character count.

Task 1 also defines `review_result_char_limit(max_tool_result_chars: int) -> int`, returning `min(max_tool_result_chars, 16000)` for positive runtime limits and `16000` otherwise. Zero/negative means no ordinary runtime cap, but this feature still honors the existing native ceiling in either protocol. This internal value is derived from trusted run configuration, never a model argument or new setting. Task 3 packs source pages, Task 4 packs public control/query responses, and Tasks 8/9 reuse that same contract for claims/check evidence. `ToolResult.content` uses the canonical complete wire response exactly once; adapters cannot add another body wrapper after fitting it. The outer native/fenced protocol wrapper remains part of final model-input estimation.

These checks govern model-bound tool delivery; native report/export readers retain their own bounded UI/file authority and may show a stored original too large for a model response. A cursor replay returns its pinned page only while current authority/capacity still admit it; otherwise refuse explicitly rather than resizing that cursor's page or silently skipping content.

For readable integration tests, `Tests/Agents/code_review_helpers.py` provides `review_harness` using a real temporary repository, `ConsoleScratchSpaceManager`, in-memory `AgentRunsDB`, real review store/service and a real `AgentService` plus `FleetCoordinator`. Add Console-controller/wake wiring when Task 7 requires it. Replace only the external model gateway, clocks and explicit fault points. Never replace the scope, quota, dispatch or acceptance method being asserted.

The harness exposes these test-only operations, with all arguments keyword-only where shown:

```text
prepare(reviewers=2, verify=False, remaining=2, target_kind="working_tree") -> str
invoke(*, role, tool, arguments, review_id, slot=0) -> ToolResult
run_script(*, reviewers, verify, prior_spawns=0) -> ReviewRunObservation
submit_finding(review_id, *, slot=0, local_key="one", trigger="empty input") -> SubmissionAck
finish_reviewer(review_id, *, slot=0, status="done", coverage=True) -> None
finish_verifier(review_id, *, status="done") -> None
read(review_id, *, role="primary", view="summary", cursor=None) -> dict
close() -> None
```

`ReviewRunObservation` contains actual gateway requests, `actual_child_launches`, `launches_per_primary_turn`, presented report events, verifier-start events, `original_reports_presented_before_verifier`, chain IDs and accepted-attempt IDs. A gateway observation records role/provider/model/params and whether the configured parent model was incorrectly used. Helpers build wire records from known captured refs and invoke the real provider under `use_run_id`; they do not write accepted findings directly into the ledger. Later tasks define additional helpers/observations at the slice that introduces their owner; tests needing faults use pytest monkeypatch on the named owner rather than inventing bypass paths.

Build helpers incrementally with each owning slice: pure payload builders in Task 1; store/source/service fixtures in Tasks 2–4; real threaded AgentService dispatch in Task 5; full native Console submission/acceptance in the combined authority/ledger slice. Service unit fixtures may issue a trusted host scope/grant directly through the internal authority issuer; only the native authority/journey tests count as evidence of actual user-intent admission. Do not make an early test depend on not-yet-built Console wiring or mock the authority check it claims to verify.

## Task 1: Strict records, defaults and admission arithmetic

**Files:** Create `Agents/code_review_models.py`, `Tests/Agents/test_code_review_models.py`, `Tests/Agents/code_review_helpers.py`, `Tests/test_config_code_review_defaults.py`. Modify `config.py::CONFIG_TOML_CONTENT` and the existing settings accessor/merge path. Reuse `Tests/test_config_console_defaults.py` patterns; do not modify the user's root `config.toml`.

**Interfaces:** Produce the shared wire types, `ReviewRefused`, constants, `canonical_record_bytes(record: dict) -> bytes`, `review_result_char_limit(max_tool_result_chars: int) -> int`, `validate_record_chunk(kind: str, records: list[dict]) -> tuple[FindingRecord | CoverageRecord | DispositionRecord | CheckProposal | CheckActionRecord, ...]`, `minimum_continuations(reviewers: int, verification: bool, remaining: int) -> int`, and `parse_review_team(value: object) -> ReviewTeam`. Define separate saved and run choice schemas now, including `ReviewDestinationOverride`, plus `SourceRange` and the closed `SourceQuery` union. The closed submission `kind` selects its strict record model; the service subsequently checks that kind against the caller role. Coverage/completion requests and verifier completion are explicit submission kinds, not a model-supplied successful-run status. Define check-action records now; enable their stateful execution handoff only in Task 9.

- [ ] **1. Write failing boundary tests.** Include this complete record fixture, then parameterize unknown version/field, invalid severity, zero/reversed/non-integer source ranges, duplicate local keys, UTF-8 size and nine-record chunks. Reject destination overrides in saved choices, empty override objects, unknown source operations/sides, invalid check actions and extra fields; accept provider-only, model-only and paired run overrides without altering the saved record:

```python
def finding_payload():
    return {
        "schema_version": 1,
        "local_key": "empty-input",
        "title": "Empty input raises before validation",
        "locations": [{"source_ref": "src-1", "start_line": 4, "end_line": 4}],
        "trigger": "Call parse with an empty string",
        "impact": "IndexError replaces the documented validation result",
        "evidence": [{"kind": "source", "source_ref": "src-1",
                      "start_line": 4, "end_line": 4, "digest": "a" * 64}],
        "severity": "medium",
        "uncertainty": "No execution performed",
    }

def test_default_child_arithmetic_does_not_change_run_budget():
    assert RunBudget().max_subagents == 2
    assert minimum_continuations(2, False, 2) == 0
    assert minimum_continuations(2, True, 2) == 1
    assert minimum_continuations(3, False, 2) == 1
    assert minimum_continuations(3, True, 2) == 1
    assert minimum_continuations(2, True, 1) == 1

@pytest.mark.parametrize("configured,expected", [
    (1024, 1024), (16000, 16000), (20000, 16000), (0, 16000), (-1, 16000),
])
def test_review_result_limit_retains_native_ceiling(configured, expected):
    assert review_result_char_limit(configured) == expected
```

- [ ] **2. Run RED:** `.venv/bin/python -m pytest Tests/Agents/test_code_review_models.py -q`. Expect collection/import failure for the new module, then named validation failures while implementing.
- [ ] **3. Implement strict schemas and constants.** Pydantic validators reject malformed counts, ranges and kind-specific target fields. `validate_record_chunk` checks serialized per-record size before parsing and rejects the whole invalid chunk without mutation. Use `parse_review_team` in both settings and preparation; permit empty saved defaults, but reject launching fewer than two reviewers. Default verifier is unset; missing verifier blocks only stage two. Preserve unrelated config keys and existing env/config precedence. Bound user-authored focus by the existing input validator and the final guide/task envelopes; never truncate a focus silently.

The TOML template uses the following empty defaults; convert the empty verifier ID to the internal unset value without trying to write TOML nulls. Place the nested table after the existing parent-table keys so following agent defaults are not accidentally nested into it. Add `code_review_enabled = true` to the existing `[tools]` table.

```toml
[agents.code_review]
reviewers = []
verifier_preset_id = ""
```

```python
MIB = 1024 * 1024
RECORD_MAX_BYTES = 8 * 1024
SUBMISSION_MAX_RECORDS = 8
GUIDE_MAX_BYTES = 12 * 1024
SOURCE_MAX_BYTES = 64 * MIB
REPORTS_MAX_BYTES = 16 * MIB
PARTICIPANT_MAX_BYTES = MIB
REVIEW_MAX_BYTES = 80 * MIB
REVIEW_MAX_FILES = 2048
TAB_MAX_BYTES = 160 * MIB
TAB_MAX_CONTEXTS = 4
VERIFICATION_MAX_UNITS = 8
VERIFICATION_MAX_CLAIM_TOKENS = 4096
EVIDENCE_PAGE_MAX_TOKENS = 1024
```

- [ ] **4. Run GREEN:** `.venv/bin/python -m pytest Tests/Agents/test_code_review_models.py Tests/test_config_code_review_defaults.py Tests/Agents/test_agent_routing.py -q`. Assert saved names can change while stable ID references survive; invalid/missing IDs remain visible instead of being replaced.
- [ ] **5. Review and commit only this slice.** Suggested message: `feat(agents): define bounded code review records and defaults`. No UI or model execution is needed to validate this slice.

## Task 2: Scratch bundle, atomic publication and quotas

**Files:** Create `Agents/code_review_store.py`, `Tests/Agents/test_code_review_store.py`. Modify `Chat/console_scratch_space.py` only where public generation-aware lease or cleanup-lifecycle hooks are necessary; preserve its existing close semantics.

**Interfaces:** Produce `ReviewStore`, constructed with the owning scratch manager/session, with host-only `create(*, target_digest: str, source_epoch: int = 0, conversation_id: str | None = None) -> ReviewContext`, `reserve(review_id: str, *, source_bytes: int, report_bytes: int, files: int, participant_id: str | None = None) -> ReviewReservation`, `bind_attempt(run_id: str, binding: ReviewRunBinding) -> None`, `publish(review_id: str, *, expected_revision: int, submission_id: str, records: list[dict], kind: str = "findings") -> SubmissionAck`, `lease(review_id: str) -> ContextManager[ReviewContext]`, `release(review_id: str, *, expected_revision: int) -> ReleaseReceipt`, `usage() -> ReviewUsage`. `bind_attempt` is host-only and rejects reassociation of a run to different identity; `publish` resolves it using existing `current_run_id()`, never wire fields. Its stream `(slot_id, attempt_id)` owns the submission revision. `release` checks context-control revision. `ReviewReservation` is a context manager with `commit(actual_source_bytes: int, actual_report_bytes: int, actual_files: int) -> None`; unused reservation is released on exit after native work settles. `ReviewUsage` exposes retained/reserved bytes, regular-file and context counts without content.

- [ ] **1. Write RED tests around real temporary files.** At each cap, test exactly the limit and one byte/file/context above it. Use a small injected `ReviewLimits` value in unit tests, while asserting production constants equal the global table. The `store` fixture maps test-local alias `review-1` to a host-minted context ID (retrieved by test helper `rid(store)`) with valid captured `src-1` and a default R1 attempt association. A `bound_attempt(slot_id: str, attempt_id: str) -> ContextManager[None]` fixture registers its host binding through `bind_attempt` and enters existing `run_context.use_run_id` on the invoking thread; it adds no testing API to production `ReviewStore`. These store unit fixtures do not depend on the service's stage-grant issuer introduced in Task 4 or prove native consent. Race two reservations with `threading.Barrier` and assert only one fits. Inject `os.replace`/write/delete errors; retain accepted data and charge orphaned bytes.

```python
def test_same_submission_replays_before_stale_revision_check(store):
    first = store.publish(rid(store), expected_revision=0,
                          submission_id="s1", records=[finding_payload()])
    replay = store.publish(rid(store), expected_revision=0,
                           submission_id="s1", records=[finding_payload()])
    assert replay == first
    changed = finding_payload() | {"impact": "Different claim"}
    with pytest.raises(ReviewRefused, match="submission_conflict"):
        store.publish(rid(store), expected_revision=0,
                      submission_id="s1", records=[changed])
```

- [ ] **2. Run RED:** `.venv/bin/python -m pytest Tests/Agents/test_code_review_store.py -q`.
- [ ] **3. Implement the session-owned store.** Use a single per-tab lock for quota reservation plus per-context serialized publication; establish one lock order, tab before context. Store files under opaque host-generated names with private permissions. Do not expose physical bundle paths to role tools. Count actual staging and destination bytes while both exist; cap growth before accepting writes.

```text
publish:
  bind owning session/epoch and acquire live scratch lease
  validate stored run/role/attempt, owning session/epoch and live context status
  select that attempt's submission stream; never accept a model-selected stream
  compute canonical payload digest; under lock check this stream's submission ID
  identical accepted ID -> return original acknowledgement
  conflicting ID -> submission_conflict; otherwise compare stream expected_revision
  validate records/references and reserve staging + final publication headroom
  write a bounded immutable chunk; atomically publish the manifest pointer
  advance this stream's submission_revision and publication sequence with its indexes
  acknowledge only after publication; leave other streams' revisions unchanged
  on failure keep previous manifest authoritative and account for remaining files
```

Keep the journal/index append format bounded by the report quota. Avoid rewriting an entire growing report on every finding; append immutable chunks and publish a small manifest. Failed/unpublished chunks remain charged until actually deleted. Do not claim crash recovery for this session ledger.

Context-control revision changes only for authority/lifecycle operations such as a newly admitted stage, explicit retry or release; opaque paging cursors bind a separate publication snapshot. Finding appends, lazy same-revision source capture and host raw-output writes do not advance another role's submission revision. Host raw capture has its own sequence and cannot occupy the model's stream. A stale same-stream write returns `submission_revision_conflict` and only that caller's current revision. The recipe permits one unchanged-payload retry with the same submission ID and refreshed revision; another conflict stays explicit/incomplete. It never resubmits changed claim bytes or retries a child launch.

The service in Task 4 supplies the additional authoritative stage/routing/source checks before invoking publication. The store independently refuses stale session/attempt associations and tombstones; its unit tests do not need the later native coordinator.

- [ ] **3a. Prove independent publication under concurrency.** Launch R1 and R2 on real threads behind one barrier, both using expected revision 0 and the same local key/submission ID. Both must receive submission revision 1 and distinct stable original IDs:

```python
def test_participants_append_without_invalidating_each_other(store, bound_attempt):
    barrier = threading.Barrier(2)

    def submit(slot_id):
        with bound_attempt(slot_id, f"attempt-{slot_id}"):
            barrier.wait(timeout=5)
            return store.publish(
                rid(store), expected_revision=0, submission_id="s1",
                records=[finding_payload()],
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit, slot) for slot in ("R1", "R2")]
        acknowledgments = [future.result(timeout=10) for future in futures]
    assert [ack.submission_revision for ack in acknowledgments] == [1, 1]
    assert len({ack.accepted_ids[0] for ack in acknowledgments}) == 2
```

Import stdlib `threading` and `concurrent.futures.ThreadPoolExecutor` in this test file. Reject a stale second write from R1 without changing R2. Within one stream replay identical IDs idempotently and reject changed bytes. Repeat through actual child tool calls in Task 5; do not count this unit fixture as gateway evidence. Raw-output interleaving belongs to Task 6, after its capture owner exists; this slice needs no capture stub or later implementation.

- [ ] **4. Implement release/close races.** Explicit release refuses active execution/read leases. Close first tombstones all contexts, revokes future callbacks, then allows existing native leases to drain. Cleanup releases byte/file/context charges only after confirmed deletion. A failed cleanup remains charged and visible. A retained context is never evicted to make room for another.

Host creation mints each review ID from a per-generation monotonic issuance component plus fresh randomness; callers cannot recreate a released identity. Keep at most `TAB_MAX_CONTEXTS` recent release receipts; older replay may refuse as expired/unknown. Discard released source/report/attempt/run payload metadata, retaining only compact ownership accounting while actual deletion is uncertain. Retire terminal generations from registry ownership after native leases settle. The scratch cleanup receipt remains a readable terminal observation and may deliver one-shot host lifecycle callbacks outside manager, receipt and invoking consumer locks; isolate callback failures, deliver bookkeeping before retirement/notification, and keep failed physical cleanup pending for retry. Missing/replaced paths alone never prove deletion, including moved bundles/files beneath an unchanged scratch root. A narrow retryable native pre-cleanup callback may reuse identity-checked owned-file unlink after leases drain and before scratch root removal. Root deletion alone never refunds unproven residual file charges. Externally moved/unlocated contents retain conservative stale-handle charges, including when later root cleanup may have removed them; no full-root inode search is required.

- [ ] **5. Run GREEN:** new store tests and `.venv/bin/python -m pytest Tests/Chat/test_console_scratch_space.py -q`. Include staged copies, concurrent output writers, cancelled-but-live workers, late callback, repeated release and close-during-publication controls.
- [ ] **6. Review and commit:** `feat(agents): retain review bundles with atomic quota accounting`.

## Task 3: Immutable admitted targets and same-view source reads

**Files:** Create `Agents/code_review_source.py`, `Tests/Agents/test_code_review_source.py`. Modify `Tools/git_tool_impls.py` for narrow internal revision/tree/blob helpers and byte-safe bounded output. Reuse `Tools/patch_tool_impls.py` without changing its apply semantics. Extend `Tests/Tools/test_git_tool_impls.py` and `test_git_tool_sensitive_paths.py`.

**Interfaces:** Produce `ReviewSourceAuthority` from the existing admitted binding/exclusions/instruction context, never from raw model paths; `capture_review_source(target: ReviewTarget, authority: ReviewSourceAuthority, store: ReviewStore, review_id: str) -> SourceManifest`; `read_review_source(manifest: SourceManifest, source_ref: str, *, start_line: int, max_tokens: int, end_line: int | None = None, cursor: str | None = None, max_result_chars: int = 16000) -> ReviewPage`; `query_review_source(review_id: str, query: SourceQuery, *, cursor: str | None = None, max_result_chars: int = 16000) -> ReviewPage`. Internal unit fixtures may use the default; the public service must pass the smaller trusted runtime-derived limit. `SourceManifest` has immutable target digest, binding/epoch, exact comparison/revisions or captured hashes, opaque source refs, byte/line counts and explicit gaps. Its append-only source inventory has a separate publication revision: discovering committed context cannot change the target identity or invalidate participant submission revisions.

Use the existing `code_review_read` tool with optional `source_query`, valid only for `view="source"`. An initial request supplies either a document reference for content or a strict source query, never both. A continuation cursor binds the original normalized query, target digest, role scope and inventory snapshot; conflicting query/cursor pairs refuse. `list` returns bounded admitted relative names/opaque refs; `search` accepts literal text, not executable expressions, and returns bounded source refs/ranges; `resolve` maps an admitted relative path plus captured side to an opaque source ref. The host selects recorded object IDs, validates paths and excludes protected entries before returning names or reading blobs. Query arguments never grant access to current checkout files or arbitrary revisions.

Add optional public `source_range: SourceRange` only for source-content reads with a `document_ref` and no `source_query`. Omission on the initial read starts at line 1. A range must fit the captured line count; oversized spans are paginated within the requested bounds and the current evidence budget, never returned as one oversized page. Include actual delivered lines and source digest in the page. The opaque continuation binds document/target/epoch/role, requested range and next offset; subsequent calls may omit repeated selection fields or repeat them identically, but cannot widen or replace the range. Seeking near the end of a file must not require delivering all preceding lines to the model.

Fit the complete serialized `ReviewPage`, including text/records, digests, actual spans, final cursor and revision metadata, within `max_result_chars` as well as the token limit. Reduce whole source spans or query entries until it fits; compute cursor progress from delivered content only. A single line/entry that cannot fit yields `source_span_too_large_for_tool_result`/`review_record_too_large_for_tool_result`, with an explicit coverage gap rather than a clipped successful page or an empty page repeating the same cursor. Final public packing in Task 4 must recheck its actor-specific metadata before publication and cannot skip candidate content dropped while fitting. Discovery queries return refs/names/ranges; source bodies require a content read after target-specific instruction preparation.

- [ ] **1. Write a real-Git RED fixture.** Create two commits, a staged change, an unstaged change, an explicitly selected untracked file, an excluded secret, a binary and a symlink escape. Capture, edit the checkout, and assert source reads still return the captured bytes:

```python
def test_worktree_read_is_frozen_after_capture(source_case):
    manifest = capture_review_source(
        source_case.working_tree_target, source_case.authority,
        source_case.store, "review-1",
    )
    source_case.file.write_text("newer bytes\n", encoding="utf-8")
    page = read_review_source(manifest, source_case.main_ref(manifest),
                              start_line=1, max_tokens=1024)
    assert "captured bytes" in page.text
    assert "newer bytes" not in page.text
```

`source_case` is a local fixture in this test file built with `tmp_path` and real Git, exposing the target/authority/store/file and a manifest lookup by its known safe fixture path. Also mutate the file between pre/post capture checks and expect `source_drift`, with no coherent manifest published.

- [ ] **1a. Write discovery tests against unchanged callers and tests.** The committed fixture's changed module imports an unchanged helper and has an unchanged test file. Through source `list`/literal `search`/`resolve`, discover both, materialize from the pinned revision, and read their old bytes after HEAD/worktree move. Deny excluded names, symlink escapes, guessed object IDs, forged/cross-chat cursors and search outside scope. For working-tree/patch targets, capture an explicitly selected helper in `context_paths` and prove its bytes remain frozen; omit it in a second capture and require `context_not_captured` instead of falling back to current files.

```python
def test_resolve_unchanged_helper_uses_pinned_commit(source_case):
    manifest = capture_review_source(
        source_case.committed_target, source_case.authority,
        source_case.store, "review-1",
    )
    source_case.helper.write_text("new helper\n", encoding="utf-8")
    page = query_review_source("review-1", ResolveSourceQuery(
        op="resolve", relative_path="helpers.py", side="head",
    ))
    body = read_review_source(
        manifest, page.records[0]["source_ref"], start_line=1, max_tokens=1024,
    )
    assert "old helper" in body.text
    assert "new helper" not in body.text
```

The source fixture additionally exposes `committed_target` and `helper`, with `helpers.py` committed as `old helper` before capture. `query_review_source` runs with the fixture's trusted source authority bound; it cannot take an arbitrary repository root.

- [ ] **1b. Test internal range reads now.** Use a captured text fixture exceeding 50,000 lines. Read lines 50,000–50,010 directly and assert exact line numbers/digest, no preceding content and the evidence-page cap. Test zero/reversed/non-integer/out-of-document ranges, a range spanning several pages, exact final boundary, and forged/cross-document/changed-range cursors. The actual public-tool search-to-range test belongs to Task 4, after its provider exists.
- [ ] **1c. Test serialized source/query pages at narrow result limits.** With `max_result_chars=1024` and ample token space, require valid complete JSON pages, intact spans/digests/cursors and strictly advancing delivered offsets with no skipped content. Include multibyte text/escaping, metadata near the limit, a single oversized line/entry and a control envelope that cannot fit. Refuse explicitly instead of silently cutting evidence or returning endless empty pages. These are source-owner tests; native/fenced delivery and the real instruction ledger belong to Tasks 4/5.

- [ ] **2. Run RED:** `.venv/bin/python -m pytest Tests/Agents/test_code_review_source.py -q`.
- [ ] **3. Add minimal internal Git readers.** Validate revision tokens before resolving them; thereafter build argv from fixed flags and validated object IDs/literal confined paths. Reuse repository preparation and exclusions before enumeration or object reads. Do not splice arbitrary options, invoke textconv/external diff, load hooks, fetch automatically or expose raw Git tools to reviewers. Preserve existing `run_git` text API; share its bounded process implementation with a byte-returning internal result instead of decoding binary blobs and re-encoding replacements. Reject truncated revision/tree metadata, shallow/missing history and ambiguous comparison explicitly. SHA-1/SHA-256 object IDs must be validated against the repository's supported object format.
- [ ] **4. Implement all four target paths with recorded comparison semantics.** PR targets require exact local commits or already-authorized integration metadata; absent acquisition returns `history_unavailable`/`target_acquisition_unavailable`, allowing a supplied patch. SSH inputs retain filesystem-only authority: no local Git fallback. Single commits require an explicit chosen parent; PR/ranges record direct versus merge-base comparison. Patch targets use `parse_patch_targets` then `apply_patch_to_text` against admitted captured preimages; mismatches refuse and never modify the checkout. Working-tree capture records staged/unstaged layers and only explicitly included untracked files.

```text
capture:
  validate binding, relative scope and current epoch
  resolve immutable comparison or exact mutable input set
  reserve estimated bytes/files before materialization
  read bounded admitted bytes, preserving newline/encoding metadata and hashes
  recheck mutable input identities/hashes; changed -> discard unpublished capture
  publish source manifest only after required scope is coherent
```

Capture changed files and the explicit admitted `context_paths`, not the whole repository. For patch/working-tree targets, freeze the selected mutable context and its inventory before launching. Discovery searches only that captured set; an uncaptured mutable helper requires a new capture or an explicit coverage gap, even if its present bytes happen to match HEAD. For PR/commit targets, bounded lazy discovery may enumerate/read the recorded committed tree through the same source authority and reservations. Reuse existing list/search per-call entry/read/time limits; report incomplete search coverage and cursors instead of scanning indefinitely or silently omitting the tail. Materialized files, metadata and snippets remain subject to all review quotas. No raw filesystem/Git/history tool is restored to the child.

Required missing/binary/truncated scope blocks launch until the user chooses a supported narrower scope; optional context omissions become explicit coverage gaps without leaking excluded names. Preserve existing project-instruction activation without copying bodies into metadata/logs. The recipe explains the mutable-context limit before preparation so the user can include relevant callers/tests in the capture.

- [ ] **5. Run GREEN:** new source tests plus both Git test files and existing patch parser tests located by `rg --files Tests | rg 'patch_tool'`. Verify no checkout/index modification, exact source hashes, malicious option/path rejection, deleted/renamed files, byte limits and unavailable PR integration.
- [ ] **6. Review and commit:** `feat(agents): capture immutable permission-scoped review sources`.

## Task 4: Review service and role-aware tool contracts

**Files:** Create `Agents/code_review_service.py`, `Agents/code_review_tool_provider.py`, `Tests/Agents/test_code_review_service.py`, `test_code_review_tool_provider.py`. Modify `Agents/tool_catalog.py`, `Chat/console_agent_bridge.py`, input-validation tool schemas and preview composition where their current APIs require registration.

**Interfaces:** Produce `CodeReviewService.prepare(target: ReviewTarget, reviewers: tuple[ReviewRunChoice, ...] | None = None, verifier: ReviewRunChoice | None = None) -> PreparedReview`, `read(review_id: str, view: str, document_ref: str | None = None, cursor: str | None = None, source_query: SourceQuery | None = None, source_range: SourceRange | None = None) -> ReviewPage`, `submit(review_id: str, kind: str, records: list[dict], expected_revision: int, submission_id: str) -> SubmissionAck`, `release(review_id: str, expected_revision: int) -> ReleaseReceipt`. `submit` checks the bound slot-attempt submission revision; Task 9 adds the separate primary check-action stream. `release` checks the context-control revision. Bind trusted actor, response-character budget and accepted-work context when constructing/invoking the provider, not in tool JSON. Produce `CodeReviewToolProvider` using the existing `ToolProvider` and optional `PathAwareToolProvider` protocols, including `path_targets(tool_id: str, args: Mapping[str, Any]) -> tuple[ToolPathTarget, ...]`. Target captures/discovery and role-specific revision fields use the clarified spec contract; none of these are generic filesystem or routing bypasses.

Produce the internal `CodeReviewService.source_instruction_targets(review_id: str, *, document_ref: str | None = None, cursor: str | None = None, source_query: SourceQuery | None = None) -> tuple[ToolPathTarget, ...]`. Derive actor/role/session/epoch and revalidate admitted source scope before returning any path. Map source refs or cursor selections through the captured manifest to their original binding-relative project paths, never the physical scratch capture. Resolve queries map a validated relative file; list/search queries map their admitted directory or binding root, matching existing broad-search behavior. Only source-content reads deliver source bodies, after exact-target activation. Other review views return no path targets. Foreign/excluded/forged/expired refs cannot produce a path or cause instruction discovery outside authority.

Define the internal authority seam now: a process-local issuer owns `ReviewStageGrant` objects bound to session, accepted message/acceptance receipt, permitted stage, epoch and target/team/override fingerprints. The service accepts only registered live issuer objects, never deserialized equivalents. The combined authority/ledger slice supplies native request/acceptance issuance. Unit fixtures use the same issuer with a trusted fixture owner; they cannot bypass role, epoch or reference checks. Follow existing run-actor binding across tool threads rather than adding identity fields to every tool.

- [ ] **1. Write RED parameterized role tests.** A reviewer can read its source and submit its own findings/coverage, but cannot read report/index/summary provenance, prepare/release, or submit another slot's output. A verifier gets source and blinded batches only. A primary sees reports/index/summary and cannot impersonate a reviewer submission.

```python
def test_guessed_review_id_does_not_grant_report_access(review_harness):
    review_id = review_harness.prepare()
    result = review_harness.invoke(
        role="reviewer", tool="code_review_read",
        arguments={"review_id": review_id, "view": "report"},
        review_id=review_id,
    )
    assert result.ok is False
    assert "role_denied" in result.error
```

Repeat with another chat, absent run actor, expired bundle, forged document ref, stale source epoch, malformed cursor, disabled feature and direct provider invocation outside a run.

- [ ] **1a. Test public search-to-range access.** Invoke the real `code_review_read(view="source", source_query=...)` tool against the large captured fixture, discover a literal hit at line 50,000, then read that source reference with `source_range={"start_line": 50000, "end_line": 50010}` in one content-read call. Assert only the selected lines/digest reach the model, within the evidence cap. Through the same public schema, reject range+query, range on another view, missing document ref on the initial range read, out-of-bounds spans and changed-range cursor replay. Repeated small pages must remain within the original selection.
- [ ] **1b. Test whole-response packing through the real provider.** Give the bound actor `max_tool_result_chars=1024` and a generous model window. Read multi-page source/query fixtures and inspect the exact canonical response, including JSON escaping, cursor, digest and submission/context revisions. Assert every successful response fits, parses whole and advances without losing entries; if one record or the minimum control envelope cannot fit, return a bounded stable refusal. Preflight prepare/submit/release acknowledgment size before capture, publication or tombstoning; a transport refusal cannot leave an unacknowledged mutation. Index/report packing tests belong to Task 6 after its index exists. Do not raise the runtime limit or rely on run-log recovery.
- [ ] **1c. Test actual instruction-ledger preparation before public source reads.** Using the real registry, provider and `InstructionActivationLedger`, capture an unchanged helper under `pkg/nested/` with nested `AGENTS.md`. Resolve its opaque ref and a continuation-only read; require the provider to map the original project path and the ledger to defer dispatch until that chain receives the guidance. Guidance preparation cannot consume the source cursor or materialize another source file. Check replay does not redeliver guidance, root versus nested query scopes, no capture-path targeting, and foreign/excluded/stale/cross-chat refs causing no unauthorized instruction read. Keep existing disabled/outside-binding/consent/token-omission behavior. Actual child-model delivery is tested in Task 5, avoiding a new forward dependency here.

- [ ] **2. Run RED:** `.venv/bin/python -m pytest Tests/Agents/test_code_review_service.py Tests/Agents/test_code_review_tool_provider.py -q`.
- [ ] **3. Implement preparation without paid work.** Convert omitted run choices from saved stable-ID defaults, then overlay only explicitly supplied per-run choices/focus/destination. Resolve every effective reviewer and verifier with `resolve_spawn_target(app_config, parent_provider=primary_provider, parent_model=primary_model, preset=captured_definition, override_provider=requested_provider, override_model=requested_model, routing=current_routing)`, using empty override strings only when absent. Preserve ADR-147's enable gate, model-only and final-pair allowlist checks, readiness, destination consent and child-owned parameters. Freeze requested overrides, resolved targets and fallback candidates together; neither preparation nor guide loading writes settings/presets or enables overrides. Do not inherit an unresolved destination from the primary. Show duplicate resolved destinations, missing/disabled presets and missing verifier. Read actual current remaining child slots plus finite chain allowance before capture; zero slots refuses `child_budget_unavailable`. Reject known inadmissible schedules before materialization. Revalidate route authority at launch even though captured identities are frozen. Initialize context/slots under store reservation; return small opaque references.

- [ ] **3a. Test the run-only routing input end to end.** Cover reviewer and verifier provider-only, model-only and paired overrides, disabled overrides, disallowed final pairs, unavailable destinations and duplicate effective routes. Compare the saved team and real preset DB rows before/after. Preparation must either return the frozen requested target or the existing routing/consent refusal; it cannot silently substitute. Extend the dispatch test in Task 5 to assert those actual gateway arguments after a later settings/preset edit.
- [ ] **4. Implement closed schemas and projections.** Public operations remain guide/prepare/read/submit/release, including the explicit run-choice and source-query fields defined above; unknown keys and stages refuse. Resolve role and submission stream from actual run binding. Expose only the caller's own submission revision, never peer stream state. Publish accepted chunks using Task 2's stream-local CAS/idempotency; validate source refs/ranges and immutable local keys before writing. Assign `R1-F1` once, in host order. Return bounded schema/version/status/size metadata to logs, previews and UI traces; no automatic source, guide, focus, search text or raw report body projections. The guide operation must never allocate a context or read a user file.
- [ ] **4a. Fit complete model-bound responses before mutation or delivery.** Derive `review_result_char_limit` from the actual run, finalize role-filtered metadata, then serialize the complete response. Pack fewer whole records/source spans when needed, recomputing the delivered cursor; never truncate JSON, evidence meaning or the cursor itself. Check the exact candidate control acknowledgment, including assigned IDs/revision under the publication lock, before committing a prepare/submit/release operation; preflight bounded release-status variants before tombstoning. A single indivisible record refuses `review_record_too_large_for_tool_result`; a minimum control envelope that cannot fit refuses `review_response_capacity_unavailable` before paid work or mutation and retains a host-visible diagnostic. Source-line refusals use Task 3's reason. This is additional to byte/token limits; the existing generic truncator stays unchanged. Task 8 adds claim-specific admission and Task 9 paginates check evidence through this same packer.
- [ ] **4b. Wire captured references into the existing activation ledger.** Implement `path_targets` through `source_instruction_targets`, with metadata-only lookup and current authority validation. Do not load instruction bodies, reserve/capture content, advance cursors or activate guidance inside the provider mapping. The existing registry owner lookup and `prepare_tool_calls` own ledger activation, per-chain deferral, payload receipts and destination consent. Relevant nested guidance remains ephemeral user context; the normal guard/permission review follows its delivery. Preserve current instruction-source staleness/symlink/exclusion rules instead of adding a second revision-pinned instruction loader. No new path argument or instruction-body tool result is exposed.
- [ ] **5. Wire live/preview parity and progressive discovery.** Apply feature/config/ephemeral/policy gates before registration. Primary operations disappear for children; direct guessed calls still refuse. Preview computes the same schemas and guidance without source capture, reservation, approval or model work. Existing non-review calls retain identical routing and permissions.
- [ ] **6. Run GREEN:** both new files plus `.venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_agent_bridge_local.py Tests/Agents/test_local_tool_provider.py -q`. Assert prepare invokes no model gateway and zero child launches.
- [ ] **7. Review and commit:** `feat(agents): expose scoped review preparation and report tools`.

## Task 5: Scoped child launch and exhaustive reviewer isolation

**Files:** Modify `Agents/agent_models.py`, `agent_runtime.py`, `agent_service.py`, `tool_catalog.py`, `Chat/console_agent_bridge.py`. Extend the new service/provider and test helpers. Create `Tests/Agents/test_code_review_dispatch.py`, `test_code_review_isolation.py`; extend routing integration tests.

**Interfaces:** Add the mutually exclusive wire spawn form `spawn_subagent(review_id: str, review_slot_id: str)`. Service-side `claim_review_slot(review_id: str, slot_id: str) -> ReviewLaunchClaim` atomically returns an existing receipt or one pending launch claim. `ReviewLaunchClaim` binds the frozen target/definition/focus/requested override/role/target digest and accepted launch identity. `attach_review_run(claim: ReviewLaunchClaim, run_id: str) -> None` publishes the real receipt; uncertain launch cannot be retried automatically. Add `AgentConfig.review_binding: ReviewRunBinding | None = None` as a host-owned field, carried into actual tool threads. Spawn cannot take a fresh provider/model override: preparation has already resolved and frozen it, and launch rechecks current authority without re-resolving from edited preferences.

`attach_review_run` calls the already-built store's `bind_attempt` with the real accepted run association before that child can invoke review tools. An unresolved launch cannot acquire a second association or stream by replaying a model request.

- [ ] **1. Write RED tests through actual `AgentService`/gateway dispatch.** Use two provider/model fixtures with different sampling and fallback destinations; overlap their calls with a barrier. Assert actual gateway arguments, not just config. Retry the same slot and count one accepted dispatch. Reject any mixture of scoped fields with task/agent/provider/model/isolation, including fence-protocol calls and direct Python invocation.

```python
def test_scoped_spawn_uses_captured_child_destinations(review_harness):
    observed = review_harness.run_script(reviewers=2, verify=False)
    children = [r for r in observed.gateway_requests if r.role == "reviewer"]
    assert {(r.provider, r.model) for r in children} == {
        ("review-provider-a", "review-model-a"),
        ("review-provider-b", "review-model-b"),
    }
    assert observed.actual_child_launches == 2
    assert all(r.parent_model_was_used is False for r in children)
```

These are scripted gateway identifiers, not claims about real provider availability. Fixture request observations are recorded at the adapter that would call `chat_api_call`.

- [ ] **2. Run RED:** `.venv/bin/python -m pytest Tests/Agents/test_code_review_dispatch.py Tests/Agents/test_code_review_isolation.py -q`.
- [ ] **3. Implement host-composed tasks and slot launch.** Validate scopes before decrementing/reserving execution budgets, then let existing accepted child accounting own spending. Once accepted, cancellations/failures do not refund the child. Compose the same shared brief and source contract, adding captured preset instructions and focus. Do not pass primary conversation history, personal-context blocks, unrelated workspace notes, suspected findings or peer outputs. Pass only relevant admitted project instructions under their existing ephemeral contract. Record requested route and actual fallback changes from provider events, never model text.
- [ ] **3a. Bind check-feedback mode from actual dispatch.** Immediately before choosing the native spawn branch, recheck whether this verifier will be a threaded fleet child or `fleet is None or inline`. Bind `check_feedback_mode` as `primary_handoff` or `unavailable_inline` before its first tool call, and include the bounded capability diagnostic in its host-composed task. Preparation may preview availability but cannot freeze a stale concurrency choice. Preserve the `max_live_subagents=1` kill switch, synchronous spawn result/receipt semantics and ordinary launch budgets; never force a fleet or advertise `wait_agents` in inline mode. Task 9 implements the proposal result for each mode.
- [ ] **4. Close actual capability gates.** For review-bound children allow only review source reads and role-owned submissions, plus narrowly required discovery. Disable mutation, raw shell/process/Python/CLI, skill scripts/install/file escapes, new/fork chat, run-log/history, Library/profile, peer messaging, worktree merge/discard and nested spawning at both schema and dispatch gates. Review source tools must not resolve arbitrary physical scratch paths. Apply the same deny when a tool name is guessed without disclosure. Verifier source access has the same restrictions; executable checks use Task 9's separate authorized host path.

```text
review-bound runtime dispatch:
  assert live run association, target digest, epoch and effective authority
  if capability is outside the role's closed set -> role_denied
  dispatch allowed review operation through the same bound actor
ordinary runtime dispatch:
  keep existing behavior
```

- [ ] **5. Complete the integration harness at the real seams.** Address scripted child replies by run/slot, using the existing `FleetChat` pattern so concurrent threads do not race for one global reply queue. Record outbound requests, accepted launch receipts, final destinations and native settlement. Include direct forbidden-call probes, a peer report canary, an unrelated personal-context canary and a project-instruction delivery control.
- [ ] **5a. Race actual concurrent structured submissions.** Make both child gateway scripts submit from their own revision 0 with identical local keys/submission IDs, using a barrier at the real provider invocation. Assert both accepted records appear exactly once with distinct original IDs and both streams advance independently. Separately cause a same-stream conflict and verify the single unchanged-payload retry and finite failure path. Do not bypass the tool provider or replace `publish` with a mock. Add raw-output interleaving in Task 6 when the capture tap is implemented.
- [ ] **5b. Prove nested guidance and complete pages at the child boundary.** Use an actual scripted review child and real activation ledger to read the unchanged nested helper, including discovery followed by an opaque source read. Assert the provider request carrying its source evidence follows the request carrying nested guidance, one activation delivery is acknowledged per chain, no body leaks into logs/metadata, and the later permission gate remains effective. Retained request context may still contain the already-delivered guidance; do not mistake that for another activation. Parameterize native/fenced calls and `max_tool_result_chars=1024` with run logs disabled; inspect actual model-bound JSON pages and all cursors/revisions after the generic append seam. Test an indivisible oversized source line as explicit incomplete coverage. Verifier claim/check-result delivery belongs to Tasks 8/9, where those APIs exist.
- [ ] **5c. Test launch-mode revalidation without check execution.** With the actual inline service (`max_live_subagents=1`, no injected coordinator), assert synchronous scoped reviewer/verifier source reads, correct run binding and no fleet wait schema. Use the Task 4 issuer's source-only verifier fixture with an admitted stage, not a full native review journey dependent on Tasks 6/7. Switch settings between preparation and verifier launch in both directions and require the binding/task diagnostic to match actual dispatch. Task 9 adds the proposal/wait assertions; this gate does not call its not-yet-built handshake.
- [ ] **6. Run GREEN:** both new tests plus `Tests/Agents/test_agent_routing_integration.py`, `test_fleet_runtime.py`, `test_fleet_continuation.py`, `Tests/Chat/test_console_agent_project_instructions.py`. Check ordinary generic spawn and worktree paths remain unchanged.
- [ ] **7. Review and commit:** `feat(agents): launch isolated review slots through existing fleet`.

## Task 6: Original report capture, sealing and complete index

**Files:** Extend `Agents/code_review_service.py`, `code_review_store.py`, `agent_service.py`. Create `Tests/Agents/test_code_review_reports.py`. Use existing fleet result rendering without increasing its global caps.

**Interfaces:** Produce `capture_review_output(run_id: str, text: str, *, terminal: bool) -> None`, `settle_review_run(run_id: str, *, terminal_status: str, receipt_id: str) -> None`, `finding_index(review_id: str, cursor: str | None = None) -> ReviewPage`. Raw capture and structured submissions count together toward the participant and aggregate report caps. Host envelopes contain actual requested/active routes/fallbacks, target/guide/preset fingerprints, capture status and seal status.

- [ ] **1. Write RED cases for order-independent completion.** Structured completion before terminal success stays pending; terminal success before coverage stays pending. Successful empty findings plus explicit coverage seals. Failed/cancelled/exhausted/malformed/unstructured roles do not seal as successful empty reports. Findings accepted before failure remain readable with incomplete status.

```python
def test_terminal_success_without_coverage_is_not_empty_review(review_harness):
    review_id = review_harness.prepare()
    review_harness.finish_reviewer(review_id, coverage=False)
    summary = review_harness.read(review_id)
    assert summary["slots"][0]["report_status"] == "incomplete"
    assert summary["slots"][0]["empty_findings"] is False
```

- [ ] **2. Run RED:** `.venv/bin/python -m pytest Tests/Agents/test_code_review_reports.py -q`.
- [ ] **3. Tap raw output before shortening.** Capture bounded policy-permitted output at the child result/stream owner before `_safe_terminal_result` and fleet aggregation lose it. Avoid double capture when stream events and final text contain the same content. Record omissions/withholding/overflow explicitly. A run row or shortened `wait_agents` result is never the original report. Check cancellation and late event paths still consult the tombstone and leases.
- [ ] **4. Publish structured reports and index mechanically.** Render authoritative originals from validated records, with host attribution separate from raw output. Index every accepted original ID, its record ref, report completeness and coverage; page the index without reading raw prose into the primary prompt. Completion is a conjunction of valid records/coverage, actual terminal success and capture/seal receipts. A rejected record marks a coverage/format gap unless corrected within the same finite run; never loop indefinitely for repair.
- [ ] **4a. Keep host capture separate from agent append revisions.** Raw stream offsets and terminal/coverage receipts publish through host-owned sequences. They cannot consume a reviewer's `submission_revision`. Original finding IDs are assigned under the existing publication lock, so interleaved R1/R2 submissions and capture callbacks cannot reuse or renumber IDs. A new explicit retry is a new bound attempt/stream; it retains earlier originals and receipts.
- [ ] **4b. Add capture interleaving at its owning gate.** Extend the Task 2 store race and Task 5 real threaded child-submission scenarios here, invoking the implemented capture tap between appends and during terminal settlement. Assert no capture-induced stale revision, both originals retained exactly once, stable IDs, truthful raw completeness and correct combined quota charges. Keep the original Tasks 2/5 tests runnable before this slice; no fake capture interface is needed to make their earlier GREEN checks pass.
- [ ] **4c. Test complete report/index response packing.** Through the already-built public provider, page originals/index/raw output with a small result-character limit and intact cursor/revision envelopes. Assert no accepted finding disappears at the generic truncation seam and no cursor skips a dropped candidate. An indivisible record too large for this transport stays intact/readable through its storage owner and yields an explicit model-read gap; it is never silently shortened into another authoritative finding. Task 9 tests native report-surface access after its reader exists.
- [ ] **5. Run GREEN:** report tests plus existing `Tests/Agents/test_fleet_reliability.py` and `Tests/Chat/test_console_agent_bridge.py`. Force both combined-result and run-summary truncation; assert separately read originals/index remain complete or explicitly incomplete. Test 1 MiB participant overflow and 16 MiB aggregate overflow with accepted earlier chunks preserved.
- [ ] **6. Review and commit:** `feat(agents): preserve structured original review reports and receipts`.

## Task 7: Native review authority, wake ledger and bounded continuations

**Files:** Create `Chat/console_code_review.py`, `Tests/Chat/test_console_code_review_authority.py`, `test_console_code_review_continuation.py`, `Tests/Agents/test_code_review_boundary_ledger.py`. Modify `DB/AgentRuns_DB.py`, `DB/automatic_work.py`, `Agents/automatic_work_budget.py`, `Chat/console_fleet_wake.py`, `console_chat_controller.py`, `console_agent_bridge.py` and existing native acceptance wiring. Extend `Tests/Agents/test_agent_runs_wake_ledger.py` and the real Console harness. Keep source/report/grant bodies out of SQLite.

**Dependency gate:** Build the native issuer and validate its real receipts first, then use that already-built coordinator in ledger tests, then wire automatic delivery. Native authority and ledger integration form one independently reviewable slice; no positive test depends on a coordinator scheduled in a later task. Intermediate GREEN checks establish prerequisites inside this slice, not a shipped continuation feature or a separate task completion.

**Interfaces:** Produce session-owned `ConsoleCodeReviewCoordinator` with `bind_accepted_request(message_id: str, *, session_id: str, source_epoch: int) -> None`, `accept_stage(review_id: str, stage: Literal["verify", "retry_reviewer", "retry_verification"], *, accepted_message_id: str, choice: ReviewRunChoice | None = None, slot_id: str | None = None) -> PreparedReview`, `seal_next_boundary(review_id: str) -> ReviewBoundaryReceipt | None`, `validate_boundary(receipt: ReviewBoundaryReceipt) -> bool`, `fence_session(session_id: str) -> None`. `accept_stage` is a native callback, never a model tool. It validates actual accepted user provenance and returns newly admitted slot references/context-control revision under the same target. Explicit retries get new attempt/submission-stream identities; prior originals/receipts stay immutable. A later chosen verifier or override is routed/frozen through the same preparation admission, then bound to the exact new stage grant.

Extend `AutomaticWorkLedger.claim_wake` with optional host-issued `review_boundary: ReviewBoundaryReceipt | None = None`; an empty run/progress batch is valid only with a registered live boundary. `AutomaticWakeAttempt` adds an optional body-free boundary identity projection. The same issuer supplies thread-safe live receipt validation at claim and acceptance; the DB owner does not import UI or derive authority from serialized JSON. Add `AgentWakeCoordinator.on_review_boundary(receipt: ReviewBoundaryReceipt) -> None`, feeding existing pending/claim/accept/deliver machinery rather than launching a model directly. Existing `accept_wake` and `AutomaticWorkContext` remain acceptance owners, using normal `fleet_wake` limits and attempt kind.

- [ ] **1. Write the native authority RED tests first.** A plain review request grants reviewers only; an explicit trusted review-and-verify request grants both stages; later acceptance grants verification for that captured review. Tool booleans, quoted/tool/repository text, automatic messages, another session and declined/unanswered acceptance cannot grant. Already authorized verification causes no second approval question. Preserve separate routing consent and permission floors. Include run-only override fingerprints in the grant so another destination cannot borrow it.
- [ ] **2. Run authority RED:** `.venv/bin/python -m pytest Tests/Chat/test_console_code_review_authority.py -q`. Expect the new coordinator import/authority cases to fail.
- [ ] **3. Build and verify the native issuer before ledger tests use it.** Bind only an actual owning Console `USER` message with no automatic origin, following the existing `trusted_profile_user_message_id` validation pattern. Recognize direct explicit review/verification instructions; test negation, quoted examples and conditional language. Ambiguity uses the existing native acceptance boundary, never a model summary or quoted substring. Bind acceptance to exact stage/review/target/preset/override fingerprints. Issue a boundary only after actual predecessor settlements and for the exact next admitted slot/stage, with current source epoch, context-control revision, original chain and runtime owner. Keep grants/receipt registry session-local and thread-safe. Run the authority tests to GREEN; now the ledger fixture can obtain real coordinator-issued receipts without bypassing issuance.

- [ ] **4. Write ledger/migration RED tests using that live coordinator.** A valid boundary claims once; identical attempt replay returns the same attempt; conflicting replay refuses. Reject fake/missing/foreign/obsolete receipts, unfinished predecessors, wrong grant/epoch/runtime/chain and consumed slots. Ordinary empty wakes still refuse:

```python
def test_empty_ordinary_wake_remains_invalid(ledger_case):
    with pytest.raises(ValueError, match="must not be empty"):
        ledger_case.ledger.claim_wake(
            ledger_case.chain_id, attempt_id="attempt-1",
            owner_id=ledger_case.owner_id, session_id="session-1", run_ids=(),
        )
```

`ledger_case` is a real SQLite ledger/runtime/finite-chain fixture. Its `live_boundary` is issued by the coordinator built in Step 3 after actual scripted child settlement; it cannot fabricate a receipt from model JSON. Also add claim → proven pre-acceptance abort → new claim with a different attempt ID, and accepted/completed/uncertain attempts → forbidden reclaim. Run `.venv/bin/python -m pytest Tests/Agents/test_code_review_boundary_ledger.py -q` to RED.

- [ ] **5. Add the existing-owner migration and claim transaction.** At the inspected baseline bump AgentRuns schema **24 → 25** using its idempotent initialization/version-audit convention. If intervening changes consumed 25, allocate the next unoccupied version and update the plan/ADR/task notes before DDL. Add body-free `automatic_review_boundary_claims`, keyed by opaque boundary ID, unique attempt and exact `(owner, chain, review, slot, slot-attempt)` identity, with source epoch/fingerprints and predecessor settlement IDs. Leave existing `cause`/`attempt_kind` CHECK constraints intact: the linked claim identifies review eligibility and consumes a normal fleet-wake generation. Do not create a review-job database or persist source/report bodies.

Within the existing admission transaction, validate the registered live receipt/current owner/conversation/finite allowance, verify real predecessor settlements, reserve one generation and insert the boundary link atomically. A boundary-only attempt may have an empty ordinary source batch; all ordinary empty attempts remain invalid. Mixed completion/progress/review eligibility consumes one generation. Revalidate live source/grant/epoch ownership at acceptance before any billable preparation.

- [ ] **6. Implement the complete claim lifecycle in the existing owners.** Inspect and extend `_attempt`, `_attempt_view`/`read_attempt`, `_accept_attempt`/`accept_wake`, `_abort_attempt`/`abort_wake`, `complete_wake`, startup recovery and coordinator pending/fence bookkeeping together. Use the following state contract:

| Existing attempt outcome | Boundary link / registry action | Reservation and replay rule |
| --- | --- | --- |
| Prepared, still pending | Retain its exact link; identical claim returns the same attempt | One reservation; no second launch |
| Proven pre-acceptance refusal or manual-priority yield | In the same abort transaction delete that attempt's active boundary link, alongside existing source claims; keep the aborted attempt audit row. Re-enable the live receipt only if its grant/epoch/owner remain valid and no stop/close fence applies. | Release only the proven unaccepted reservation. A fresh attempt ID may reclaim the same still-live boundary; replaying the aborted ID returns aborted. |
| Accepted | Retain the immutable link and mark the live registry accepted | Never delete/refund/reclaim, including cancellation before a child visibly starts |
| Completed | Retain the consumed link with the completed attempt; clear pending delivery | A duplicate callback/new receipt cannot dispatch the same slot-attempt again |
| Accepted outcome uncertain, owner lost or restarted | Retain link/charges and enter existing `review_required` recovery | No automatic replay or reconstructed ephemeral context |
| Stop/close/source fence | Revoke live eligibility first; abort only proven prepared attempts, retaining accepted/uncertain links | No stale callback can reactivate or claim the boundary |

This deliberately extends the existing abort helper, which currently knows only completion/progress claims. Delete only the aborted attempt's active link; retain accepted/completed evidence. A later explicit retry/resume is new accepted user work with fresh slot-attempt authority, never deletion of consumed evidence. Recovery cannot reconstruct an expired bundle from its IDs.

- [ ] **7. Run ledger GREEN and race controls before delivery wiring.** Run the new ledger tests plus `Tests/Agents/test_agent_runs_wake_ledger.py`, `Tests/Chat/test_automatic_wake_budget.py`, `test_automatic_wake_recovery.py`, `test_automatic_work_lineage.py`. Cover schema upgrade/reopen, ordinary completion/progress/goal controls, abort → reclaim exactly once, aborted-ID replay, completion replay, accept-versus-abort, close during DB commit, wrong-runtime recovery and no charge refund after acceptance. Assert link rows and actual reservation spending, not only returned statuses.

- [ ] **8. Write the real Console launch-count matrix and run RED.** Drive actual scripted model/tool turns and record all launches, including earlier ordinary spawns. At this slice use sealed zero-finding reviewer reports and a fresh verifier that returns without calling the batching APIs built in Task 8; assert launch/ordering/acceptance, not completed verification. Full disposition/history behavior belongs to Tasks 8 and 11, avoiding another forward test dependency:

```python
@pytest.mark.parametrize("reviewers,verify,expected", [
    (2, False, [2]),
    (2, True, [2, 1]),
    (3, False, [2, 1]),
    (3, True, [2, 2]),
])
def test_default_two_child_schedule(review_harness, reviewers, verify, expected):
    observed = review_harness.run_script(reviewers=reviewers, verify=verify)
    assert observed.launches_per_primary_turn == expected
    assert len(set(observed.chain_ids)) == 1
    assert observed.original_reports_presented_before_verifier
```

Run `.venv/bin/python -m pytest Tests/Chat/test_console_code_review_continuation.py -q`. With one prior spawn only one review child fits initially; retain that prior launch in the total. With two prior spawns refuse preparation before capture.

- [ ] **9. Wire wake delivery through existing acceptance and fences.** Stage the live boundary in pending eligibility, claim through Step 5, and propagate its body-free identity through `AgentWakeAuthorization`. Preserve automatic origin, original chain/allowance root, finite generation/child/call/token/time spending, manual draft/queue priority, two automatic-primary capacity slots and manual reserve. Deduplicate normal child completion and review-boundary notifications; one accepted wake can advance the already-admitted review schedule without minting another generation. Never synthesize survivor results or manual queue entries.

On disabled/exhausted/fenced/unknown acceptance show `continuation_blocked` with retained progress. Manual resume is new explicit user work using the retained context and normal new-turn budget, not an automatic allowance reset. Stop revokes pending boundaries before child cancellation; close/context replacement tombstones access. Uncertain accepted attempts require review without replay; restart cannot continue expired session state.

- [ ] **10. Run the whole slice GREEN, review and commit.** Run all three new authority/ledger/continuation files, the existing wake suites from Step 7, and `Tests/Chat/test_console_fleet_wake.py`, `test_automatic_wake_dispatch.py`, `test_automatic_wake_scheduling.py`, `test_console_close_session_fleet.py`, `test_fleet_runtime_admission.py`. Inject duplicate callbacks and manual-priority/acceptance cancellation races; count actual launches/reservations. Commit only when the combined slice passes: `feat(console): admit review continuations with complete wake claim lifecycle`.

## Task 8: Blinded units, context admission and incremental verifier ledger

**Files:** Create `Agents/code_review_verification.py`, `code_review_history.py`, `Tests/Agents/test_code_review_verification.py`, `test_code_review_context.py`, `test_code_review_history.py`. Extend service/store, `Agents/agent_service.py` request construction/capacity and scoped verifier binding. Reuse existing protocol-round/continuation ownership helpers; do not change the primary-only run-log eviction gate or expose run-log tools to verifier children.

**Interfaces:** Produce `project_claim(finding: FindingRecord, manifest: SourceManifest) -> dict`, `build_verification_units(originals: tuple[dict, ...], manifest: SourceManifest) -> VerificationUnitSet`, `next_verification_batch(review_id: str, *, remaining_input_tokens: int, estimate: Callable[[str], int], max_result_chars: int, cursor: str | None) -> ReviewPage`, `expand_dispositions(review_id: str) -> tuple[dict, ...]`. `max_result_chars` is the trusted run-derived limit already used by the public response packer, never a wire override. `VerificationUnitSet` contains private original→unit map and public unit records; the provider never serializes its private fields. Produce `AgentService.review_input_capacity(config: AgentConfig, api_endpoint: str, request: ModelRequest) -> int | None`, using the existing exact message/tool estimator and response reserve; `None` means unknown, not unlimited.

Also produce `CodeReviewService.verifier_checkpoint(review_id: str) -> ReviewHistoryCheckpoint` for the bound live verifier and `project_verifier_history(messages: list[dict], checkpoint: ReviewHistoryCheckpoint, *, continuation_groups: tuple[ContinuationOwnerGroup, ...], native: bool) -> list[dict]`. The projection runs in the service-owned request builder before effective-capacity estimation and final provider-continuation preparation. It is a view of this one run's send history, not another run or a general compaction engine. Existing trusted system/task/instruction assembly remains authoritative.

- [ ] **1. Write RED projection tests.** Build two identical substantive originals from different reviewers and one with a changed trigger. Assert two opaque units, a private three-original map, no original IDs/model/severity/order/receipt/multiplicity fields in serialized verifier pages, stable opaque ordering and distinct verdict requirements for the changed trigger. Normalize evidence to source ref/range/digest before equality; never compare participant read receipts.

```python
def test_only_exact_substantive_duplicates_share_units(verification_case):
    units = build_verification_units(
        verification_case.three_originals, verification_case.manifest
    )
    assert len(units.public_units) == 2
    mapping = units.original_to_unit
    assert mapping["R1-F1"] == mapping["R2-F1"]
    assert mapping["R3-F1"] != mapping["R1-F1"]
    public = json.dumps(units.public_units, sort_keys=True)
    assert "R1-F1" not in public
    assert "severity" not in public
    assert "multiplicity" not in public
```

`verification_case` constructs valid original envelopes/manifest using Task 1 builders; its two exact claims differ only in fields explicitly stripped from the public projection. Test shuffled reviewer/report input order as well as exact repeats.

- [ ] **2. Run RED:** `.venv/bin/python -m pytest Tests/Agents/test_code_review_verification.py Tests/Agents/test_code_review_context.py Tests/Agents/test_code_review_history.py -q`.
- [ ] **3. Implement deterministic substantive projection plus random unit identity.** Select target, locations, trigger, impact, normalized evidence and uncertainty only. Canonical JSON bytes define equality; UUID randomness defines unrelated public IDs. Keep the reversible map private in the charged ledger. Stable opaque order is independent of reviewer order. Do not add duplicate proposals/group counts to initial batches. A verifier may later propose related units but still submits an independent disposition for each nonidentical unit.
- [ ] **4. Implement explicit checkpoint-backed history reduction before budgeting a batch.** Existing `log_active` requires a primary agent, and `bound_history_for_send` is a no-op for these verifier children. Do not depend on that path, the user's eviction preference or generic Console compaction. At each protocol-coherent request boundary:

```text
load the live verifier's accepted checkpoint and verify attempt/target/stream revision
pin trusted system + original verifier task + currently admitted project guidance
pin the active batch's bounded claim/evidence refs and pending host check receipts
use existing round/continuation ownership utilities to identify whole protocol groups
keep unacknowledged/current tool calls, their paired results and required private state
replace only fully completed checkpoint-recoverable rounds with bounded checkpoint refs
retain current-batch evidence until that batch's accepted decisions are checkpointed
build the exact send request using this same projection in staging and live dispatch
prepare/validate provider continuation, estimate its full effective input and reserve
if mandatory protocol state still cannot fit, refuse review_context_reduction_blocked
```

The host checkpoint includes only this verifier's opaque units/cursor and authorized evidence/check refs, never original attribution or peer reports. Pending claim bodies remain in the host ledger and can be fetched through bounded batch reads. Source excerpts that an unfinished disposition still depends on cannot be discarded before acknowledgment. Never split native call/result pairs, replay already executed tools, strip required reasoning/private owner state, silently reset a provider conversation or start a new verifier. When the provider's indivisible continuation group prevents reduction, retain accepted progress and finalize remaining units unresolved rather than asserting that the claim itself is oversized. Normal run/chain call/token/time/child budgets still apply.

- [ ] **4a. Write a real-child RED/GREEN history regression.** Extend the test harness with `run_batched_verifier(*, units: int, context_window: int, native_tools: bool, run_log_enabled: bool) -> ReviewRunObservation`. It prepares scripted structured claims, starts one actual verifier through `AgentService`, checkpoints through real tool submissions and records exact outbound requests using the existing estimator. Do not manually trim the fixture's message list or mock `project_verifier_history`.

```python
@pytest.mark.parametrize("native_tools", [False, True])
def test_verifier_child_reduces_history_without_run_log(review_harness, native_tools):
    observed = review_harness.run_batched_verifier(
        units=24, context_window=8192, native_tools=native_tools,
        run_log_enabled=False,
    )
    assert observed.verifier_launch_count == 1
    assert observed.history_reductions > 0
    assert observed.all_provider_requests_fit
    assert len(observed.accepted_unit_dispositions) == 24
    assert observed.replayed_tool_executions == 0
```

Add these observation fields to the test helper, derived from actual run IDs, request projections, ledger acknowledgments and tool-execution receipts. Choose bounded fixture claims/evidence so the unreduced transcript exceeds the 8,192-token window but every individual batch fits after projection; pin finite test run/chain budgets sufficient for those calls without resetting them. Cover restored references after reduction, active native tool pairs, private continuation state, an irreducible owner group, unchanged primary/generic child requests, and both disabled and enabled global run-log settings.

- [ ] **4b. Implement capacity-admitted batching using the projected request.** Use actual effective verifier model/provider and the fully staged system/task/history/tools/guidance/reserve. Reuse `Utils/token_counter.py::resolve_context_window` and `Chat/console_prepared_request.py::resolve_request_capacity` for effective server/model/user/provider limits, plus the existing estimator; do not introduce a second tokenizer or rely on character counts. A provider/application fallback with `verified=False` is unknown for this feature unless an explicit effective user/provider input cap establishes capacity; do not accidentally treat `get_model_token_limit`'s numeric fallback as known capacity. Preserve the existing project-instruction final-request fit check and add the corresponding final check for review-bound requests, including when no project instructions are active. Unknown/invalid capacity blocks explicitly. After reducing recoverable history, before emitting a batch compute:

```python
claim_budget = min(4096, remaining_input_tokens // 2)
evidence_budget = min(1024, remaining_input_tokens)
```

Then choose at most eight whole units whose serialized public records fit `claim_budget` and whose complete response envelope fits `max_result_chars`. Only a unit too large for an otherwise empty batch after recoverable-history reduction becomes unresolved `claim_too_large`; a unit that only fails to fit the remainder of a partially filled batch moves to the next batch. A unit that fits model input but cannot fit alone with its metadata/cursor in the tool-result envelope becomes unresolved `review_record_too_large_for_tool_result`, with the original intact. Do not truncate its meaning or switch models. Finalize cursors only for delivered units or explicit host-recorded unresolved gaps; packing cannot skip an undisclosed claim or return a nonterminal empty page at the same cursor. Source pages use both the smaller actual evidence budget and the character limit. Estimate the full next provider request again after final continuation preparation; callbacks/tool results/history reduction can change available space. Unknown/irreducible request capacity is a request-level gap, not evidence that every remaining claim is individually too large.

- [ ] **4c. Test model capacity and transport capacity independently.** With run logs disabled, parameterize native/fenced delivery, a generous model window and `max_tool_result_chars=1024`. Use several individually fitting claims whose combined page exceeds that character budget, then one otherwise model-admissible claim that cannot fit alone. Inspect actual post-append/provider payloads: every successful page parses whole with intact metadata/cursor, all fitting claims are delivered exactly once, and the oversized unit expands to an explicit unresolved original. Include the existing native 16,000-character cap when the configured ordinary cap is zero or larger, JSON escaping/multibyte text, and repeated cursor reads; do not mock the generic truncator or increase its limits.
- [ ] **5. Commit progress outside model history before reduction.** Submission validates live verifier/attempt/unit/evidence and that attempt's submission revision, then atomically stores dispositions, evidence and cursor before acknowledgment. Repeated cursor returns a stable page; replayed identical submission returns its prior ack; conflicting/revised verdict requires a new explicit verification attempt. The history projector may replace completed rounds only after that checkpoint is accepted. Once a verifier settles, finalization marks every missing unit unresolved `verification_incomplete`; it does not spawn another verifier or grant fresh batches as new runs. Empty original sets finish with explicit zero-finding coverage.
- [ ] **6. Expand and mechanically validate every original.** Require one disposition per public unit, no invented IDs, valid duplicate references and no verdict transfer between nonidentical units. Expand exact-repeat units through the private map; require every original exactly once, preserving original attribution. Missing evidence leaves unresolved. Non-reproduction alone is not refutation; include this in the verification recipe and behavioral qualification rather than pretending a schema can prove semantic truth.
- [ ] **7. Run GREEN:** all three new verification/context/history files plus `Tests/Agents/test_run_log_eviction.py` and relevant existing project-instruction/provider-continuation tests located from `rg --files Tests/Agents Tests/Chat | rg '(project_instruction|provider_continuation)'`. Select tests exercising modified request-builder methods. Assert actual serialized request cost, low/unknown capacity, genuinely oversized units, multi-batch real-child reduction with run logs disabled, interrupted last batch, duplicate cursor, irreducible native continuation and post-return submission refusal. Preserve the existing run-log feature's primary-only behavior.
- [ ] **8. Review and commit:** `feat(agents): verify blinded claims in bounded checkpointed batches`.

## Task 9: Authorized disposable checks and user-readable reports

**Files:** Extend `Chat/console_code_review.py`, service/provider/store/verification and `Tests/Agents/code_review_helpers.py`. Modify `Agents/agent_service.py` at `wait_agents` and actual tool preparation/settlement, plus existing fleet wait tests. Create `Widgets/Console/console_code_review_modal.py`, `Tests/Chat/test_console_code_review_checks.py`, `Tests/UI/test_console_code_review_reports.py`. Modify `UI/Screens/chat_screen.py` report action wiring and its source stylesheet only as needed. Reuse `ConsoleSaveMarkdownModal` and existing authorized destination tools for explicit export.

**Interfaces:** Add primary-only `code_review_submit(kind="check_actions")` accepting one `CheckActionRecord` per call, and enable verifier `check_proposals` with at most one outstanding proposal. Produce host-only `claim_check(review_id: str, proposal_id: str) -> ReviewCheckClaim` and `register_check_result(review_id: str, proposal_id: str, execution_receipt_id: str) -> CheckReceipt`. The submit adapter invokes `claim_check` after validating the primary's action; only the actual execution owner can register a native result. Produce `pending_check_for_wait(primary_run_id: str, handle_ids: tuple[str, ...]) -> ReviewCheckNotice | None`, where the bounded notice carries only review/proposal/child references, and `await_check_result(review_id: str, proposal_id: str) -> CheckReceipt` under the bound verifier's remaining deadline/cancellation context. The latter is reached through that verifier's `code_review_read(view="verification_batch", document_ref=proposal_id)`, not a new public tool or a caller-controlled wait duration. Primary proposal details/claim specification use `code_review_read(view="summary", document_ref=proposal_id)`; a reviewer cannot read either view. Both modes use the complete-response packer; result evidence is paged without clipping the receipt/ref/cursor envelope.

Add host-only store association `bind_check_actor(primary_run_id: str, review_id: str, verifier_attempt_id: str) -> None`. Check-action publication uses its own stream `(primary_run_id, review_id, verifier_attempt_id)`, derived from the live host actor rather than JSON, with the same CAS/idempotency rules as participant publication. Primary summary readback exposes that actor's `submission_revision`; check actions/host results cannot advance verifier/reviewer revisions. Reject reassociation or actions on another review/attempt. These associations admit no reviewer/verifier impersonation.

Produce `render_original_report(review_id: str, slot_id: str) -> str`, `render_unified_report(review_id: str) -> str`; deterministic renderers consume accepted records/receipts and never change findings. Modal constructor accepts the owning service and opaque review ID, not an unrestricted path.

- [ ] **1. Write RED authority/lifecycle tests.** A suggested command without an execution receipt remains proposed. Existing permission denial is preserved. A check must run against a disposable captured-source location charged to the same review, not the user checkout; generated/staging files count. Real results reach the still-live verifier through its bounded evidence read before a final disposition. After verifier return, a new check cannot rewrite its verdict without an explicit new verification attempt.

```python
def test_finished_verifier_cannot_accept_late_decision(review_harness):
    review_id = review_harness.prepare(verify=True)
    review_harness.finish_verifier(review_id)
    result = review_harness.invoke(
        role="verifier", tool="code_review_submit", review_id=review_id,
        arguments={"review_id": review_id, "kind": "dispositions", "records": [],
                   "expected_revision": 0, "submission_id": "late"},
    )
    assert result.ok is False
    assert "verification_attempt_closed" in result.error
```

For this test the harness starts and registers a live verifier using the real scoped launch before calling `finish_verifier`; it never manufactures a terminal verifier association.

- [ ] **1a. Reproduce the waiting-parent case at the real fleet seam.** Add test helper `run_check_handoff(*, outcome: str, primary_waiting: bool = True, duplicate_actions: bool = False) -> ReviewRunObservation`. Its script makes the primary enter actual `wait_agents`, then lets the live verifier submit a proposal and await its result. Observe `wait_yielded_while_verifier_live`, `execution_launch_count`, `result_seen_by_same_verifier`, actual accepted proposal/claim/native-receipt lifecycle events and terminal mappings from host owners, not script expectations. `check_events` excludes replayed calls that cause no state transition. Use the real wait/service/store with a bounded fault-controllable existing-executor adapter at the external process boundary; never mock the wait or claim method being asserted.

```python
def test_check_request_yields_wait_before_verifier_finishes(review_harness):
    observed = review_harness.run_check_handoff(
        outcome="executed", primary_waiting=True, duplicate_actions=True,
    )
    assert observed.wait_yielded_while_verifier_live
    assert observed.execution_launch_count == 1
    assert observed.result_seen_by_same_verifier
    assert observed.check_events == [
        "proposal", "wait_yield", "claim", "execution_accepted",
        "receipt", "verifier_read", "disposition",
    ]
```

Repeat for proposals arriving just before wait entry and while it polls. Test explicit decline, permission denial, unavailable quota-enforcing executor, primary ending/cancelling, verifier ending/cancelling, source/close fences and deadline expiry. Include a runtime limited to one tool worker: actual capacity refusal must resolve/signal the pending check and release the waiter, with zero process launches and no model retry loop. Assert an honest unresolved reason, no extra child/automatic turn and unchanged ordinary fleet waits. Reject forged/cross-attempt/target-mismatched receipts; race repeated claims/tool calls and late callbacks, including an accepted execution whose outcome is uncertain. Check same-stream action replay/conflict independently of concurrent verifier dispositions/raw capture.

- [ ] **1b. Test the synchronous verifier without a fleet.** Add `run_inline_check(*, max_live_at_prepare: int, max_live_at_launch: int) -> ReviewRunObservation` using the actual service/coordinator factory and scripted child replies, not an always-injected fleet. Observe actual launch feedback mode, `host_check_wait_count`, `primary_check_notice_count`, `execution_launch_count`, source-based dispositions and unresolved reasons. With both values 1, let the inline verifier submit a proposal then read its result: assert immediate host `unavailable`/`check_feedback_unavailable_inline`, zero condition waits/notices/process launches, and successful return from synchronous spawn without spending the child deadline. Cover prepare 3 -> launch 1 and prepare 1 -> launch 3 when the later primary turn constructs its executor under the new setting. Also change settings within an already-created turn and after launch: the actual executor/branch, rather than the latest setting alone, determines the bound mode. Keep ordinary inline spawning intact and retain the exact-two-child continuation matrix.
- [ ] **1c. Test bounded check-result delivery.** With a small result-character limit, deliver multi-page host evidence to the same live verifier through actual native/fenced calls. Each page retains the correct proposal/receipt ref, target digest and cursor; if the minimal receipt cannot fit, preserve the host execution fact and finish the dependent claim unresolved with a transport reason. Proposed/denied/unavailable results must also fit whole and never appear as truncated executed success. Reuse existing store quotas and the Task 4 response packer.

- [ ] **2. Run RED:** `.venv/bin/python -m pytest Tests/Chat/test_console_code_review_checks.py Tests/UI/test_console_code_review_reports.py -q`.
- [ ] **3. Implement a bounded request/claim/result handshake.** A verifier proposal is an idempotent own-stream record bound to its opaque units, live attempt and captured target. Allow one outstanding proposal per verifier; it is neither execution permission nor an executed receipt. Use the host-bound actual launch mode from Task 5. With `unavailable_inline`, atomically publish the accepted proposal and host `unavailable` result with reason `check_feedback_unavailable_inline` before acknowledging it; a proposal-specific result read returns immediately. Create no pending parent notice, condition wait or execution claim. The verifier may complete source-based decisions; claims requiring that execution stay unresolved. Do not enable a fleet, restart the verifier or defer execution until after its return. Preparation diagnostics are advisory; actual launch mode remains authoritative despite later settings changes.

With `primary_handoff`, inspect the host predicate in `wait_agents` before blocking and on each existing poll, alongside stop/deadline fences. For a pending proposal belonging to a waited review verifier, return `check_requested` immediately, retaining that live handle and any ordinary terminal entries without collecting or marking the verifier finished. The return resumes the same primary turn; it consumes no boundary receipt or new wake allowance. Ordinary waits retain their current behavior.

The primary reads the proposal, then submits a claim or decline. Atomically transition `pending -> claimed -> running -> resolved`, allowing a terminal decline/unavailable/cancelled/expired result before execution. A claim returns a host-generated specification for one existing process tool, normalized arguments and admitted disposable source location. It does not invoke the tool or bypass approval. At actual tool preparation, bind only the claiming primary's matching call to the private claim; reject a changed specification/target or duplicate acceptance. Existing tool/permission owners perform authorization and dispatch. Identical claim/action replays return the existing claim/result without launching again; accepted or uncertain execution cannot be automatically retried or declined as if it never started. Unrelated process receipts and model-written “passed” strings cannot satisfy a claim.

- [ ] **3a. Preserve execution and storage authority.** Reserve captured-source copy, generated-file and output headroom before execution and require an existing executor whose actual write boundary enforces that reservation; a directory scan after execution or unbounded shell redirection is not a storage bound. If no such executor is available, resolve the proposal `unavailable`, return its required next check and leave the dependent unit unresolved; do not add a sandbox/execution engine. Retain native process/file leases and charges until actual exit and confirmed cleanup, including cancellation or uncertain outcomes. Charge bounded stdout/stderr to report limits. Only the actual execution owner registers the accepted receipt, exact captured digest and observed outcome; a nonzero exit or non-reproduction alone does not establish a verdict.

- [ ] **3b. Return host evidence to the same verifier without model polling.** Its proposal-specific `verification_batch` read waits on a host condition and rechecks live authority, close/stop and the smaller remaining verifier wall deadline/real monotonic deadline. Publishing a result or any terminal fence signals the waiter; the result includes no reviewer attribution or private mapping. Use the existing `ToolExecutionPolicy.DEFINITIVE_AFTER_START` for `code_review_read` once this waiting form is enabled, so the generic abandonable per-call wrapper cannot strand a live waiter. Because policy is selected by tool name, every read form must retain its own finite I/O/deadline/cancellation bounds and leases; no read may wait indefinitely. Preserve normal tool/call/wall accounting. Pin an active wait/tool-result pair during Task 8 history reduction; after a result is acknowledged its bounded reference may enter the checkpoint.

The result wait retains its normally admitted tool-operation charge. Wire actual tool-admission refusals for the bound verifier's result read or its owning primary's proposal/action/process calls to the host check state: resolve unstarted work `unavailable` and signal before returning the refusal. This includes `tool_capacity`/`automatic_tool_capacity`; do not raise global limits, bypass reservation or leave the only worker waiting after a primary capacity refusal. Accepted running execution remains under its real owner and cannot be relabelled unstarted by another refusal.

If the primary ends, declines, cannot execute or exhausts capacity, resolve pending work with the corresponding reason and signal the verifier. Wire primary finalization before fleet draining, so settlement itself cannot wait on an unsignalled verifier. If the verifier ends, cancel unstarted claims and finalize dependent units unresolved. A running process keeps its native owner/leases; late settlement can update its receipt and cleanup accounting but cannot reopen a closed verifier, rewrite a disposition or start another execution. No role termination, timeout or failed check grants a replacement child, primary turn or budget. Evidence-independent dispositions may proceed; dependent units need a resolved receipt or an explicit unresolved disposition, never assumed success.

- [ ] **4. Make originals separately readable before the unified view.** Show actual requested/active routes and fallback switches, coverage, incomplete slots, record originals and raw-output tabs/pages. Source/report text is sanitized and cannot trigger tool execution. Page bounded content off-loop; modal navigation does not enqueue agent work. Prove the native reader/export can access a retained original that the model's smaller transport cap cannot carry, without changing its authority or the stored record. The unified renderer requires the complete host mapping, displays confirmed issues ordered by impact, unresolved next checks, refutations/evidence, every original→unit/canonical disposition, actual checks and target identity. A newly discovered issue is explicitly separate from original claims.
- [ ] **5. Add explicit lifecycle actions.** Offer verification only when not already authorized; an existing grant proceeds after original report presentation. Export/keep uses a user-selected admitted destination and the current permission path. Release requires a trusted explicit discard or completed export+release intent; a model boolean is insufficient. Refuse active leases; display cleanup-pending/failure and expired references honestly. Transcript report links identify session retention, not durable full copies; opening a saved chat after bundle expiry cannot recover raw output.
- [ ] **6. Run GREEN:** new tests, `Tests/Agents/test_fleet_runtime.py`, `test_fleet_reliability.py`, `Tests/UI/test_design_token_governance.py`, `Tests/UI/test_css_bundle_sync_guard.py` and relevant existing Console save/export tests. Rebuild CSS with `.venv/bin/python tldw_chatbook/css/build_css.py` if styles changed. Pilot narrow terminals, focus/keyboard navigation, literal markup, long report pagination, partial/cancelled states, export denial and close while the modal holds a read lease. Do not add terminal-convention keybindings.
- [ ] **7. Review and commit:** `feat(console): present original and verified review evidence`.

## Task 10: Saved team UI, packaged recipes and optional inline skill

**Files:** Modify `Widgets/settings_agents_panel.py`, its existing source CSS only if needed, `pyproject.toml`, `MANIFEST.in`, `Packaging/check_manifest.py`. Create both `Agents/code_review_guides/*.md`, `Docs/Examples/skills/multi-model-review/SKILL.md`, `Tests/Packaging/test_code_review_guide_manifest.py`, `Tests/Widgets/test_settings_code_review_team.py`. Extend `Tests/Agents/test_code_review_tool_provider.py` and existing settings off-loop tests.

**Interfaces:** Produce `load_code_review_guide(stage: Literal["review", "verify"], team: ReviewTeam, *, max_result_chars: int = 16000) -> dict`, returning a versioned ≤12 KiB envelope also within the smaller trusted runtime-derived character budget, with admitted team diagnostics; no endpoint URLs/credentials. The provider always supplies that live limit. Settings uses the shared `parse_review_team`, real preset IDs and existing config persistence. Saving preferences never mutates preset provider/model fields; per-run overrides never call the settings save path.

- [ ] **1. Write RED tests.** Rename and disable/delete a preset while a saved team references it; preserve order/ID and show missing diagnostics. Reorder/add/remove rows, edit independent focuses, save/load verifier and cancel edits. Capture a review, change saved config, and assert captured slots remain unchanged. Test the guide's exact byte limit including large/multibyte metadata and a smaller runtime character cap; overflow returns a bounded diagnostic rather than truncated instructions, with no source capture or child/model launch.

```python
@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_review_guides_are_required_in_distribution(kind):
    required = getattr(check_manifest, f"REQUIRED_{kind.upper()}_PATHS")
    for name in ("review.md", "verify.md"):
        assert f"tldw_chatbook/Agents/code_review_guides/{name}" in required
```

Also build and inspect an actual wheel/sdist; the manifest-only assertion is an early guard, not installed-artifact evidence.

- [ ] **2. Run RED:** `.venv/bin/python -m pytest Tests/Widgets/test_settings_code_review_team.py Tests/Packaging/test_code_review_guide_manifest.py Tests/Agents/test_code_review_tool_provider.py -q`.
- [ ] **3. Add compact team controls to canonical Agents settings.** Reuse preset selection/form/status components and existing async/off-loop reads; values are stable IDs, labels current names. One ordered reviewer list with focus per row and one verifier selector is sufficient. Empty defaults explain setup; missing/disabled/duplicate destinations are visible. No new screen, workspace layer, team registry or model catalog. Read feature config alongside current routing controls without writing on form load.
- [ ] **4. Write concrete recipes.** Review guide sequence: prepare target/team and any explicit run-only destination overrides, explain capture/route gaps and mutable `context_paths`, use bounded source list/search/resolve and direct `source_range` reads to inspect relevant callers/tests, launch admitted scoped slots, wait for actual terminal receipts, present each original with attribution/coverage, offer verification. Explain instruction deferrals and complete-page transport refusals without asking the agent to bypass them through raw paths/history. Describe obtaining the caller's `submission_revision` and the single unchanged-payload conflict retry; another participant's append is not a conflict. Verification guide sequence: require stage grant, present originals first, launch one fresh configured verifier, use opaque batches/evidence, checkpoint dispositions before host history reduction, leave insufficient evidence unresolved, validate expanded coverage and present a new unified report. For threaded executable checks, instruct the verifier to submit one proposal and await its proposal-specific result read; instruct the primary to handle `check_requested` from its wait, read and claim/decline the proposal, use the existing authorized process tool only with the claimed specification, then resume waiting for actual verifier settlement. With an inline verifier, use synchronous spawn receipts and source-based verification; executable checks return unavailable immediately, without waiting/repeated proposals or enabling a fleet. A check-request notice is not child completion; absent execution authority produces unresolved evidence. Include the exact two-child continuation matrix and commands/schemas; forbid primary-supplied suspected findings, consensus-as-proof, automatic fixes and silent launch retries. The verifier task includes the evidence standards, not reviewer identities or severity votes.
- [ ] **5. Package explicit guide paths and prove installed access.** Add `"tldw_chatbook.Agents" = ["code_review_guides/review.md", "code_review_guides/verify.md"]` to package data and explicit sdist entries; add these exact paths to packaging guards and narrowly allow their Markdown. Load with `importlib.resources`, not a checkout-relative path. Build using the repository's existing packaging verification commands and load both resources from a temporary installed wheel outside the source tree. The optional skill has `context: inline`, `user_invocable: true`, `disable_model_invocation: true` and calls the same guide under normal skill trust rules; it is not auto-installed or model-advertised as a child skill.
- [ ] **6. Run GREEN:** the new tests plus `Tests/UI/test_settings_agents_category.py`, `test_settings_agents_offloop_read.py`, `Tests/Widgets/test_settings_agents_panel_routing.py`, `Tests/Packaging/test_canvas_guide_manifest.py`, and skill metadata/trust tests covering the optional example. Pilot settings at the repository's supported narrow width and keyboard-only usage.
- [ ] **7. Review and commit:** `feat(console): configure review teams and ship primary recipes`.

## Task 11: End-to-end qualification, docs and evidence-based closeout

**Files:** Create `Tests/Chat/test_console_multi_model_review_journey.py` and a small qualification fixture/evidence directory under `Docs/superpowers/qa/2026-10-18-multi-model-code-review/`. Modify `Docs/Architecture/agent-runtime.md`, `tool-catalog.md`, `console-file-authority.md`, `Docs/User_Guide/console/agent-runs-and-tools.md`, the spec, ADR-281 and TASK-35582. Update related architecture/user-guide indexes only where a new document is added. Preserve unrelated working-tree changes.

**Interfaces:** The complete Console entry path is the interface under test. User-visible outcomes must match all 25 acceptance criteria; there is no new production API in this task. The qualification manifest records checkout, config fingerprints without secrets, actual destinations/fallbacks, source digest, run/attempt IDs, counts/budgets, observations, unresolved gaps and evidence paths.

- [ ] **1. Write a scripted RED journey through actual Console composition.** No installed skill; natural-language user request; two configured reviewers; structured report submission; first-stage presentation; explicit later acceptance; one fresh verifier; two exact-repeat originals, one changed-trigger claim, a false positive, confirmed/refuted/unresolved results; complete expansion. Test pre-authorized verification under the two-child limit as a second journey. Assert no caller can use source/report text as stage authority.

```python
def test_pre_authorized_journey_keeps_originals_before_fresh_verifier(review_harness):
    observed = review_harness.run_script(reviewers=2, verify=True)
    assert observed.launches_per_primary_turn == [2, 1]
    assert observed.original_reports_presented_before_verifier
    assert len(set(observed.chain_ids)) == 1
    assert observed.verifier_run_id not in observed.reviewer_run_ids
    assert observed.original_ids == set(observed.final_dispositions)
```

`ReviewRunObservation` gains `verifier_run_id`, `reviewer_run_ids`, `original_ids` and `final_dispositions`, all read from actual host receipts/ledger and presentation events rather than copied from the script.

- [ ] **2. Run the journey plus the targeted regression matrix below.** Fix failures at their owning seams. Do not widen to an unrequested full suite. Include an actual per-run reviewer and verifier route override with saved config unchanged, simultaneous report submissions/raw capture, unchanged committed caller discovery with a deep search-hit/range read and nested instruction delivery, explicit mutable-context gaps, a check request yielding an already-waiting primary and returning its receipt to the same verifier, inline source verification with immediate unavailable checks, launch-mode changes across turns, complete native/fenced pages under narrow character caps, real-child history reduction with run logs off, and prepared-abort/reclaim versus accepted replay denial. Use deterministic provider fault controls for failed reviewer, truncated fleet result, capture drift, unknown/small context, storage overflow, concurrent reservations, lease cleanup failure, disabled autowake and uncertain accepted attempt. Keep the source checkout unchanged by reviewer/check execution.
- [ ] **3. Perform a bounded real Console/provider qualification when implementation is authorized.** Use a small temporary repository with a seeded introduced bug and an explicit false-positive claim/control, two already-configured reviewer destinations and one saved verifier preset. Run one pre-authorized request, preserving `max_subagents=2`, with at most three review/verifier launches and at most 20 provider calls across the qualification. Apply any narrower configured limits; do not reset budgets or invent retries to finish the demonstration. Record actual provider/model/sampling/fallback dispatch and stage ordering, useful defect analysis, uncertainty and complete mapping. If credentials/permission/capacity are absent, record AC-24 as unverified and keep it unchecked.
- [ ] **4. Separate live observations from scripted controls.** Include seeded defect, false positive, exact duplicate, distinct trigger, failed reviewer, truncation, drift, small context and storage overflow in the qualification report; clearly identify which used real model output and which used scripted gateway/fault injection. Valid JSON or a provider-success status alone does not demonstrate a useful review. Never fabricate consensus or claim a false positive was refuted without specific evidence.
- [ ] **5. Update implemented documentation and ADR links.** Describe setup, target inputs/comparison, permissions, original/verified report semantics, actual destination attribution, partials, limits, continuation/manual resume, session expiry/export/release and cancellation. Document the implemented schema migration and typed review-boundary eligibility in ADR-281 and link the amendment from ADR-134/135 without rewriting their historical decisions. The optional skill examples must match installed tool names and record schemas. Do not describe planned APIs as shipped until their tests and wiring are present.
- [ ] **6. Run scoped static checks and self-review.** Use the repository's configured formatter/linter on changed Python files; compare pre-existing debt rather than formatting whole large owners. Run `git diff --check`, packaging guards and the relevant CSS checks. Review actor/role entry points, raw output destinations, every automatic acceptance path, late callbacks, quota release and final disposition completeness. Check no new secrets/body projections, global budget changes or review recovery promises slipped in.
- [ ] **7. Record evidence and finish task hygiene only after all ACs pass.** Add the task's Implementation Notes with approach, changed files, ADR/migration and concrete test/live evidence. Check only supported criteria. If live qualification or a required acceptance remains blocked, keep the task In Progress and state the gap. Only after the Definition of Done is met use the Backlog CLI to mark Done. Suggested final slice commit: `test(console): qualify multi-model review and document limits`.

## Targeted regression matrix

Run the named new tests in each task when that slice is implemented. The following existing suites protect shared boundaries; select only the entries reached by changed code and avoid rerunning completed expensive checks without a new change or unresolved concern.

| Boundary | Existing targeted evidence |
| --- | --- |
| Preset routing, actual gateway and fallback | `Tests/Agents/test_agent_routing.py`, `test_agent_routing_integration.py`; named-model provider-boundary control |
| Generic/concurrent fleet and cancellation | `Tests/Agents/test_fleet_runtime.py`, `test_fleet_continuation.py`, `test_fleet_reliability.py`; `Tests/Chat/test_console_close_session_fleet.py` |
| Binding/sensitive source/patch parsing | `Tests/Tools/test_git_tool_impls.py`, `test_git_tool_sensitive_paths.py`; existing patch-tool tests; `Tests/Chat/test_console_agent_project_instructions.py` |
| Session lease ownership | `Tests/Chat/test_console_scratch_space.py`; new close/publication/cleanup races |
| Ledger migration, claims and finite allowance | `Tests/Agents/test_agent_runs_wake_ledger.py`; `Tests/Chat/test_automatic_work_lineage.py`, `test_automatic_wake_budget.py`, `test_automatic_wake_recovery.py` |
| Native wake acceptance/priority/capacity | `Tests/Chat/test_console_fleet_wake.py`, `test_automatic_wake_dispatch.py`, `test_automatic_wake_scheduling.py`, `test_fleet_runtime_admission.py` |
| Prepared review-claim abort/reclaim versus consumed replay | New `Tests/Agents/test_code_review_boundary_ledger.py` and native continuation races; inspect link rows and spent reservations |
| Verifier child history with run logs disabled | New `Tests/Agents/test_code_review_history.py`, `test_code_review_context.py`; existing `test_run_log_eviction.py` and provider-continuation ownership controls |
| Concurrent participant report publication | New `Tests/Agents/test_code_review_store.py`, `test_code_review_dispatch.py`, `test_code_review_reports.py`; real threaded submissions interleaved with raw capture |
| Check request during a primary fleet wait | New `Tests/Chat/test_console_code_review_checks.py` plus existing fleet wait/cancellation tests; real wait yields without collection, one accepted execution, bounded result wait, unavailable/cancel/late-result controls |
| Direct bounded source line selection | New `Tests/Agents/test_code_review_source.py` and `test_code_review_tool_provider.py`; deep search hit followed by one public range read, bounded pagination and cursor-conflict refusals |
| Inline source verification and actual launch-mode binding | New dispatch/check/journey tests plus `Tests/Agents/test_fleet_runtime.py` inline controls; no injected fleet, immediate unavailable check result, settings changes across and within turns |
| Complete model-bound response envelopes | New source/provider/report/context/check tests; actual native/fenced append with run logs off, small `max_tool_result_chars`, native ceiling, intact cursors/revisions and explicit indivisible-record refusals |
| Captured refs activating nested guidance | New provider/dispatch tests with real `InstructionActivationLedger`; existing `Tests/Agents/test_project_instruction_runtime.py` and `Tests/Chat/test_console_agent_project_instructions.py`; guidance before source, per-chain delivery, exclusions and consent preserved |
| Live/preview tools and role capabilities | `Tests/Chat/test_console_agent_bridge.py`, `test_console_agent_bridge_local.py`; new direct guessed-call isolation controls |
| Settings/off-loop layout and tokens | `Tests/UI/test_settings_agents_category.py`, `test_settings_agents_offloop_read.py`, `test_design_token_governance.py`, `test_css_bundle_sync_guard.py` |
| Runtime guides in distributions | `Tests/Packaging/test_canvas_guide_manifest.py`; new guide manifest and actual installed-wheel probes |

## Acceptance coverage and review gates

| TASK-35582 AC | Deliverable / decisive evidence |
| --- | --- |
| 1 | Tasks 4, 5, 10, 11: primary guide discovery, optional inline alias, actual Console journey with no installed skill |
| 2 | Tasks 1, 4, 5, 10: stable-ID config, settings round trip, prepared run overrides at actual gateway, no automatic paid defaults |
| 3 | Task 5: equal shared brief/source, supplemental focuses, peer and personal-context canaries absent |
| 4 | Tasks 3/4: all four targets, frozen list/search/resolve and unchanged caller discovery, direct bounded line-range reads, explicit mutable-context gaps, exact bytes/revisions and scope controls |
| 5 | Tasks 3–5, 7: real source/routing/consent/permission gates, captured-ref instruction mapping with nested guidance before source delivery, and missing/duplicate destinations |
| 6 | Tasks 2, 4, 5, 7: role/run/source/grant binding, mutually exclusive spawn and accepted-launch idempotency |
| 7 | Tasks 1, 2, 4, 5: strict v1 byte/chunk/reference/local-key and per-attempt revision checks, simultaneous accepted appends and preserved chunks |
| 8 | Task 6: terminal-plus-coverage sealing and explicit failed/malformed/empty distinctions |
| 9 | Tasks 5, 6, 9: actual destination/fallback receipts, original/raw readback before fleet caps |
| 10 | Tasks 6, 8: complete paged host index, no prose reconstruction/raw concatenation |
| 11 | Tasks 5, 7, 11: native stage grants and a fresh saved-preset verifier without repeated approval |
| 12 | Task 8: random unrelated units, private map, byte-identical coalescing, projection leak tests |
| 13 | Tasks 8, 10, 11: distinct triggers remain distinct; individual evidence standards and minority claims |
| 14 | Tasks 8, 9, 11: complete unit/original expansion and valid canonical references, honest partials |
| 15 | Tasks 5/9: actual launch-mode binding, immediate unavailable inline checks, threaded request yielding the primary wait, primary-only claim/decline, one authorized disposable execution, whole native receipt pages and bounded same-verifier feedback, cancellation/late-result controls |
| 16 | Tasks 1, 4, 7: actual two/three-reviewer launch matrix and prior-spawn controls |
| 17 | Task 7: exact boundary acceptance, same chain, prepared-abort/reclaim, consumed replay denial, uncertain recovery and manual/stop/close fences |
| 18 | Tasks 1/3/4/8/9: exact capacity estimation after child history reduction, 8/4096/1024 token/unit limits plus whole-response character limits, native/fenced delivery, unknown/oversized/irreducible protocol controls |
| 19 | Tasks 2, 8: stream-local dispositions/cursors, real-child checkpoint-backed history reduction with run logs off, interruption and no per-batch budget reset |
| 20 | Tasks 1–3, 6, 9: all exact byte/file/context caps, staging and physical copies included |
| 21 | Tasks 2, 6, 9: concurrent reservations, overflow/disk errors, explicit release, cleanup-aware accounting |
| 22 | Tasks 3–5, 7–9: actual runtime capability denial, role/ref/chat/epoch/expiry isolation |
| 23 | Tasks 1–11: targeted deterministic tests at the actual changed boundaries |
| 24 | Task 11: bounded real Console/provider qualification plus separately attributed fault controls |
| 25 | Tasks 7, 10, 11: existing-owner migration, packaged skill/docs, ADR amendment and scope audit |

Review each slice after its GREEN checks, then make a scoped commit if execution is authorized. These are implementation slices inside TASK-35582, not newly created backlog tasks or a requirement for 11 separate PRs. Keep incomplete feature wiring gated until the full path satisfies its boundaries; do not label an early foundation slice as a user-ready review feature.

## Implementation-plan review resolutions

These are revised planning requirements and tests, not implemented or passing behavior.

| Review finding | Resolution and implementation gate |
| --- | --- |
| 1. Verifier history reduction assumed unavailable primary-only eviction | Task 8 adds a verifier-only checkpoint-backed request projector before capacity admission, pins live protocol state, refuses irreducible groups and tests a real child with run logs disabled. |
| 2. No input path for per-run provider/model overrides | Tasks 1/4/5 separate saved and run choices, admit reviewer/verifier overrides through ADR-147 at preparation, freeze them into slots, and check actual gateway dispatch plus unchanged saved config. |
| 3. Concurrent reviewers invalidate shared-revision submissions | Tasks 2/4–6 use slot-attempt submission streams and separate context/publication revisions; raw capture cannot consume model revisions. Barrier tests exercise both direct storage and real child tool calls. |
| 4. Frozen-source discovery missing | Tasks 1/3/4 define strict source list/literal-search/relative-path-resolution queries, pinned committed context and pre-captured mutable context, with exclusion/cursor/quota/gap tests. |
| 5. Ledger tests depend on a later coordinator | Task 7 combines native authority, ledger and delivery into one gate: build/test the real issuer, use its receipts in ledger tests, then wire the native launch matrix. Store fixtures remain independent of later service/native issuance. |
| 6. Wake boundary claims lack abort/completion lifecycle | Task 7 extends claim/read/accept/abort/completion/recovery and pending/fence bookkeeping together. Only proven prepared abort removes the active claim; accepted/completed/uncertain links remain non-replayable. |

The subsequent review identified three additional issues, resolved at these gates:

| Remaining finding | Resolution and implementation gate |
| --- | --- |
| A. Primary and verifier can wait on one another for a check | Task 9 makes review-bound `wait_agents` yield a pending request, adds a separately revisioned primary claim/decline stream and one native execution claim, and returns host evidence through a deadline/cancellation-bound read to the same live verifier. Tests cover already-waiting parents, exact event order, replay, unavailable execution, finalization before fleet draining and late results. |
| B. Early GREEN checks require the later raw-capture tap | Tasks 2/5 test independent concurrent submissions using their existing owners. Task 6 introduces raw capture and extends both scenarios with actual interleaving; earlier gates require no later code or capture stub. |
| C. Search hits cannot be read directly by line range | Tasks 1/3/4 add strict `SourceRange` and the public `source_range` argument, bounded selected-range pagination and cursor identity checks. Task 3 tests the source owner; Task 4 proves a deep search hit can be read directly through the real public tool. |

The latest follow-up review identified three further gaps, resolved as follows:

| Follow-up finding | Resolution and implementation gate |
| --- | --- |
| D. Inline verifier cannot yield the synchronous parent through `wait_agents` | Task 5 binds actual threaded/inline feedback mode at launch; Task 9 atomically resolves inline check proposals unavailable before any wait. Real no-fleet tests preserve source-based verification, synchronous receipts, launch budgets and settings-change behavior. |
| E. Token-sized pages can still be clipped by the runtime character cap | Tasks 1/3/4 define a trusted complete-response budget and pack full envelopes before delivery/mutation. Tasks 5/6/8/9 test actual native/fenced delivery, cursor progress, narrow result caps, the native ceiling and explicit indivisible-record gaps; Task 10 applies both guide byte and character limits. |
| F. Opaque source refs do not participate in nested instruction activation | Task 4 implements validated `PathAwareToolProvider` mapping to original admitted project paths and tests the real ledger; Task 5 proves actual per-chain guidance precedes child source evidence while preserving exclusions, consent and permission review. |

## Execution entry and stop conditions

When implementation is explicitly authorized: inspect current worktree changes, read TASK-35582 and this plan/spec/ADR, move the task to In Progress through the CLI, add the Implementation Plan link and ADR check, then execute sequentially. Respect current repository instructions if they have changed. Preserve all unrelated files and stage only scoped changes.

Stop the affected operation, retain accepted progress and surface its stable reason when source authority/capture, destination consent, actual capacity, quotas, native acceptance or finite causal allowance cannot be established. Do not weaken a boundary to make a demonstration pass. If implementation research requires changing an approved contract, update the spec/ADR/task criteria first and record the reason; routine placement details remain the implementer's judgment.

Execution is authorized. Implement the approved contracts with targeted verification and scoped commits; any live provider qualification must retain the bounded plan limits and existing provider/permission authority. Mark acceptance only against recorded evidence.


### Slice 7 native integration amendment

ADR required: yes. ADR path: `backlog/decisions/281-primary-agent-multi-model-code-review.md`
(native acceptance and continuation amendment above). The host registers actual USER
stage intent first, then narrows an exact grant during native prepare admission.
Existing local tool permission and per-target project-dispatch consent owners remain
mandatory. Actual later primary associations preserve immutable originals and use
current manual admission; automatic turns preserve the original finite chain.

Slice 7's Console matrix uses native per-reviewer transcript summaries with opaque
original references and sealed zero-finding reports. This proves actual presentation
before dispatch, without a future batching dependency. Task 9 must provide full
original dereferencing in its native review UI, and Task 11 must qualify that journey.
The schema25 link extends the existing wake lifecycle including prepared-abort
reclaim, accepted replay denial and review-required recovery; no review-job DB exists.
The implementation/evidence handoff is `.superpowers/sdd/2026-10-18-multi-model-code-review-implementation/task-7-report.md`.


Task 7 review corrections: verification authority accepts only a complete,
unqualified native stage clause; conditional, negated or unknown qualifications
leave verification ungranted until a later accepted instruction or exact native
offer acceptance. `Retry reviewer R1` binds R1 from the accepted USER message;
caller/tool slot proposals cannot substitute another slot. `Resume review` binds
only never-started reviewer slots in the single retained review to fresh current
manual admission, preserving the original capture, frozen routes, earlier streams
and terminal receipts. It revalidates destination consent and current child source
permission and assigns fresh attempts; it does not carry old verification intent
into a reviewer-only request. Ambiguous reviewer/review selection fails explicitly.
Old live wake eligibility is superseded and removed from the existing wake owner
outside review locks; stale callbacks cannot overwrite the new revision's status.
Prepared ownership is released only by the existing proven-abort lifecycle;
accepted/uncertain evidence is never refunded or replayed. A still-accepted missing
slot blocks manual resume until its outcome is known. The old automatic chain and
its used/reserved budget remain unchanged.

Later manual-stage grants record accepted stage intent against the retained context;
they are composed before the new primary's local permission callback is installed.
They never substitute for file permission: native child claims and source
read/materialization fail closed if that current callback is missing or denies.
The initial preparation continues to authorize source access before capture and
exact preparation-grant issuance. Fixture-only direct issuer/preparation helpers
remain distinct from the actual Console admission evidence.


Task 7 acceptance-race correction: the registered native boundary authority also
supplies an independent acceptance scope. Existing ledger acceptance acquires that
scope before its SQLite admission transaction and retains it through the
post-commit native accepted transition. Final manual-stage publication acquires
the same scope before the review-store lock, then revalidates exact selected-slot
claims and accepted/uncertain boundary state before issuing or retiring authority.
Destination confirmation holds neither scope nor publication locks. This gives
old acceptance versus manual supersession one winner, including the interval
after validation and the interval after commit but before native notification.
Ordinary claim-time validation never acquires the acceptance scope: its existing
progress-lock→DB→store/registry order cannot wait on a scope held by acceptance
while acceptance waits on DB. Manual publication makes no AgentRuns ledger calls. Accepted
links/charges remain consumed; manual wins leave the old preparation available
only to the existing proven-abort cleanup. No schema or durable-body changes.

### Task 8 approved seam clarification and implementation handoff

ADR required: yes. ADR path: `backlog/decisions/281-primary-agent-multi-model-code-review.md`.
Reason: final review capacity must be checked after the gateway owns continuation preparation.

Extend `Chat/console_agent_bridge.py::_StreamingModelAdapter` with the internal optional
`review_request_admission` scope. Capture its reject-only callback before async work,
prepare with the existing gateway (without generic safety-window trimming), then admit
the actual `PreparedProviderRequest` before dispatch. Use effective selected/fallback
capacity and accounting. Direct AgentService requests use existing exact estimators;
private direct continuation without a safe prepared seam refuses at request level.
Neither path may reset state, widen limits, change model, or classify that request gap
as an individually oversized unit. This is the controller-approved narrow extension
of Task 8's request-builder scope, not a new workflow or tokenizer.

Implementation and focused RED/GREEN evidence:
`.superpowers/sdd/2026-10-18-multi-model-code-review-implementation/task-8-report.md`.


#### Task 9 implementation boundary clarification

ADR required: no new ADR; [ADR-281](../../../backlog/decisions/281-primary-agent-multi-model-code-review.md) applies.
Controller-approved file decomposition adds `Agents/code_review_checks.py` for
per-review check state and native receipt transitions, consumed by the existing
service/store. Stock Raw CLI does not implement aggregate write bounds, so its
checks resolve unavailable. The optional capability is owned by the actual
external executor; fault-controlled protocol tests are not evidence of stock
bounded execution. Native originals are displayed in one session-retained reader
before verification, with body-free transcript references. Related proposal unit
IDs do not prohibit independently supported source-only dispositions.

## AC23 repair follow-up (authorized 2026-10-26)

The user requested resolution of the remaining AC23 regression and resource gaps.
Keep AC23 unchecked until the targeted matrix passes without unresolved resource
warnings; AC24/live-provider qualification is separate and remains unchecked.

ADR required: yes (existing decision clarification).
ADR path: backlog/decisions/228-sqlite-store-kernel.md; ADR-281 continues to govern review authority.
Reason: apply existing operation-owned lifetime semantics to a finite Workspace registry read without changing held-kernel, transaction, or cross-thread contracts.

### AC23-A: Retire finite Workspace registry read acquisitions

- Read the acquisition census and diagnosis in the SDD ac23-resource-diagnosis.md.
- Add failing tests for the real get_runtime_binding read, absent/Default repair/error paths, repeated/concurrent workers, borrowed handles/transactions, and native close failure custody. Assert exact native retirement, not only registry bookkeeping.
- Wrap the entire DB-using method in existing operation_owned_connection(self.db), including any Default repair. Preserve input validation, exception mapping, fresh exclusions and transaction semantics; no producer/kernel/global shutdown redesign.
- Extend actual Console teardown evidence for exact Workspace native leases after worker drain. Use no forced GC, cross-thread close, warning suppression, or registry deletion.
- Run affected registry/SSH/exclusion/finite-retirement and native review journey tests; amend ADR-228 and QA with actual before/after evidence.
- Repair the separately observed registry-test ownership through explicit yielding factories for exact test-created DBs, same-producer worker cleanup, and the one Media owner; assert native custody and repeat the focused owner census. Do not add a global teardown sweep.
- Commit scoped implementation and obtain independent review.

### AC23-B: Repair test profile ownership and qualify regressions

- Read ac23-regression-diagnosis.md once available; reproduce representative boundaries before editing.
- Use existing private/bootstrap profile fixtures according to actual test requirements; preserve intentional per-test config cases. Do not reset or bypass production raw-source guards.
- Verify the previously blocked fleet/runtime, reliability, run-log, local-provider and Settings boundaries with targeted suites selected from the recorded required matrix. Fix newly exposed real defects at their owner; distinguish evidence from test harness errors.
- Run combined affected resource/Console cases with exact native owner observations and unsuppressed FD guard. Update current QA and AC23 only against observed results, preserving historical logs.
- Commit and independently review the completed repair; keep task In Progress while AC24 is unverified. No full-suite, live provider, push or merge authorization is added.


### AC23-B discovered owner boundary (authorized 2026-10-26)

ADR required: yes (existing amendment).
ADR paths: backlog/decisions/126-complete-local-backup-and-recovery.md and
backlog/decisions/228-sqlite-store-kernel.md; ADR-281 links the qualification.
Reason: independently admit the concrete fixed worktree directory without
widening an enclosing AgentRuns operation or dropping refused-launch custody.

1. Preserve the recorded isolated-child RED and native ownership traces.
2. Add real no-argument fixed-root preparation/restoration, native failure,
   unrelated-path, pause-before-admission and refusal tests under native owners.
3. Implement only the Backup_Recovery-owned directory preparation seam; retain
   the parent DB operation and existing Git/source authority. Translate only
   native recovery refusal into a typed, body-free worktree refusal.
4. Run the actual isolated-child, refused/pause handoff, thread-start failure,
   uncertain-cleanup and captured source-authority controls; then complete the
   previously blocked affected files and bounded combined AC23 matrix.
5. Record actual native retirement and unsuppressed FD results, archive logs,
   update current notes and independently review. Keep AC23 open for any
   unresolved required failure/resource warning and AC24 open without live calls.

### AC24 diagnostic repair: observed Luna token-cap rejection (2026-10-27)

One separately user-approved diagnostic POST returned HTTP400
unsupported_parameter for max_tokens on gpt-6-luna, requiring
max_completion_tokens. It was not an exact replay of the original reviewer request.
The diagnostic budget is exhausted; this repair is offline only.

ADR required: no new ADR.
ADR path: N/A; existing provider routing and shared request-capability boundaries remain.
Reason: correct a proven request-parameter projection for one known model; do not infer sampling or unprobed family behavior.

- Add failing shared-capability and real chat-payload tests for gpt-6-luna, streaming and nonstreaming, plus normalized-ID and legacy controls.
- Extend only openai_model_requires_max_completion_tokens for the observed model. Preserve sampling handling, model selection, API routing, existing family behavior, budgets and authority.
- Run the affected request/capability tests, scoped static checks and independent review. Archive sanitized diagnostic evidence and keep AC24 unchecked until useful live qualification is authorized and completed.

### AC24 fresh bounded continuation and role-schema repair (2026-10-27)

The user's “ok continue” authorizes a fresh bounded continuation: at most twenty
new chat transport attempts and three new review children, preserving native
max_subagents=2 per primary turn. Retain historical accounting separately:
twenty failed qualification attempts and one separately approved diagnostic.
Do not automatically repeat a paid attempt or reset this new ceiling.

ADR required: no new ADR; existing ADR-281 role/capability policy applies.
ADR path: backlog/decisions/281-primary-agent-multi-model-code-review.md.
Reason: align model-facing read/submit schemas and guidance with existing host-bound role permissions; no new execution authority or API.

- Add failing real-provider schema/description tests for primary, reviewer and verifier roles, cross-instance isolation, and unchanged denial of guessed forbidden calls.
- Narrow advertised read views and submission kinds from authenticated host binding/stage, using existing owners. Reviewer discovery must show the actual source-query entry path. Preserve execution validation, revision semantics, permissions and peer-report blindness.
- Run affected schema/dispatch/isolation tests and obtain independent review before more generation.
- Prepare a separate temporary QA-runner revision with a fresh twenty-attempt ceiling, exact HTTP status/transport-error observation without credential/body leaks, and all prior evidence retained. Default remains non-generating; validate counter/redaction/normal forwarding offline.
- Run at most one valid native preauthorized journey with the selected exact models; record actual native OpenAI API routing rather than assuming an endpoint. A normal temporary Responses/fence setting may be chosen if supported; do not rewrite payloads at transport or claim it live-verifies the separate Chat Completions cap fix.
- Keep AC24 unchecked unless useful review/verification, ordering and full original mapping are demonstrated. Any partial/error/cap outcome stops further live work at the ceiling.

### AC24 offline fenced-child disclosure qualification (2026-10-27)

The fresh20-call journey exhausted its allowance without structured submissions.
Luna claimed submit was unavailable; direct offline catalog/planner checks disclose
both tools, so no production defect is established. Existing real child-dispatch
role tests cover native tools; extend that exact test for fenced mode before
considering any behavioral change or another live run.

ADR required: no.
ADR path: N/A; existing ADR-281 authority and provider contracts are unchanged.
Reason: test-only evidence for an existing request path.

- Exercise reviewer and verifier roles through the existing actual child-spawn fixture with native tools both enabled and disabled.
- Assert both read and submit are disclosed in each actual child request, with matching role views/kinds and no primary-only review tools. Preserve existing source/submission assertions.
- Use small assertions against the actual native tool schemas or fenced system/tool instructions; no new production parser, helper framework, scripted review-quality claim or model-response rewrite.
- Run the focused test and affected isolation file/static checks; preserve any genuine failure and stop for diagnosis rather than weakening an assertion. Independently review the scoped test change.
- Keep AC24 unchecked and task In Progress. No more HTTP, generation, paid probes or user-config changes are authorized by this offline check.

### AC24 isolated provider protocol diagnostics (2026-10-27)

The user approved continuation after41spent calls. Prepare/review a synthetic two-turn fenced-tool diagnostic for exact Luna and DeepSeek using normal runtime/request/feedback owners, at most2actual transports per provider and4shared; historical41remain charged, aggregate45. No local coordinator/Anthropic generation or end-to-end replay. Capture exact synthetic schema/feedback disclosure and observed output/status without user source/keys. Default offline and explicit execute; no retries, redirects, transport rewriting or fabricated model results. Validate offline and independently review before one live invocation; fix only a demonstrated defect under a separate scoped plan. Preserve all prior evidence and keep AC24 unchecked until actual useful qualification.

ADR required: no. ADR path: N/A; existing ADR-281 boundaries apply. Reason: diagnostic-only observation of existing provider/runtime behavior.

### AC24 role-correct child brief and captured-target context (2026-10-27)

The four live synthetic streaming probes pass marker round trips but do not qualify review. Source inspection shows reviewers still receive the primary coordination guide, and launch_task gives a digest without captured target kind. Correct this separate prompt/context gap offline: preserve the primary guide, use one shared independent-review brief stored/fingerprinted by the existing preparation owner, retain verifier guidance, and include validated captured target.kind for both child roles without paths/URLs/peer content. Keep all source/role/permission/revision/transport behavior unchanged; no extra paid calls. Add failing actual child-task/fingerprint/role/target tests, apply the smallest owner change, run affected targeted tests/static checks, update relevant architecture and independently review. Keep AC24 unchecked absent useful live qualification.

ADR required: no new ADR; existing backlog/decisions/281-primary-agent-multi-model-code-review.md applies. Reason: repair role-correct context composition within existing owners, with no new authority or public API.

### AC24 next qualification preparation — execution not approved

Prepare a separate copy of the reviewed native Console runner for one future40-attempt/3-child journey (historical45spent; aggregate85onlyifapproved), preserving native2-per-primary, exact models, source controls and600-second limit. Capture bounded passive actual child schema/disclosure/feedback metadata so model availability claims can be checked. Do not capture full source/prompts or alter authority/requests/responses. Keep default offline, require explicit40-attempt acknowledgement and one-use claim after newuserapproval, validate history/counter/forwarding/trace/config offline and independently review. No furthergenerationorHTTPisapprovednow.

ADR required: no. ADR path: N/A; existing ADR-281 provider/role boundaries unchanged. Reason: temporaryQAobservability and a concrete reviewable proposedqualification, not a production feature or acceptedbudgetincrease.

### Correct generic QA wording before any multi-model preparation

The approved40-request invocation spent5localcalls on an ordinary single-agent review because the QA request omitted multi-model. It created no review context or child. Preserve that complete outcome/one-use claim and chargeall5. Correct the QA message to Do a multi-model code review of my changes and verify the findings, using the unchanged native parser/entry path. Permit one correctly specified journey using only residual35oforiginal40 (historical50, unchangedaggregate85), with frozen first-attempt/zeroacceptance proofs, separateone-usecopy, native2/child3/wall600 andallnormalauthority/source/UI/trace/cleanup unchanged. No reset/refund/acceptedlaunchreplay/newallowance or automaticsecondcorrection. Validate/reviewofflinebeforeuse.

ADR required:no newADR; existingADR281applies. Reason: correct the QA user intent to the already-authorized multi-model workflow before any workflow launch, with unchanged authorization/totalphysicalceiling.

### Zero-generation catalog preflight refusal — bounded resume

The corrected explicit request hit OpenAI GET/models500 beforeanygeneration or workflowacceptance; anothermetadata-onlyGETalso500. Preservebothrecords/claims and residual35withoutrefund/reset. Permitoneguardedpreflightresumeonly afterfrozenzero-generation/zero-acceptance/quiescence/source/team/config proofs. OnlythetemporaryQA supplemental catalog check may be marked degraded for exactOpenAI/LunaHTTP500 whennativeexplicit routing/credentials/verifiedpositivecapacity alreadyready; catalogavailabilityremainsunknown, notpositive. Actualnativegenerationdecidesavailability. Allotherreadinessfailuresandallproductionguardsremainblocking. Validatepositive/negativeofflinecasesandindependentreviewbeforeoneinvoke; no nativepolicy changes, furtherretries orlargerbudget. Historical50/residual35/aggregate85unchanged.

ADRrequired:no newADR; existingADR281applies. Reason: remove an overstrict supplemental QA health assumption without bypassing normal model/source/permission admission.


### AC24 post-run observation correction and native-tool feasibility

The approved40series used25requests (genericQA5 plus actualmulti-model20), with70totalspent/15unspent. Twoacceptedreviewers remainincomplete; noverifier. Preserveclaims/accounting, do notautomatically replayacceptedroles orstartanotherfulljourney. The passive trace omitted the separate Prepared.system_message carrier; a focused zero-network Consolechild/provider-wire control preserves schemas and source-listfeedback. Correct only the privateQAobserver to inspect bothcarriers, label prepared evidence accurately and retain historical false flags. Add small offline positive/negative carrier checks, independently review and archive exact probe/outcome. No production sender fix is justified. Read-only native-tool feasibility may inspect existing owners and one narrow synthetic control before recommending furtherpaid qualification; no model/route/configchange or generation is authorized by this offline check. AC24staysunchecked.

ADRrequired:no newADR; existing ADR-281 applies. Reason: QAobservation/evidence correction and bounded read-only diagnosis, with no production contract or authority change.


### AC24 exact Luna native-function route correction

The zero-network exact-route probe demonstrates native function round trips for
DeepSeek/Haiku and real OpenAI Responses request/response/history gaps for Luna+none.
This is separate from the earlier false sender-trace flag. Official exact Luna
documentation supports Chat Completions function calls with explicit reasoning_effort
none. Repair only the existing OpenAI routing/payload owner for this exact model,
explicit normalizednone, nonempty canonicalChat functiontools and absent Responses-
onlysummary/verbosity controls; forward Chat reasoning_effort=none. Preserve all
plain/fenced/empty/mixedhostedtool/othermodel/non-none/summary/verbosity branches and
existing sampling/token/permission/source policy. No generic mode knob, fullResponses
implementation or protocol parser. Add RED request and actualjoined Console/gateway/
provider nativecall/result tests in stream/body modes with negative controls; apply
smallest ownerfix, run targeted checks/static comparison, update architecture and
independentlyreview before any paid qualification. Scope/callers/failure evidence
and exact requirements: Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/luna-native-route-fix/brief.md.
No furthergeneration or acceptedreviewerreplay is authorized by this offline repair.

ADRrequired:no newADR. ADRpath: backlog/decisions/006-provider-aware-generation-settings.md
(existing), backlog/decisions/222-provider-http-session-reuse.md (existing) and
backlog/decisions/281-primary-agent-multi-model-code-review.md (feature). Reason:
routine model-specific compatibility repair in existing provider owners, without
a new protocol/runtime/storage/authority boundary.


### Future native-tool qualification proposal — execution unapproved

Prepared a fresh native-tool Console journey under independently reviewed source
26c291443ce77a268444a3ed9adc4bc666061c64. Historical70 requests remain charged;
the prior40 allowance has25 used and15 unspent. A proposed fresh40-request journey
requires explicit approval for25 additional requests and three fresh child launches,
with aggregate110 only if approved. Native two children per primary, shared three
children,600seconds, exact models/source/authority/UI/continuation and cleanup stay
bounded. Old accepted contexts and claims are preserved; no old stage is replayed.
The runner defaults offline and requires all three exact acknowledgements before
credential/I/O access; preparation grants no live authority.

Ordinary temporary native_tool_calls=true uses actual saved-role schemas and normal
provider owners. Offline physical-payload observations select Luna Chat/none/current
completion cap and existing DeepSeek/Haiku native routes. Passive metadata hashes
provider call/result IDs and body/argument bytes while preserving objects and host
run IDs; fence fallback inspects both prepared carriers. Counter, forwarding, narrow
readiness, frozen70 history, trace and normal composition checks pass with zero real
network/generation/children. Local Qwen native acceptance and live review quality
remain unqualified. Independently re-review the small metadata fix, then request
new human approval before one invocation; no automatic retry/reset/refund. Exact
reviewable runner, patches and receipts: Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/native-qualification/.

ADR required:no new ADR. Existing ADR-006/222/240 apply; temporary QA preparation
changes no production/runtime/storage/permission boundary or saved user setting.
AC24 remains unchecked and TASK-35582 In Progress.


### AC24 native selected-destination continuity repair

Before implementation, adopt the exact 14-file existing-owner design and controller
clarification in Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/continuation-identity-design/.
Both defects must be covered: raw custom-ep selection versus canonical family, and
disclosure origin versus full API base. Freeze canonical continuation provider,
protocol and exact base in the accepted destination (nested JSON V2); carry the exact
raw selected provider in existing assistant continuation JSON V3. Bind at the actual
per-call host adapter before runtime retention and managed signing; independently
validate store/repository handoff and terminal settlement under existing CAS/owner
barriers. Host history/recovery checks raw plus canonical targets; lower provider
replay checks execution only after host admission. V2 signatures/bytes stay unchanged;
V3 managed signatures include the raw pin. No mutable lookup, alias wildcard, candidate
relabel, new registry/table, permission, route substitution or automatic retry.

Legacy V1/V2 continuation remains byte-canonical and explicit direct-provider resume
works; unbound virtual legacy restore refuses. Legacy four-field destinations remain
readable but cannot authorize new native handoff without the tuple. Session-only
endpoints remain absent from durable authority; ordinary text/fenced and explicitly
ephemeral paths keep their behavior. Preserve privacy/2,048-byte cap and validate both
raw-slug equality and existing execution/model/mode/base comparisons.

TDD must include actual Console two-request/native persistence, closed/reopened A vsB
virtual endpoints sharing URL/model, exact target changes, foreign/stale ownership,
terminal settlement, independent builtin child under custom parent, V1/V2/V3 codec
and managed signature controls, and selected DB/sync/chatbook/history/trace carriage.
Use the design's bounded RED/GREEN gates and existing fixtures. Run no full suite or
paid calls. If a fixture contradicts the new contract, preserve the failure and fix
its explicit authority prerequisites; never weaken a guard. Independent scoped
review follows a single production implementation wave. Green offline checks do not
complete AC24 or authorize another live run; current71 spent/39 unspent/three unused
children and consumed one-journey claim remain unchanged.

ADR required:yes; existing ADR-063 and ADR-147 amended before code, cross-linked
from ADR-281 and ADR-179. SQL schema/columns/constraints do not change; JSON versions
are explicit and old readers are tested. No SQL migration/backfill is needed for
this chosen layout. A discovered DDL/new-owner requirement needs a new assessment.


Native identity implementation refinement: the actual Console second request
reveals missing provider_continuations forwarding in Chat/Chat_Functions.py.
Authorize this fifteenth existing owner: one custom-hosted-only map entry, with
legacy custom maps unchanged, a documented parity exception and actual forwarding
plus second-request/completion coverage. No broader map/provider/authority change.
Existing ADR-063/147/179 applies; no additional schema or ADR change.


Native identity implementation refinement: actual reopened pending Resume exposes
a missing production gateway callback hidden by fake translators. Implement that
existing callback in console_provider_gateway.py with canonical assistant/native
call rows and only existing result rows, exact IDs/arguments, no ordinary-text
reasoning or invented pending results. Existing explicit action/target/permission/
executing-state and provider replay guards stay in place; verify real pending
Resume and exact-once wire pairing. Existing ADR-063 applies; no new owner/schema,
automatic execution or financial allowance.


Native identity recovery refinement: real hydration gives restored assistants a
terminal live status, so explicit Resume must activate its existing generation
owner using expected active checkpoint and expected persisted message/continuation
version. Validate under generation scope, then reset only live status and old
terminal-attempt bookkeeping; preserve content/checkpoint/CAS. No generic retry
clearing or automatic activation. Add exact stale/foreign/executing/partial-content
controls alongside actual closed/reopened completion. Existing store/controller
owners and ADR-063 apply; no new schema/permission/financial allowance.


### AC24 consecutive native-round owner projection repair

Manual retry reached two successful native rounds and valid bound V3 persistence,
then request three correctly refused two owner-marked rows for one current group.
Repair only the current owner's send-copy attachment in Agents/agent_service.py:
retain the existing newest exact native-call-ID anchor once; clear older validated
current-owner native echo markers, preserving original history/rows/order/steering,
call/results/private rounds and other owners. Keep strict prepared-request ownership
validation and fail closed for missing/malformed matches. No new state/codec/schema
or authority. Exact diagnosis: /private/tmp/task35582-ac24/multiround-owner-diagnosis.md
(to be archived with results).

TDD uses normal configured Console with two then three native tool rounds and final
response, plus one-batch multi-call, historical/current owners, steering, CaptureOn,
restore-plus-new-round, and selected small-window/pruning/verifier protection.
Assert exact POST count/call-result order, one current anchor, complete private
callset, unchanged input objects, proper V3 pins and final state. No broad suite or
live request. Existing ADR-063/240 ownership applies; no new ADR/migration.
Current73 spent/37 unspent/three unused children remain; consumed manual claim
prohibits automatic retry. Offline fix and independent review precede any new
manual live authorization.


Consecutive-round test-scope refinement: full Capture On fails afterPOST1 at the
existing trace reservation count/span boundary on exact5ee baseline, before this
repair's duplicate-owner point. Keep the one-owner source fix; validate its CaptureOn
provenance/single-artifact attachment through the real request factory, plus complete
normal CaptureOff multi-round/recovery/order controls. Preserve and report the
blocked full CaptureOn receipt/version as an unresolved baseline limit; do not
weaken trace guards or claim successful end-to-end CaptureOn qualification.


Consecutive-round repair completed offline through3d1b4442f1 after scoped review
closed one Important malformed-owner case. Existing projection now rejects present
non-string or blank markers before normalization; absent keys stay valid.
Normal Console gates at72737 and the later18-case projection matrix are versioned
receipts, with zero introduced static/format findings. Full CaptureOn baseline
limit and semantic-only evidence remain as above. No plan/schema/authority extension.

The remaining37 proposal is bound to reviewed3d1 for one offline normal composition
and independent QA review. Its possible execution scope requires new explicit human
approval:73spent+atmost37=110, three TOTAL unused children; preparation retries only
before any new child acceptance, certain cleanup/accounting and a fresh separately
reviewed one-use driver. Earlier accepted roles/claims remain immutable; no budget
replenishment or automatic retry is inferred from unused requests. AC24 remains open.


### Qualification observer correction after the remaining37 journey

The approved run used32requests (20Qwen7Luna5Deep), total105/5unused, with two
accepted reviewers and one unused child. Luna sealed one real finding; DeepSeek
ended incomplete. The host admitted native V1 continuation, but the temporary QA
observer rejected its ordinary primary's expired-cache fallback (32000 estimated,
3835 input, no overflow) before any POST. Actual production primary sends are
intentionally nonblocking; existing cache contract tests require this behavior.
No production capacity defect is established.

Ruling57 corrects that QA-only false failure in a new offline-only patch: retain
known-overflow refusal for everyone and verified-capacity requirements for all
reviewer/verifier/unclassified requests; permit the host primary's explicit
estimate through unchanged native admission and label it unverified. AC18 actual
review-bound batching remains strict. Preserve the paid driver/claim/outcome and
require role/overflow negative controls, actual prepared33, independent QA review.
No new production contract/flag/ADR, cache injection or generation. After accepted
children the scope forbids a new preparation retry; financial105+5max110 and one
unused child do not replenish/replay accepted work. AC24 remains unchecked.


### Repair actual native Capture On transport parity

The two passive actual Console reproductions locate the first CaptureOn failure:
verify_provider_request_shadow rejects api_base_url only. Gateway dispatch strips
/chat/completions for custom-hosted, but independent reconstruction retains the full
chat URL. This invalidates POST1 capture to one omission item; the following strict
count/span refusal is downstream. Exact RCA: /private/tmp/task35582-ac24/capture-trace-diagnosis.md.

Ruling59 requires the smallest existing-owner repair: share the existing pure adapter
API-base projection from console_provider_endpoints between gateway and independent
reconstruction. Preserve raw accepted/durable identity, actual dispatch, endpoint
privacy and every strict trace guard. No new config, dependency, DDL or trace format.
ADR required: no new ADR. ADR paths: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md, backlog/decisions/179-generic-hosted-provider-engine-and-preset-registry.md, backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md.
Reason: implement existing endpoint/final-value parity contracts without changing
ownership or capture semantics.

Before declaring CaptureOn fixed, actual normal Console2/3native-round and multi-call
CaptureOn/Off controls must pass with meaningful completed trace assertions, plus
provider endpoint positive/negative and forged-kwargs alignment refusal tests.
The endpoint fix alone does not prove later private continuation delta support; any
further structural failure needs exact evidence and controller ruling before guard
or contract changes. Targeted tests/static checks and fresh scoped review only.
No live retry or additional financial allowance; current105/5unused/1unusedchild.


### Bounded live-native trace advance implementation

ADR required: yes, existing ADR097 amended before implementation.
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md.
Reason: narrow new live-native admission and trace-only stored semantic projection;
existing ADR063/179 checkpoint custody and ADR281 review scope remain unchanged.
Detailed requirements: QA/ac24/native-trace-delta-design/{design.md,controller-adoption.md}.

1. Read the adopted contract and inspect existing final-values/capability/runtime/
   store/repository/reader/maintenance callers. Audit codec/enum/archive constraints
   before asserting no SQL migration. Preserve current native runtime/checkpoint.
2. Reuse existing host invocation projection/sealing provenance and current
   generation/message/CAS authority to build one bounded capture-only live witness.
   Revalidate exact source/run/previous call/target/policy and raw old-round prefix.
3. Add the narrow one-continuation replacement plus exact native message-suffix
   append through existing delta/capability/events transaction; preserve domain
   ordinals, other owners, existing plain/completed-turn/retry behavior and fences.
4. Encode only the new sanitized round and bounded checkpoint metadata using a
   distinct trace-only version and the existing replacement predecessor edge.
   Legacy full artifacts remain readable; provider/current checkpoint bytes stay.
5. Share node-aware materialization across durable comparisons and native reader.
   Cover graph corruption/count/byte/depth limits and archive/fork/GC dependencies;
   re-establish cold custody or refuse explicitly before provider entry.
6. TDD existing actual Console CaptureOn2/3-round and multicall REDs to full
   completion, with endpoint and CaptureOff controls. Add focused strict raw/custody/
   policy negatives, atomic rollback/races/owned readback, privacy, legacy/corrupt
   read/retention and measured artifact-plus-metadata growth. No full sweep.
7. Run scoped exact-BASE static/format/whitespace comparison and self-review.
   Update trace architecture/user-facing limitation docs only to proven behavior.
8. Commit only owned source/test/docs and request fresh independent scoped review
   of both compliance and quality. Root retains all evidence and updates AC23 only
   after complete passing native matrix; AC24 remains live-incomplete.

Do not silently expand into new registries/tables/generic patch engines or alter
private checkpoint formats/permission. A missing necessary owner or DDL requirement
needs controller assessment before that edit. No extra paid calls or accepted-role
replay; actual105/max110, five requests/one child financially unused.


Runtime custody prerequisite (Ruling61, before this runtime edit): the existing
continuation persistence callback receives only projected checkpoints/events and
returns None. Exact raw-to-projected provenance is currently discarded. Add one
optional capture-only host acknowledgment after that callback succeeds; carry
exact invocation raw/projected relation to bounded current adapter custody, then
existing store projection/sealing and final owner/version/CAS validation. Keep
ordinary/CaptureOff behavior, durable persistence barriers and transition semantics.
Raw values never enter generic event serialization, provider kwargs, DB bodies,
logs, diagnostics or a history registry. Failed/missing/stale acknowledgment grants
no capture witness or execution power. Cover private-body projection, managed
sealing, persistence callback failure and no duplicated side effects explicitly.
No SQL/checkpoint-format or review-job owner change is introduced.


Native trace repair outcome:7c4d+9234 implements the adopted ADR097/Ruling60/61
contract. Independent spec+quality review approves with zero open findings after
one P1 unchanged-error retry fix. Final22native/5existing retry controls and prior
56owner controls pass (overlapping/versioned). ExactBASE13failed/3error and one
GC-reopen failure remain explicit. No new schema/current checkpoint format or
permission; compact graph reads/retention/growth and scoped atomic sweep covered.
AC23 reclosed for actual native CaptureOn; AC24 remains live-incomplete. Full
commands/hashes/limits and initial/final reviews are in QA/ac24/native-trace-delta-fix/.
No more provider calls;105spent/5financialunused/1oldchildunused.


Futurefresh40 qualification is prepared/reviewed offline on923439, retaining
R57ordinaryprimary versus strictchildcapacity behavior and originalnativebudget.
ONE0network actual-owner composition and independentQA review pass. The proposed
cap40 uses5carry+35NEWrequests and3freshchildren, aggregate145 onlyafter explicit
humanapproval. No source/role/target/parameter substitutions or authority changes.
Before any new-scope childacceptance, bounded preparation retries may be allocated
onlyfrommeasuredremainingbalancewithcertaincleanup/accounting and reviewednewcopy;
afteracceptance no newprepjourney. Oldroles/claimsremainimmutable. No additional
paidI/O authorized by this plan/proposal;AC24stillunchecked.


### Accepted review-wake SYSTEM and tool-plan repair

ADR required: yes, amend existing ADR281/230; clarify existing ADR063/179.
ADR paths: backlog/decisions/281-primary-agent-multi-model-code-review.md and
backlog/decisions/230-console-send-preparation-and-io-ownership.md.
Reason: narrow accepted-review-only bounded pinned capacity and stage reachability,
plus supported ordered SYSTEM preparation under unchanged strict engine validation.
Adopted before code: QA/ac24/review-wake-repair-design/{design.md,controller-adoption.md}.
No SQL migration, new registry/job owner, global wait/clamp change or extra budget.

1. Read diagnosis/design and inspect direct callers and architecture. Reproduce
   independent mixed-carrier and estimated-window no-tools failures at actual
   gateway/handler/planner boundaries, keeping role=USER diagnostic-only.
2. Select distinct_roles for registry-derived engine requests with later SYSTEM
   rows. Preserve ordered roles/provenance and one immutable prepared artifact;
   Safe and Full capture must honor that carrier. Keep engine guard unchanged.
3. Factor the existing accepted-boundary authority/source check. In controller
   immediately post existing review-wake acceptance, use typed actual resolution
   in existing context-window operation, with bounded cache/singleflight/deadline.
   Check grant/source/session/Stop/priority both sides of await with no lock held.
4. Before model dispatch, inspect normal schema plan for admitted next-stage
   spawn/review reachability or explicit refusal. Do not force tools, enlarge
   budgets or alter ordinary nonblocking/bootstrap/progress behavior.
5. Add configured Console→actual handlers→fake physical HTTP two sealed/presented
   reviewers→fresh verifier under [2,1], expired/cold pinned metadata and exact
   finite chain. No stream/model capacity monkeypatch substitutes for this oracle.
6. Run focused target/metadata/lifecycle/permission/small-window/carrier/provenance
   negatives and ordinary behavior controls using existing fixtures. Source and
   cancellation guards must preserve accepted charges, partials and no replay.
7. Scoped exact-BASE static/format/whitespace, self-review, architecture notes, owned
   source/test commit and fresh independent spec+quality review. Preserve versioned
   earlier native trace/GC/ack/ownership evidence; do not repeat unrelated cohorts.

AC23 reopens for actual native accepted-stage reachability regression until these
gates/review pass. AC24 stays unverified: latest30calls/135total/10unused from145,
both reports sealed, no verifier. No new preparation retry after accepted children.


### Source correction to accepted-wake carrier diagnosis (Ruling63)

Exact BASE7952422 controller source deliberately projects automatic SYSTEM
transcript notices as payload-only trailing USER while preserving AGENT_WAKE
authority and order. The prior reconstructed SYSTEM-carrier diagnosis was not
the actual wire path. Its carrier conversion/Safe-capture clauses are superseded;
ordinary serialization and the strict engine stay unchanged. Accepted-review-only
pinned capacity/revalidation and normal schema reachability requirements remain.
The full handler-backed fixture instead exposes completed checkpoint forwarding
without native replay rows. A focused replay contract is pending before source
changes at that extra boundary. Original diagnostic evidence is retained with
QA/ac24/review-wake-repair-design/diagnosis-controller-correction.md. No authority,
schema or additional live allowance follows from this correction.


### Canonical completed-history replay amendment (Ruling64, before code)

Adopt QA/ac24/review-wake-repair-design/completed-replay-correction.md and
completed-replay-controller-assessment.md. Preserve the corrected existing
transcript SYSTEM/payload USER policy; carrier-only work is removed. The real
accepted-wake failure is retained complete checkpoint forwarding without its
canonical native owner rows, alongside the independently proved capacity/tool loss.

After existing host owner/target admission, derive canonical recorded assistant/
tool history at final preparation from the exact complete persisted checkpoint.
Reuse the existing translator; keep one collapsed semantic owner plus attachment
as the atomic budget/eviction unit. Preserve exact IDs/args/results and one final
content, private reasoning sidecar, all retained complete groups alongside current
active groups, child isolation and explicit active Resume gates. No re-execution,
fuzzy owner match, missing group tolerance or engine relaxation. Count the same
projection before fit and avoid duplicate completed-expanded body charge.

A narrowly deterministic one-to-many trace derivation may reuse existing completed
owner/revision, private attachment and terminal lineage/range machinery. Recompute
all output rows under frozen policy; derived rows are not fabricated saved ordinary
revisions. Preserve count/span/source/actor/turn/CAS and final shadow gates. No
arbitrary tool span acceptance, active-custody resurrection, growing per-call
history arrays, codec/GC change, registry or new schema. Any needed durable shape
requires concrete assessment before adding it. Existing provider/checkpointV3 and
legacy readers remain. Actual handler-backed [2,1] plus CaptureOff/On, replay/live
coexistence, counting/eviction and strict negatives are required before closure.


Engine schema clarification (Ruling65): exact provider_continuation._parse_value
lines633–645 allows a call-free final reasoning round only for the specified
Moonshot case, not engine/custom-hosted checkpoints. Their completed checkpoint
contains tool rounds; final visible content belongs to the admitted saved owner.
Project recorded canonical assistant/tool rows then that exact saved final row
once. The earlier design's canonical-terminal-row equality assumption is superseded
for engine history. No invented round, codec/JSON/schema change or final loss.
Owner/revision/target and derived provenance/counting gates remain unchanged.


Completed replay compound-transition clarification (Ruling67): the actual later
MANUAL CaptureOn request changes both the old native tool span and its same-owner
active checkpoint attachment. Permit only this exact compound replacement under
the existing saved-response/terminal witness, canonical replay derivation and
frozen-policy recorded-round/target proof. Preserve the whole unchanged prefix,
other attachments, source/run/turn/policy lineage,256 span/count and CAS/final
shadow checks. Changed/foreign checkpoints, altered recorded rounds and pending/
executing ownership cannot pass. Use the existing graph transaction and retention;
no generic mixed span, codec/GC/schema or active custody extension. The required
negative controls and independent review must establish this boundary.


### Accepted-wake repair execution outcome and next qualification proposal

Source48e90ce+c3f99ec implements corrected R63–69. Fresh scoped re-review approves
I1/I2 with0openCritical/Important; one Minor precedence-fixture gap stays explicit.
Final27covering cases pass; earlier22/31/59/52-success receipts are versioned, not
a unique sum. AC23 is requalified; AC24 stays open. Existing ADR281/230/063/179/097
links remain governing; no new owner/schema/permission or recovery expansion.
Exact reports/source snapshots/reviews and R70limited actualBASE cleanup proof
are in QA/ac24/review-wake-repair-fix/manifest.json.

QA/ac24/fresh40-after-wake-fix/manifest.json records an independently reviewed
one-use proposal: exact c3f99ec source,248hashes, one zero-network normal composition,
no paid call/claim and no current execution authority. Proposed40 =10carry+30NEW,
3FRESH children, aggregate175 only after explicit human approval. Actual135/max145
is unchanged; the prior accepted scope cannot be replayed. The same four models,
native2/shared3/wall600 and no automatic retry apply. Record every actual POST;
any approved preparation continuation is allowed only before new child acceptance
with certain accounting/cleanup, reviewed new copy and shared remaining40/3 limits.


### AC24 bounded cutoff

The latest human35-request/3-child grant is the already-recorded authorization
for runner a9cac4ec, not a second grant. That attempt spent30 generations and
accepted two reviewers; ten financial requests remain below145, but accepted
roles ended preparation-retry authority. Preserve every paid receipt and the
unapproved next-runner proposal without executing it, creating another driver,
requesting a higher cap or starting another diagnostic repair wave. The original
Task11 gate was20calls; subsequent separately approved attempts total135 without
a live verifier or unified report. AC24 is blocked at this cutoff and stays
unchecked/In Progress. Close out the reviewed implementation and offline
regression evidence honestly; the live continuation, Haiku verification and
complete verified/repudiated original-finding mapping are still unproved.


### Approved fresh40 execution outcome

The user explicitly granted30 new requests and three fresh children and directed
the prepared run, superseding the earlier cutoff for one invocation. The unchanged
reviewed driver used35 requests (170total/175ceiling), two reviewers and no verifier.
Both originals were presented; DeepSeek correctly identified the seed and rejected
the false string-input claim, but its own report is incomplete after a conflict.
The accepted V1 primary reached a spawn proposal, then native usage_unknown paused
the automatic chain before Haiku. Cancellation precedes runner shutdown; retained
metadata cannot distinguish missing provider usage from parser/adapter loss.
Keep AC24 unchecked and task In Progress. Five unused requests and one unused child
do not authorize an automatic replay. Exact evidence and independent outcome review:
QA/ac24/fresh40-after-wake-outcome/{report.md,review.md,manifest.json}. Source c3f99ec
is unchanged; AC23 remains checked. No new architectural decision or speculative
accounting/provider change is made by this documentation-only closeout.


### TASK-35582 usage accounting repair (2026-10-28)

ADR required: yes, existing ADR179/063/134/131; ADR path:
backlog/decisions/179-generic-hosted-provider-engine-and-preset-registry.md.
Reason: correct the existing custom-hosted streamed-usage request contract and
retain strict automatic token settlement; no new schema or runtime owner.

A two-request localhost differential proves the observed omission: current custom
payload without stream_options ends with no usage; include_usage=true yields
complete top-level usage. Set CUSTOM_HOSTED.stream_include_usage=True through the
existing record/builder field. Preserve no-usage tolerant parsing and unchanged
manual handling; automatic work still refuses absent/malformed usage. Unsupported
stream_options remains an explicit provider error with no new hidden retry or
parameter-dropping fallback; existing non-streaming is available. This deliberately
adds the standard OpenAI-compatible option to custom streaming requests.

Also preserve already-supported terminal choice-level usage in the gateway's
usage tap: only one terminal choice and no top-level usage key; top-level presence
remains authoritative, emitted event bytes stay identical, complete-count
validation remains at the existing budget owner. A real-handler/SQLite RED proves
this independent accepted-wire shape was previously lost. No estimates may clear
usage_unknown and no authority/accounting/capacity guard is weakened.

Implementation: new focused REDs for actual custom payload option and native
ledger settlement; smallest preset/tap repair; covering valid top-level/terminal
choice/non-stream usage plus missing/malformed/nonterminal/multi-choice/top-level
precedence and unsupported-option controls. Compare static/format to exact base,
independent scoped review, then spend at most remaining3 requests on decisive live
validation.170previous+2diagnostic=172spent/max175. No new review context or accepted
result may be fabricated to claim a full verifier journey within that balance.


### Usage repair execution outcome and remaining AC24 limit

Source7c67c6d fixes custom streaming usage requests and the accepted terminal-choice
usage tap. Independent review approves with no actionable findings. A real local
handler/gateway/ledger check settles83 tokens, leaves zero reserved and passes the
next authority check. Missing/malformed counts still fail closed. Custom servers
rejecting the standard option return a typed error without a hidden retry; existing
non-streaming behavior remains available. Existing ADR179 was amended before code;
ADR063/134/131 still apply, with no new schema or runtime owner.

26 new cases;147 covering passes/one exact-base inherited restore failure,12 parser
controls and2 final typed400 passes (overlapping). Zero introduced static/format
findings; the covering +221 descriptor warning remains unresolved. Exact source,
review, test and live evidence: Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/usage-unknown-repair/.

Three of the user-authorized remaining five requests were used:173 total/max175,
two left, zero new children. The accounting check uses a synthetic accepted fixture
and cannot count as a completed Console review. The prior native session closed
before its verifier and has no complete original-to-disposition mapping. A fresh
journey requires at least three role generations plus coordinator work; two cannot
finish it. No old grant/session is reconstructed, no extra requests are authorized
and no new retry is attempted. AC23 stays checked, AC24 unchecked, task In Progress.
Full suite, push, PR, merge and Done transition were not performed.


### Fresh42 outcome and narrow Haiku5.5 request correction

The user approved40additional requests and3fresh children (173spent+42available=215ceiling). Reviewed driver ee40e6f3 ran once on7c67c6d:38requests, two sealed/presented reviews, accepted freshV1 with native[2,1] and one commonchain. Haiku5.5's first request returned400 with temperature/top_k present; exacterrorbody was not retained. No verifier dispositions/unified result;211spent/4remaining/all3children consumed. Do not reopen closedacceptedauthority or count this asAC24complete.

Before furtherlivevalidation, correct the existing shared Anthropic capability table for exactHaiku5.5. The official migration guide requires omission oftemperature/top_p/top_k and adaptive rather thanfixedthinkingbudget. Audit existingpredicatecallers, add actualbuilder REDs, smallestexactfamilyentry+correctdatedcomments, focusedGREEN andstaticdelta, independentreview. ExistingolderHaiku controls remain. ADRrequired:no newADR; routineexistingprovider-validitytable update, no new owner/schema/authority. Officialsource:https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide. A futurediagnostic within4remaining provesonlycompatibility; no newchildorfulljourney is authorized beyond theconsumed3slot scope.


### Fresh42 qualification and Haiku correction outcome

The newly approved40additional requests plus2carry authorized one42-request,
3fresh-child run. Reviewed ee40e6f3 ran once on7c67c6d:38POSTs (20Qwen/6Luna/11Deep/
1Haiku), bothreviewers sealed/presented, freshV1 under native[2,1]/onechain. Local
continuation usage settled15,552tokens; Haiku's firstPOST400 preceded V1error and
parentcancellation. No verifierdispositions or unifiedmapping; finalauthority is
expired. Nativecleanup certain and source/team/configchecks unchanged.

Captured Haiku5.5 sampling fields conflict with official migration guidance;
exacthistoricalerrorbody absent. Source d87d165c corrects the exactexisting family
entry and its shared payload/Console controls. Causal9fail/1pass RED,671coveringpass,
10finalfocusedpass (overlap), zero newstatic/formatdebt; independentreview approved.
Neighbor54fail/39pass atprofileadmission remain, with one exact-base representative
and no universalattribution; earlier resource/fixture limits are retained.

One separately reviewed real Anthropic handler call returned200 with the same2
tool schemas andsamplingremoved, max512outputtokens. Nochild/reviewsession was
created. Total212/max215,3financialrequests remain, all3freshchild slots consumed.
This proves compatibility, not retroactive verifiercompletion. AC23checked,
AC24unchecked, task InProgress. No newjourney/replay/fullsuite/push/merge/Done.

ADR required:no newADR; routineexisting model-validity correction underADR063;
ADR281/134/135 remainunchanged. Exactsource, reviews, live results andlimitations:
Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/fresh42-after-usage-outcome/.


### Fresh43 Haiku-fixed qualification outcome

The approved40additional requests plus3carry funded one43-request,3fresh-child
journey on source d87d165c. Independently reviewed runner2fa4121d ran once:
43POSTs all200 (13Qwen/6Luna/8DeepSeek/16Haiku), native[2,1], one causal chain.
Both reviewer reports were presented first. Luna sealed empty; DeepSeek retained
3originals but ended incomplete atrevision1. Haiku accepted one submission then
rejected submissions including submission_conflict and ended incomplete atrevision1.
The retained events omit bodies; accepted kind/records and exact first rejected
payload are unproved. Final report/expansion reads refuse stage_grant_unavailable.

The physical stop is shared_generation_attempt_ceiling. Final automatic ledger
also records review_required/usage_unknown; this alone does not prove the earlier
usage repair regressed or explain the submit conflict. Native shutdown is certain
with zero remaining owners; recorded source/team/config integrity checks pass.
Total255/max255,0requests and0fresh child slots remaining. No replay or automatic
retry. Haiku request compatibility now works through the actual verifier path,
but no complete verified mapping exists: AC23checked, AC24unchecked, InProgress.

Source/Tests unchanged; prior targeted tests, independent reviews, inherited
profile/static/resource limitations, Minor fixture and recovery gaps retain their
exact scope. No full suite, new source fix, push, merge, Done or new allowance ask.
ADR required:no newADR; evidence-only closeout underADR281/134/135/179.
Exact receipts and independent review:
Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/fresh43-after-haiku-outcome/.


### Bounded delivery recovery (2026-10-28)

The user froze feature scope and prohibited another broad repair/requalification
cycle. Delivery is isolated on current dev e3d07afa; the original worktree and
raw history remain preserved. PR3286 is draft. The supplied result gate is
adopted in the next private runner; real Haiku tool-result/accounting preflight
must precede one final Console journey with native [2,1], presented originals
and complete dispositions/mapping/unified output. Reuse unchanged evidence.
New failures receive only a first-causal-boundary reproduction and narrow fix.

Actual historical receipts total255 requests/18 children with zero remaining.
One consolidated question proposes66 additional requests (preflight6, including
at most one manual causal-fix retry; one final run60) and3 fresh children. It is
pending, not a grant; do not revive any historical allowance or automatically
rerun the full workflow. See the bounded delivery recovery QA summary for exact
CI corrections, offline evidence and remaining acceptance boundary. AC24 remains
unchecked, task In Progress; no merge is authorized.

ADR required: yes (existing ADR281).
ADR path: backlog/decisions/281-primary-agent-multi-model-code-review.md.
Reason: the existing runtime/ownership contracts are preserved; these integration
corrections introduce no new architecture.


### Bounded final outcome (2026-10-28; supersedes pending-grant status)

The human approved the consolidated66-request/3-child scope. Preflight consumed
2 failed requests (private prompt omitted schema_version), then its one manual
retry passed in3 requests with native tool results and settled usage. The one
final Console run on47093 used54 requests/3 children, native[2,1]/one chain, and
failed the result gate with live_report_incomplete_R2. R2's corrected new-ID
submission left a rejected record open; V1 accepted2 correct decisions but
replayed cached revision0 after checkpoint revision1 and did not complete.
Mapping and two unified pages exist; neither incomplete role sealed.

Exact offline rejection/correction and cached-page controls reproduce these
boundaries. No new product repair or full rerun was started. CI also has8
reproduced module-size failures; its composer-state failure passes locally.
Cumulative314requests/21children;7financialrequests unused but no permitted
retry/final invocation/fresh child remains. AC24 stays unchecked, task InProgress,
PR3286 draft. See bounded-final/report.md and outcome-review.md in the QA directory.
No merge or further live execution is authorized by this outcome.

## Bounded offline repair continuation (2026-10-28)

User authorized fixing the observed R2/V1 completion failures and the eight
module-size CI failures. Keep PR #3286 draft and AC24 unchecked. No live
qualification, provider probe, new product child launch, or budget request is
part of this continuation.

ADR required: yes (existing decision; no new ADR).
ADR path: backlog/decisions/281-primary-agent-multi-model-code-review.md
Reason: restore the existing strict submission, replay, checkpoint and completion
contracts and move task-owned integration code out of oversized modules without
changing ownership or policy.

1. Add narrow offline regression tests for rejected reviewer coverage recovery
   and accepted verifier dispositions followed by history reduction and page
   replay. Fix actionable correction/completion feedback and current-state
   recovery while preserving same-ID correction, immutable page replay, native
   permissions, request admission, accounting and the two-child limits.
2. Extract this task's additions from the eight failing modules into cohesive
   existing owners or narrow helpers. Preserve callers and lazy import behavior;
   do not raise size limits or perform unrelated cleanup. Re-run the eight exact
   ratchet cases and tests of the moved boundaries.
3. Review the changed boundaries, run relevant static/startup checks where imports
   changed, and record the offline result separately from failed live evidence.
   Update the existing draft PR only; do not infer AC24 completion from these
   tests or restart qualification.


Offline continuation outcome: completion repair3e5054c5ad and size extraction
32f40af781 are verified and independently reviewed with no actionable findings.
All17 size/no-slack/helper guards and3 startup guards pass;25 observed test failures
or errors and existing lint findings reproduce on the original candidate and are
not claimed repaired. See the [offline evidence](../qa/2026-10-18-multi-model-code-review/offline-repairs/report.md).
AC24 remains unchecked, PR3286 draft, no new live requests/product children or
budget request; no merge. This closes only the authorized offline repair scope.

### Final single-run outcome (2026-10-29 UTC)

The user authorized one final run capped at60 new requests/3fresh children after
CI, with no automatic retry, broad repair cycle or merge. Required normal PR CI
passed on0ab1e549f0; the separate Windows GGUF timeout remains disclosed.
The runner ran once and failed `live_report_incomplete_V1`: both reviewers sealed
and were presented, but the accepted fresh primary failed locally on its next
request after a code_review_read model response, before physical request25 or V1
launch. All24 actual sends returnedHTTP200; the UI400 classification is not a
server400 receipt. Exact malformed field remains unproved by retained metadata.
The accounting ledger subsequently paused usage_unknown; resource shutdown was
clean. Actual new spend24requests/2children; cumulative338/23. The36unspent
requests/1unlaunched child do not authorize another invocation. AC24 stays open,
task InProgress, PR3286 draft. No new repair/live retry/budget request or merge.
See QA/final60/report.md under the task's existing QA directory. Existing ADR281
applies; no architectural change.

### Verified offline continuation repair (2026-10-29)

The authorized narrow follow-up reproduces the continuation request rejection
through the actual loop, Console preparation, dispatcher projection and strict
custom-hosted payload builder. Native deferrals now omit the fenced-only name;
the prepared serializer also drops the internal instruction-delivery row key
without mutating canonical rows. Each defect has a retained failing control;
the complete path now defers without dispatch, executes the reissued read once
and completes. No validation/permission/accounting/budget policy changed.

The final targeted selection has 191 passes and three independently reproduced
pre-existing profile-selection failures. Formatting/whitespace pass; no new lint
findings; scoped review has no actionable findings. See the task QA directory's
continuation-offline/report.md. ADR required: no new ADR; ADR-069 and ADR-281 apply.
No live calls, children, budget request or retry; AC24 remains open and PR draft.
The original live request-body attribution remains unproved.

### Continuation-repaired final qualification outcome (2026-10-29 UTC)

All CI passed on a1b4b8dac9 after one explicitly authorized performance-job
rerun. One live qualification used48requests/3children (cumulative386/26),
with native[2,1], both originals presented and the local continuation repaired.
Haiku's first3dispositions were accepted, then its changed new-ID resubmission
was refused disposition_conflict; completion refused unresolved rejections.
V1 stayed unsealed despite complete mapping and readable unified pages, so
the runner exited2 and AC24 remains unchecked. Supplied checkpoint guidance
was verified; no new source defect established. One narrow existing conflict
control passed; scoped outcome review agrees. Accounting settled and cleanup
completed. Preserve PR3286 draft/InProgress; no retry, new budget or merge.
See QA/continuation-final60/report.md under the task's QA directory. Existing
ADR281 applies; no architectural change.

### Final handoff qualification outcome (2026-10-29 UTC)

Normal same-head CI passed onffe3cfdf5a after one authorized UI25 rerun.
The one approved60-request/3-child qualification used44requests/3children,
cumulative457/31, under native[2,1]. Both reviewer reports sealed and were
presented; V1 accepted four dispositions then became stuck after three consecutive
invalid_source_range reads, before the overall deadline. V1 never sealed; final
expansion/unified reads refused. The later primary stream lacked a terminal
marker and its last usage reservation remained uncertain. AC24/DoD stay open,
PR3286 draft, no retry or merge. The16unused requests retire and the follow-up
was deleted. See QA/final-handoff60/report.md and its independent outcome review.
Existing ADR281; no new architectural decision or production repair.
## Offline V1 source-range/completion follow-up — 2026-10-29

Reproduce the retained worktree/base/worktree invalid-range sequence through real
source, tool and native/fenced runtime boundaries. Add bounded range-recovery
feedback only after source authority/reference checks, preserve that feedback in
the model conversation, and guide accepted dispositions through the existing live
checkpoint and verifier_complete flow. Keep strict range refusal, failure limits,
immutable decisions, terminal sealing and accounting unchanged. Verify captured
empty/short files, revoked/foreign references, capacity refusal, native/fenced
recovery and terminal sealing offline. Keep AC24 open and PR #3286 draft; no live
requests, qualification, new children, budget or merge.

ADR required: no new ADR. ADR path: backlog/decisions/281-primary-agent-multi-model-code-review.md.
Reason: bounded diagnostic feedback and guidance within existing authority and completion contracts.

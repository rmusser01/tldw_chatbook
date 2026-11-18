# TASK-35582 final whole-branch review

Reviewed range: `b07fc0425288fb51d17cde5f745051720a7c9db6..e03c177f91` (25 commits).
Date: 2026-10-26. **Not ready to merge: 0 Critical, 4 Important, 2 Minor.**

## Scope and strengths

Cross-slice source/requirements review using the requested code-reviewer rubric. Read the governing task, specification, ADR-240/amendments, plan global constraints/contracts, QA README/manifest and all 39 rulings. Traced native Console composition, role/runtime gates, prepared request admission, source capture, publication/cleanup, blinded verification, process receipts, continuation acceptance, native reports, Settings and packaged guides. Inspected connected existing skill expansion and runtime wait handling. Earlier approvals are not assumed to prove current correctness; historical QA reports remain unchanged.

The implementation keeps existing fleet, routing, automatic-work, scratch and process owners. Strong details include closed child catalogs plus runtime gates, exact target/role associations, immutable original submissions, opaque exact-repeat projection, final prepared-request capacity admission, and honest unavailable receipts when the stock executor cannot enforce aggregate quota. QA distinguishes scripted judgments from live evidence and retains earlier failures/resource warnings.

## Critical

None found.

## Important

### I1 — Revalidate source authority before delivering model-bound reports

**Location:** `tldw_chatbook/Agents/code_review_service.py:1129–1144`, especially line 1142.

The non-source `read` branch calls `_read_report` after actor/grant/store checks but never enters `_source_context` or checks the current admitted root guard. Grant validation checks accepted user provenance, not current workspace authority. Raw reports and structured records may contain captured source. Revocation therefore denies `view=source` while the same primary continues receiving report bodies. Native report rendering correctly enters `_source_context`; this is a specific model-tool bypass.

**Reproduction:** A fixture-backed probe prepared a real capture, bound a reviewer, and stored `SENSITIVE_REPORT_CANARY` with `capture_review_output`. It set the admitted root guard to false, then invoked the public provider under the same primary actor. Observed:

```text
REVOKED_REPORT_READ: True CANARY_DELIVERED: True
REVOKED_SOURCE_READ: False source_authority_unavailable
```

The assertion that the report must refuse failed. The probe wrapped the existing `test_prepare_freezes_routes_without_model_or_settings_writes` fixture invocation in memory; no repository test was edited or permission check replaced with an allow result.

**Impact/requirements:** Retained references bypass revocation and may feed captured content into another model request. The spec requires current workspace authority on every read and invalidates read grants on revocation; AC5/6/22 are not fully satisfied.

**Fix:** Apply shared current source/permission authority before report/index/summary delivery and cursor replay, keeping callbacks outside publication locks. Cover raw and structured reads after guard revocation and newly excluded paths.

### I2 — Restore the nonblocking close fence before taking the publication lock

**Location:** `tldw_chatbook/Agents/code_review_store.py:1443–1453`.

`close()` now acquires `_tab.lock` before setting `closed` or revoking scratch. Writers hold the same lock across `os.write`. A stalled writer therefore prevents close from publishing its fence, and may publish before the pending close acquires the lock. This regresses the earlier slice's lock-independent implementation and the preserved explicit ruling.

**Reproduction:**

```sh
.venv/bin/python -m pytest Tests/Agents/test_code_review_store.py::test_store_close_revokes_without_waiting_for_stalled_writer -q -p no:cacheprovider --basetemp=/private/tmp/task35582-final-close-repro
```

**1 failed in 1.44s:** `closing.result(timeout=1)` raises `TimeoutError` at test line 995. The held writer is the explicit event barrier in the existing test. Commit `146082cb60` sets the event before any publication lock; current code does not.

**Impact/requirements:** The review owner cannot honor prompt close/revocation during stalled native I/O; AC21 and the close-before-drain ruling are regressed. This is a current product defect, separate from baseline/profile failures. Console also has a direct scratch-manager close path: the reproduction establishes the `ReviewStore.close` contract failure, not that every Console close hangs.

**Fix:** Publish the fence and revoke the exact generation first; defer lock-owned check signaling/reservation bookkeeping to existing drain/cleanup ownership. Preserve accepted/uncertain process charges. Retain both the held-writer test and check-close acceptance races.

### I3 — Advertised natural-language and inline-skill requests omit review tools

**Location:** `tldw_chatbook/Chat/console_code_review.py:66` and `:1041–1042`.

The parser recognizes initial review only when accepted text starts with `review` after optional `please`. Without an existing offer, other requests return no provider, including guide discovery. The spec's literal `do multi-model code review` and the shipped `$multi-model-review Review my changes` invocation fail. Skill expansion changes provider messages (`console_chat_controller.py:21810–21852`); review composition uses the original stored accepted USER message (`:27026–27040`), so trusting/rendering the skill does not fix the integration.

Executing the actual parser definitions extracted from current source yielded:

```text
'Review my changes' -> ['review']
'Review my changes and verify' -> ['review', 'verify']
'do multi-model code review' -> []
'Can you review my changes?' -> []
'$multi-model-review Review my changes' -> []
```

**Impact/requirements:** The explicitly specified example and optional alias cannot start the workflow in a fresh chat. AC1 is checked despite this integration failure. Fail-closed behavior is safe, but incomplete. Skill tests verify trust/import/render metadata without a real native review journey.

**Fix:** Bind intent from accepted native user provenance for supported natural-language requests and trusted inline invocation arguments. Do not grant stages from arbitrary expanded skill/source text. Add real Console tests for the spec wording and leading trusted skill, with untrusted/qualified-verification negative controls.

### I4 — A successful verifier can never receive a complete report envelope

**Location:** `tldw_chatbook/Agents/code_review_store.py:1101–1103`; `tldw_chatbook/Agents/code_review_service.py:2037–2055`.

`verifier_complete` is an accepted append but never sets completion intent or validates a verifier completion state. `_report_envelope` applies reviewer-only coverage, empty-findings and completion-request conditions to V1. Verifiers cannot submit reviewer coverage, so complete successful verification always reports `incomplete` with no seal receipt. Native unified pages and exports include this envelope.

**Reproduction:** A fixture-backed probe submitted an original and valid reviewer completion, delivered its blinded unit, accepted a confirmed disposition with source evidence, submitted `verifier_complete`, captured output and settled V1 as done. The resulting envelope was:

```text
{'terminal_status': 'done', 'report_status': 'incomplete',
 'capture_status': 'complete', 'seal_receipt': None, 'coverage': None}
```

The assertion that this complete verifier should seal failed. Existing journey assertions check sealed R1/R2 envelopes only (`Tests/Chat/test_console_multi_model_review_journey.py:222–223`). No live calls were involved.

**Impact/requirements:** Host receipts cannot distinguish completed verification from genuine incompleteness. Completion/coverage requirements and AC14 remain incomplete.

**Fix:** Record verifier completion intent and derive completeness from its own unit/disposition ledger, separately from reviewer coverage; require actual terminal/capture receipts and valid accounting. Cover full, premature, missing-decision and failed/cancelled completion. A reasoned unresolved disposition may be complete accounting; a host-filled unprocessed unit must remain incomplete.

## Minor

### M1 — Saved-team assertion runs after restoration

**Location:** `Tests/Chat/test_console_multi_model_review_journey.py:315`, manifest claim at `:322`; connected cleanup at `Tests/Chat/test_console_code_review_continuation.py:572–575`.

`run_console` restores the saved configuration in `finally` before the outer assertion. Thus it cannot detect a runtime rewrite and does not justify this journey's `saved_team_unchanged=True`. The deferred slice-11 Minor is valid. Compare the configured fixture team during `after_schedule`/collection, before restoration, and retain a separate cleanup assertion. Existing lower-level coverage means this is not evidence of a proven production rewrite.

### M2 — Packaged guide uses the wrong wait selector

**Location:** `tldw_chatbook/Agents/code_review_guides/review.md:76`.

The guide says `wait_agents({"run_ids":["RUN"]})`. Actual schema accepts `ids` containing fleet handle IDs (`tool_catalog.py:394–418`). Runtime ignores `run_ids` and waits on all children (`agent_runtime.py:3067–3076`). Use `{"ids":["HANDLE"]}` or intentionally omit the selector. Otherwise unrelated ordinary children can consume the primary's remaining finite wait time.

## Requirements and readiness

The branch substantially implements the approved ownership and bounds, but the four Important findings invalidate an unqualified AC1–22/25 completion statement. Entry integration, current read authority, close fencing and verifier completeness need correction. Keep TASK-35582 **In Progress**. AC23 remains open and now includes an independently reproduced current implementation regression; retain separate attribution of earlier baseline/profile failures. AC24 remains unverified: no saved team/presets, zero recorded live calls/children. This review did not invent destinations or run providers.

Seven retained Workspace foreign-thread handles and the earlier FD warning remain qualification debt. Stock Raw CLI checks correctly return unavailable before copy/launch. The external finite fault-controlled executor proves the protocol, not stock enforcement or live execution; deterministic judgments do not establish live model usefulness. All 39 rulings and their stated costs remain retained. I2 directly violates the nonblocking-close ruling.

**Ready to merge: No.** Fix I1–I4 and record focused regressions at the actual read/close/entry/completion seams. Update current QA/AC status without rewriting historical reports. Address or explicitly retain M1/M2. Required regression/resource and live qualification remain necessary before Done.

## Execution and limits

No full suite, routine gate reruns, push/merge, new task, subagents, or repository implementation/test edits. Only this report was written in the checkout. Three narrow reproductions ran: the unchanged close node and two in-memory fixture probes for report revocation and verifier sealing; the parser probe executed only its real definitions. Custom probes used the existing isolated pytest profile without guard resets or warning suppression. Incidental `raw_source_selection_changed` diagnostics did not prevent the demonstrated operations; these failures occur at the new explicit assertions, not at profile admission.

I inspected production seams and selected connected tests; I did not re-execute the 1.8 MB historical package, requalify providers, rebuild distributions or claim exhaustive test coverage. Scoped historical evidence was read as historical evidence rather than proof all current tests pass. Final report save required a narrow sandbox escalation because the managed worktree is outside this reviewer's writable roots; no broader write was requested.

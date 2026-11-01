### Spec Compliance

- ✅ Spec compliant for Task 3 and both 2026-10-18 amendments in the exact range `75b671c001e84bf2713c380e30de3e587bcc8ae0..20f3f3e24b5c9ba5a58f6793af1cad0ad76034ce`. The production changes stay in the approved controller reseal, shared factory/caller and four reference-classification paths; the two existing test files, ADR097 clarification and Task3 QA/evidence supply the requested supporting changes.
- ✅ The reseal uses the existing canonical preflight projection and the same three locator exclusions on both sides. It retains historical-row equality, current-text-only checks and exact current saved-source checks; the request is replaced only at its current active row and corresponding provenance. Evidence: `tldw_chatbook/Chat/console_chat_controller.py:10573`, `:10603`, `:10638`; canonical conflicting-marker refusal is at `:1814`.
- ✅ The factory comparison normalizes only the caller-supplied thinking-owner spelling when the canonical marker is absent and the exact string owner occurs in an admitted thinking group. It compares the complete frozen row against admitted rows and returns the original incoming rows without rewriting their provider values. Unknown/missing/changed/conflicting controls cover the requested negative cases. Evidence: `tldw_chatbook/Chat/console_agent_bridge.py:2838`, `:2847`, `:2862`, `:2873`, `:2887`; `Tests/Chat/test_console_archive_recovery_followup.py:771`. The sole production caller passes the fixed app-owned alias from the adapter: bridge `:3202`, `:3341`.
- ✅ Reference creation, lookup, durable comparison and node matching share the exact typed classification. Only MESSAGE_REWRITE + outer THINKING artifact + direct ProviderArtifact TOOL_CALL first input becomes tool_call; ordinary thinking retains thinking. No generic suffix allowance, migration or legacy-node inference is added. Evidence: `tldw_chatbook/Chat/console_trace_service.py:6347`, `:5310`, `:5380`, `:5468`, `:5522`; `Tests/Chat/test_console_trace_service.py:4952`; `backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md:631`.
- ✅ The actual shared Library factory is used by the existing native captured fixture. Its four Direct/RAG × plain/thinking recovery cases require exact confirmation, retain the saved transcript/envelopes and prior captured rows/reconstruction, assert historical/current saved pins, and preserve legitimate historical thinking omission under current-exchange replay. Evidence: `Tests/Chat/test_console_archive_recovery_followup.py:354`, `:594`. Continuation compatibility and conflicting-marker refusal are covered at `:844`.
- ⚠️ The task-level implementation is reviewable and compliant; whole-feature live qualification is not established by this diff. Root must perform its separately authorized fresh native live phase before claiming AC1 or actual-server reasoning-content qualification. The four offline cases use frozen synthetic transport capabilities and three supported start-anchored thinking blocks, whereas the earlier live diagnosis involved five blocks. Evidence: `Docs/superpowers/qa/console/2026-10-18-archive-capture-history-projection.md:13`, `:47`, `:49`.
- ⚠️ Frozen destination/Persona/kill-switch/one-shot authority contracts remain unchanged in this supplemental range; their complete earlier qualification is outside this Task3 gate and was not reopened. The connected reseal caller retains its policy/kill-switch/native-tool gates before obtaining the follow-up context: `tldw_chatbook/Chat/console_chat_controller.py:26589`. The current native success test waits at the exact target card before calling the real recovery coordinator: `Tests/Chat/test_console_archive_recovery_followup.py:594`.

### Strengths

- The correction reuses existing canonical projection and provenance machinery rather than weakening the runtime or completed-tool validator. The classifier is one small shared function, and all four reference paths use it. Evidence: controller `:10603`; trace service `:6347`, `:5310`, `:5380`, `:5468`, `:5522`.
- Typed provenance still validates the attachment and exact causal owner before the classifier can observe the descriptor: `tldw_chatbook/Chat/console_trace_provenance.py:555`, `:641`, `:695`, `:707`, `:715`. This makes the helper's first-input access safe and prevents the new classification from accepting an untyped thinking artifact.
- The new SQLite service control covers production reference creation, all four identity paths, repeated prefix reuse through both existing and fresh service instances, ordinary thinking and refusal of a deliberately persisted legacy thinking node. Evidence: `Tests/Chat/test_console_trace_service.py:4952`.
- The native regression checks custody and immutability as well as successful restore. It also preserves the distinction between aggregate tool-call thinking and the final replay group. Evidence: `Tests/Chat/test_console_archive_recovery_followup.py:594`; the reported covering command additionally includes the existing active-versus-aggregate and masked-prefix controls: `Docs/superpowers/qa/console/2026-10-18-archive-capture-history-evidence/final-focused-command.json:1`.
- Evidence distinguishes setup failures, product REDs, later amended-scope failures, test-only errors, final GREEN and existing static-analysis debt. It does not convert those earlier failures into passing evidence or claim live completion. Evidence: `Docs/superpowers/qa/console/2026-10-18-archive-capture-history-projection.md:19`, `:21`, `:29`, `:47`.

### Issues

#### Critical (Must Fix)

- None. Count: **0**.

#### Important (Should Fix)

- None. Count: **0**.

#### Minor (Nice to Have)

- None. Count: **0**.

### Named Surrounding Checks

- **Risk: the canonical projection could discard provider values or tolerate contradictory ownership.** Inspected the unchanged `_flatten_preflight_messages` implementation at controller `:1814`: it handles only the two reserved owner markers, refuses conflicting string owners and maps the association to the native locator. The changed reseal excludes only its existing three locator fields and preserves its original before/after and current-source guards.
- **Risk: a loosely supplied alias could become a general saved-owner matching API.** Inspected the sole production factory construction/call sites and the adapter's alias producer: bridge `:3078`, `:3202`, `:3341`. The production caller supplies its fixed private alias; admitted thinking-group membership and full frozen-row equality gate reuse. Inspected the remainder of `build` at `:2887` because the diff hunk ends mid-function: it retains incoming rows, frozen capture policy and the existing request construction.
- **Risk: historical replay filtering could be bypassed to make the new test pass.** Inspected the adapter's sidecar handoff at bridge `:3202` and the unchanged prepared-request thinking serialization/attachment logic at `tldw_chatbook/Chat/console_prepared_request.py:1280`. The change preserves the saved descriptor before gateway preparation; it does not force historical groups into replay. The native regression expressly requires their omission for current-exchange mode.
- **Risk: the component classifier could promote malformed or contradictory rewrites or index empty inputs.** Inspected `DerivedTraceProvenance.__post_init__` and MESSAGE_REWRITE validation at provenance `:555`, `:695`, `:707`, `:715`: inputs must be nonempty typed descriptors; rewrite sidecars must be exact attachments of the first input; a thinking rewrite artifact requires a typed thinking attachment.
- **Risk: the new component kind could change materialized artifact disclosure or legacy identity.** The diff retains the same `_store_artifact` calls, values and artifact policies in service `:5468`, and the same durable credential/PII projection in `:5380`; only component-kind selection changes. The SQLite regression rejects legacy thinking nodes in both node matching and durable matching at test service `:4952`.
- **Risk: a context-cut reseal return could replace other immutable request fields.** Inspected the remainder of the reseal at controller `:10638` because the diff hunk ends mid-function. It replaces only `active_request` and that row's provenance; source-proof failure still becomes the existing controlled pre-reservation trace refusal.
- **Risk: the reported GREEN could mask failures or refer only to an earlier semantic version.** Read the retained command/exit/output evidence: `final-focused-command.json:1` and `final-focused.log:1` report exit 0 / 41 passed before formatting-only cleanup; `final-style-smoke-command.json:1` and `final-style-smoke.log:1` report exit 0 / 10 passed against final bytes for the directly changed alias/classification controls. Final outputs contain no warnings or teardown errors. No tests were rerun.
- **Risk: existing static debt could be presented as clean lint/format results.** The retained static summary explicitly records unchanged Ruff diagnostics and formatter debt in controller/bridge/service-test, with no newly introduced diagnostic signature; service/follow-up formatting and authored whitespace pass. Evidence: `Docs/superpowers/qa/console/2026-10-18-archive-capture-history-evidence/final-static-summary.json:1`, `final-static-clean.log:1`. Root's independently verified manifest/guard hashes were not redundantly recomputed.
- Read the complete brief, implementer report, source-complete companion and retained diagnoses. Read current architecture README/Console documentation as context; no current-checkout implementation was transplanted into this older worktree. No app, model, server, network, full suite, subagent, git-state operation or source/index/HEAD change was performed.

### Assessment

**Task quality:** Approved.

**Reasoning:** The three small corrections address the reproduced composition mismatches while preserving existing ownership, typed provenance, replay and disclosure boundaries. The regression and SQLite controls exercise the actual changed seams and preserve prior trace evidence; no actionable task-scoped defect was found.

**Counts:** Critical **0**, Important **0**, Minor **0**. **Blocking implementation concern:** none.

**Unresolved evidence limits:** Fresh live Direct/RAG qualification remains root-owned and pending; the five unused inference requests are not consumed or qualified by this review. Expired earlier temporary diagnostics, historical unsupported-surface failure, aggregate FD growth, provider/optional-RAG errors, dictionary second-send control and Next Send preview/estimate omission remain the explicitly preserved earlier limits.

Skipped rerunning tests, app/network/model execution, broad repository review and integration. Actual-server qualification remains the material open risk.

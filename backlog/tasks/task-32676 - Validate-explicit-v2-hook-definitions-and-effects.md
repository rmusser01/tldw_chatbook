---
id: TASK-32676
title: Validate explicit v2 hook definitions and effects
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:22'
updated_date: '2026-09-16 22:02'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32645
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make richer hook declarations inspectable and deterministic while preserving the legacy six-event configuration.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Legacy hooks.hook parsing and stdout behavior remain unchanged; hooks.handler and native v2 files validate the exact event/effect/matcher/template contracts.
- [x] #2 Required success, required nonempty context and dependency-scoped requirements stay distinct, and invalid controlling requirements never disappear through optional parsing.
- [x] #3 PreToolUse phase classification is exhaustive, including mixed transformer/deny, final guards, context-only handlers and effect-free required completion.
- [x] #4 Payload and result bounds reject whole invalid batches, reserved ownership fields cannot be supplied by output, and malformed teardown/approval/Stop requirements are refused.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/148-console-run-hooks.md
Reason: Direct implementation of accepted v2 hook validation, with R41 closing nested schemas and host projection boundaries. Amend ADR163 and the expanded-hook spec before related code; preserve legacy ADR148 behavior and one existing runtime owner.

1. Read task9 brief, actual TASK32676, h1-config-schema-research.md and controlling h1-schema-ruling.md. Recheck final prior-task APIs and actual config/run_hooks loaders. Read expanded-hook spec event/effect matrix and limits. Preserve legacy hooks.hook parsing, live settings, stdout handling and event names. Notify controller before additional paths or changes to a settled contract.
2. Amend existing ADR163/spec with R41 host-only dependency phase context, strict raw decoder, matcher availability, closed child limits/continuation/env shapes and documented event projection. Distinguish H1 structural validation from H2 launch, H3/H4 producer authority and M1/M2/M3/H6 raw transport qualification.
3. Establish behavioral RED in Tests/Agents/test_hooks_v2_validation.py through the public parsing/phase boundary with an ordinary valid declaration control. Test the mixed transformer/deny phase and controlling requirements before implementation; do not count incidental fixture/import errors as evidence of the intended behavior.
4. Create Agents/hooks_v2/__init__.py, models.py, validation.py and matching.py. Use frozen normalized models, closed fields/effects and bounded whole-document validation. Keep validation independent of Plugins and preserve caller-owned input from normalization side effects or mutable aliases. Implement exact13event matrix, duplicate IDs/keys, handler type-specific fields, forbidden output ownership and illegal teardown/approval/Stop requirements.
5. Implement separate raw UTF8/result and object-shape gates. Bound raw output at16KiB before parse, reject nested duplicate keys/nonfinite values/trailing/nonobject data, and keep successful empty command output distinct. Enforce declared effects, context lifetime/require_context, R41 child-cap shapes and continuation4KiB bound; runtime intersection and combined settlement limits remain their actual owner responsibilities.
6. Implement host-only dependency_required phase context, structured matchers and one-pass typed MCP templates against the documented event projection. Preserve unavailable canonical-operation distinctions and support SessionStart.reason, actual catalog identity, lowercase env names and declared-variable references. Missing fields fail expansion or leave controlling requirements unsatisfied; never infer identity from tool-result data.
7. Integrate hooks.handler parsing/config projection separately from legacy hooks.hook through run_hooks.py/config.py as needed. Master disable suppresses execution without deleting required or invalid-required state. Keep the default config path lazy and avoid importing Plugins or creating a v2 execution engine. Cover mixed legacy/v2 load-save behavior if those paths change.
8. Run Tests/Agents/test_hooks_v2_validation.py and Tests/Agents/test_run_hooks.py, plus exact modified config neighbors. Include exhaustive phases/events, supported nonmatch versus unavailable required matcher, raw/object distinctions, malformed/oversized batches, positive SessionStart.reason/lowercase-env cases and real legacy unknown/nonJSON stdout. Targeted only; no full sweep. Recheck startup only if import changes justify it. Freeze code during the final runs.
9. Update author/config docs within the modified spec/config surfaces, then report actual commands/output, changed paths, public schema APIs, raw provenance limits and H2-H6 handoffs in task-9-report.md. Run syntax/Ruff/Black/whitespace with established shared-file baseline handling. Controller owns commit, independent spec/quality review, AC/notes and Done only after all gates pass.

10. Include Tests/Chat/test_run_hooks_metadata.py as the explicit additional test path for the existing real configuration load/save production entry. Prove mixed legacy/v2 preservation across hooks.enabled saves, including required and invalid-required v2 state while disabled. This implements existing R41 and AC1-2; do not expand to unrelated settings or runtime wiring.

11. Fix independent-review H1-R1 through H1-R4 within the existing authorized H1 source/test/contract paths before dependent runtime work: preserve bounded explicit-required versus event-policy invalid-state scope without promoting optional require_context; reject forbidden dependency_required event combinations; translate out-of-range numeric durations into validation refusal with retained config state; enforce byte/node/depth budgets during object/template copying before excessive allocation. Establish actual public-entry RED and positive controls, complete the already-planned invalid-required mixed save/load case, then freeze and rerun the exact three H1 feature/legacy/config test files plus static baseline checks. Append the fix report and corrected H2 config-state API; controller owns fresh scoped re-review.

12. Fix remaining H1-R1 malformed-effect metadata classification after scoped fix1 review. In Agents/run_hooks.py and Tests/Agents/test_run_hooks.py, positively validate the full supported effect-array shape before labeling a rejected entry optional, preserve recognized explicit/event controls, and retain all other malformed/unsupported shapes as unresolved. Cover types, unknown/event-incompatible effects and duplicate/invalid entries with public loader RED and optional/control positives. Run only Tests/Agents/test_run_hooks.py and the existing Tests/Chat/test_run_hooks_metadata.py neighbor for final qualification unless another changed boundary/failure justifies more; preserve unchanged Fix1 schema/matching evidence. Freeze changed files, run static baseline parity/whitespace and append Fix2 report before fresh scoped review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Explicit v2 hook declarations now validate all 13 events, declared effects, structured matchers and typed input templates independently of the existing six-event hook format. Normalized values are immutable and detached from caller data. Original command-result/native-file text uses strict UTF-8 JSON decoding with duplicate-key, size, depth and nonfinite-value refusal; decoded objects have a separate shape/provenance contract.

Invalid configuration retains bounded event/policy metadata without promoting optional context into a global requirement. Explicit required state survives master disable and real mixed legacy/v2 settings saves. Forbidden required dependency edges are rejected at approval/Stop/teardown boundaries. Numeric overflow becomes validation refusal, and object/template budgets apply incrementally before excessive materialization. Existing legacy execution/stdout behavior remains covered.

Validation: the full H1 schema/legacy/config run passed 201 tests in 6.84 seconds; the final narrow classification fix then passed the exact two loader/config files, 101 tests in 6.48 seconds. Both runs had no warnings or skips; counts overlap and are not aggregated. Nine of the eleven frozen paths were unchanged by the final fix, preserving its earlier schema/matching qualification. Syntax/new-file Ruff/Black/whitespace passed; shared existing-file Ruff/Black results matched baseline (config retains 164 pre-existing Ruff diagnostics). Eleven source/test/contract hashes remained frozen across the run and matched the committed files. The first fix regressions established 11 failures with two valid controls; the final narrow fix established five malformed-effect failures with five passing controls before correction; the original scaffolding RED only established missing imports and is recorded as such. No full suite or new startup/platform qualification was run.

Architecture: direct implementation and amendment of backlog/decisions/163-expanded-console-hook-runtime.md under the accepted R41 schema contract; backlog/decisions/148-console-run-hooks.md remains the legacy authority. Changed areas are Agents/hooks_v2, the existing run_hooks/config projection, targeted Agent/Chat config tests and the expanded-hook spec. These parser tests do not establish command/MCP transport ownership, actual event producer identity, permission enforcement, child authority intersection or scheduler settlement; their public consumer contracts are documented in the spec and implementation report.

Independent spec/quality review closed after two focused fix rounds: all H1-R1 through H1-R4 addressed, with no new Critical/Important or out-of-scope findings. Remaining producer/transport/runtime verification belongs to the documented consumer boundaries rather than this pure schema task. Historical RED, shared static baselines and existing startup/platform limitations remain explicitly recorded.
<!-- SECTION:NOTES:END -->

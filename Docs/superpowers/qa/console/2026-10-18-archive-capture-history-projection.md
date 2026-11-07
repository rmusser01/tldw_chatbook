# Task 3: captured thinking-history archive recovery — 2026-10-18

**DONE_WITH_CONCERNS — implementation complete; root owns independent review and live qualification.** Review baseline is `75b671c001e84bf2713c380e30de3e587bcc8ae0`; the corrective commit is recorded in the SDD Task 3 report. All four plain/thinking Direct/RAG native recovery cases pass through real SQLite, native admission/controller/bridge/gateway and exact confirmation. No live model result or whole-feature completion is claimed; root retains AC1 and the five unused inference requests.

The correction preserves ownership across three existing composition boundaries:

1. Reseal compares admitted history through the existing canonical preflight projection, excluding only the same three native/persisted locator fields on each side. Historical equality, conflicting-marker refusal and exact current saved-source/text-only checks remain.
2. The agent request factory accepts the adapter's explicit thinking-owner alias only when its exact string owner belongs to an admitted canonical thinking group and the complete row, including every nested multimodal value, matches. Unknown/missing/contradictory owners and changed fields gain no saved authority. Gateway replay selection is unchanged: current-exchange mode legitimately omits historical thinking while retaining its saved assistant descriptor.
3. The four service reference paths share one exact classification: a validated MESSAGE_REWRITE with outer THINKING artifact and direct TOOL_CALL artifact first input retains the tool_call surface kind. Its typed transform, complete serialized bytes, attachment inputs and disclosure policy stay unchanged. Ordinary thinking remains thinking. Legacy stored thinking nodes remain immutable and ineligible for retrospective promotion.

Root recorded both scope amendments, Backlog plan and the [ADR-097 clarification](../../../../backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md#clarification-recorded-2026-10-18-thinking-serialized-tool-call-surface-identity) before each source continuation. ADR-090/097/166 govern; no new ADR, schema, service, config, dependency or authority contract was introduced. The root-authored ADR clarification is the sole approved metadata staging exception; plan/task/SDD ledger edits remain excluded. No earlier review wave was restarted.

## Verification

All commands ran in the isolated worktree using the existing installed interpreter, `PYTHONPATH=.`, `PYTHONDONTWRITEBYTECODE=1`, explicit DNS denial and the repository socket guards. Only the documented external `chat_api_call_fn` transport is fake. It returns real native tool_calls JSON and supported start-anchored thinking under frozen llama_cpp/Gemma displayable-v1/native-tool capabilities, with no preservation support. Search parses and persists two aggregate tool-call thinking blocks and one final block. The final replay group contains only the final block.

[Evidence](2026-10-18-archive-capture-history-evidence/) contains exact argv/cwd/environment/exits and phase source hashes in `*-command.json`; raw subprocess output is unchanged. `run-phase.py` and `static-compare.py` retain the executable capture/check procedures. `source-hashes.json` covers final source/tests/ADR plus unchanged connected files; `sha256-manifest.json` hashes the complete evidence package and authored QA/source/ADR.

- `final-focused`: **41 passed in 38.47 s**, clean teardown and network guards. This covers all four native restore modes, exact historical/current saved pins, unchanged transcript/envelopes and predecessor rows/reconstruction, typed transform composition, continuation/matching/conflicting owners, unknown/missing/foreign/current/history/non-text/multimodal refusal, authority refusal/Stop cleanup, all four reference paths with warm/cold repeated prefix reuse, ordinary thinking/legacy refusal, strict completed-tool positive/cold/non-tool/range controls, active versus aggregate thinking and masked-prefix refusal.
- Formatting-only cleanup followed that run. `final-style-smoke`: **10 passed in 1.88 s** against final file bytes, covering the directly touched alias/classification cases. No semantic source change followed the 41-test run.
- `final-static-clean`: exit 0. Baseline/current diagnostic signatures are identical: controller 191, bridge 29, service 33, existing service-test file 15. The follow-up test has zero diagnostics. Service/follow-up formatting passes; the existing controller/bridge/service-test formatting debt remains. The newly appended service test is independently formatted.
- The strict service functions `_validate_completed_tool_turn`, `_completed_tool_turn_parent`, `prepare_current_surface_delta`, `persist_request` and `_validate_replacement_projection` are byte-for-byte unchanged against baseline. Prepared-request thinking serialization, typed provenance, provider gateway, trace runtime and shared Library runtime are unchanged.
- Scoped authored whitespace checks pass. Nine raw logs retain 26 original trailing-whitespace lines, inventoried in `raw-whitespace.json`; raw output was not trimmed.

## Preserved RED and diagnostic phases

| Phase | Exit | Evidence and interpretation |
| --- | --- | --- |
| `setup-native` | 1 | Initial recreated fixture omitted native-tool capability and hit the HTTP client guard. Harness failure only; no socket/DNS egress. |
| `red-native` | 1 | Original archive_trace_admitted_rows reproduced; old fixture cleanup order also failed. Both raw failures retained. |
| `red-native-clean` | 1 | Same original refusal with zero successor capture/transport and unchanged predecessor; native Cancel before Discard gives clean teardown. |
| `green-focused` | 1 | 15 passed, 4 failed, 2 teardown errors. Two test-only counter-placement errors were corrected; thinking cases exposed the later strict unsupported_surface_change. |
| `surface-probe` / `owner-transition-probe` | 1 / 1 | Transparent read-only observers retained exact reference state and owner transitions; each failed at the later product boundary with its unknown-reservation cleanup limitation. |
| `green-thinking-correction` | 1 | Exact alias correction restores the saved witness; both thinking cases then reproduce completed_tool_turn_range from prior thinking-labeled tool-call artifacts. Root recorded final classification scope before implementation. |
| `alias-controls` → `alias-controls-final` | 1 → 0 | Initial frozen-JSON versus ordinary-list test assertion corrected; 8 pass. No production relaxation. |
| `classification-red` | 1 | Real SQLite shows thinking versus expected tool_call; ordinary-thinking control passes. |
| `classification-green` → `classification-green-final` | 1 → 0 | Classification works; legacy fixture initially omitted the existing required content_kind constructor argument. Corrected harness gives 2 pass. |
| `native-green` | 0 | All 4 complete native recovery cases pass in 11.15 s. |
| `final-static` → `final-static-clean` | 1 → 0 | One dict-literal test style issue and touched service formatting corrected; existing diagnostic signatures remain unchanged. |

`owner-transitions.json` records the same historical owner at each stage: admitted `_tldw_thinking_owner` and saved revision with one compatible final block; adapter `_tldw_call_thinking_owner`; the former factory fallback to ACTIVE_REQUEST artifact; and later legitimate zero-group replay filtering under current mode. These distinct transitions are not equated. `surface-probe.json` records the original later service refusal with prefix 2, incoming_changed 2, active_changed 4 and missing completed-tool witness. `retained-downstream-diagnosis.md` preserves the read-only diagnosis and parent-directed scope pauses; it is historical evidence, not current completion status.

## Self-review and limits

The diff changes only the approved reseal, factory/caller and exact reference classification, two existing test files, root's ADR clarification and this narrow QA/evidence. Source is compared completely across the single validated owner alias; no role/content-only or arbitrary metadata stripping was added. The component classifier does not permit generic thinking suffixes, alter typed THINKING provenance, change credential/PII storage or infer legacy ownership. Strict completed-tool validators and all source/history/authority gates remain intact. No actionable self-review issue remains.

The older Task 3 diagnosis is preserved byte-for-byte as `retained-task3-diagnosis.md`; its expired /private/tmp setup/probe artifacts were not reverified. Fresh phases are separately retained. Earlier Task 1/2 qualifications and limitations remain: historical unsupported-surface incident, pre-correction dictionary second-send control, aggregate FD growth, optional staged-RAG capture errors and Next Send preview/estimate omission are outside this correction. The fresh later refusals above do not retrospectively explain that historical incident.

No live app/model/server/network, subagent, full-suite, original-checkout edit, push, PR, merge or cleanup was performed. Root's independent exact-range review and fresh disposable live qualification remain pending. Existing legacy thinking-labeled tool surfaces remain subject to strict refusal; this correction applies to new references and does not rewrite prior captures.

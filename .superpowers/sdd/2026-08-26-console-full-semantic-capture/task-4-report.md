# Task 4 implementation report

## Outcome

Implemented the ADR-089 shared Console capture-policy controls and governed
per-call exchange export in commit `d685a90009` (`feat(console): expose
governed full capture controls`) and hardened the independently reviewed
surface in fix-round commit `478a9f1bae` (`fix(console): harden capture policy
disclosure`), then closed the remaining scoped findings in fix-round commit
`26218ae5aa` (`fix(console): close remaining capture review gaps`). Fix-round-3
commit `b53bf54631` (`fix(console): keep capture off as kill switch`) removes the
new acknowledgement regression without weakening enabled Global Full writes.
The Backlog task remains **In Progress** and all acceptance
criteria remain unchecked for independent review.

## TDD evidence

- Exchange projection: RED was `ModuleNotFoundError` for
  `console_exchange_export`; GREEN was 5/5, later 6/6 after adding the durable
  production-shaped sentinel inspection.
- Shared Trace labels/warning: RED was the missing public label/copy exports;
  GREEN was the focused Trace warning test 1/1.
- Exchange export modal: RED was the missing dialog module; GREEN was 5/5 for
  Safe unavailability, destinations, repeat Full confirmations, overwrite,
  atomic write, revision fences, and compact geometry.
- Capture policy modal: RED was the missing dialog module; GREEN was 6/6, then
  7/7 after a focused RED proved Capture Off incorrectly blocked Safe edits
  and did not warn for dormant conversation Full. The corrected case is 1/1.
- Inspector wiring: the immutable-target/global-owner regression is GREEN;
  the focused Inspector/loader gate reached 49 cases. A stale assertion using
  an unsupported Textual attribute selector failed, was corrected to inspect
  mounted Button IDs, and the focused case passed 1/1.
- Live/imported Trace: the first full gate was 43/44 because the legacy launch
  harness intentionally supplied a store-only runtime. The compatibility
  seam was narrowed without changing production controller wiring; the failed
  case then passed 1/1 and the later 80x24 aggregate passed.
- Settings/config: initial collection RED exposed an eager `Chat` package
  circular import from `config.py`. The authorized deferred canonical enum
  import made `Tests/test_config_save_settings_semantics.py` GREEN 9/9.
  Focused structured outcome/confirmation assertions passed 4/4. The full
  Settings/config/layout gate was 377/378 with one stale exhaustive ownership
  tuple; after adding the existing rail key and the two new capture keys, that
  case passed 1/1.
- Production sentinel inspection: initial RED assumed DB row order; immutable
  `run_tag` indexing fixed the inspection and it passed 1/1.

### Fix round 1 RED/GREEN

- Global Full acknowledgement: RED collection failed because the distinct
  restart-aware modal did not exist; GREEN requires its acknowledgement
  checkbox before enabling the fixed-action confirm button at 80x24.
- Policy preview: RED showed scope radios and prospective/dormant resolution
  drifting from the selected scope; GREEN synchronizes radios and resolves the
  dormant Off state through `resolve_capture_policy(enabled=True, ...)`.
- Settings coordinator: focused RED established that F9 bypassed the live
  policy owner; GREEN routes live apply through
  `apply_global_capture_settings`, including dormant-Full confirmation. The
  incumbent controller regression continues to prove Off reserves the shared
  revision and disarms every live next-send override.
- Purge repaint: RED allowed a post-commit refresh exception to replace a
  `DELETED` result; GREEN returns the authoritative result and renders
  `Deleted ...; refresh failed`.
- Export revision fence: both Full-confirmation and overwrite-confirmation
  race cases were RED because projection still ran; GREEN is 2/2 and proves
  `_project_async` is never called after either awaited revision change.
- Atomic disclosure logging: RED was `TypeError` for the missing privacy-safe
  mode; GREEN log-sink coverage proves unique path, body, and exception-value
  canaries are absent while the stable category and exception class remain.
- Production sentinel: successive REDs exposed a false storage-owner test and
  a Redacted export that retained a provider-combined automatic system body.
  GREEN drives the real gateway/controller/store/persistence/cache seam and
  inspects whole `ExchangeCapture` owners, both exports, and logs.
- Screen extraction: the first reviewed fix was 20,125 lines versus the Task 4
  base of 20,099. Moving immutable target resolution, revision binding, and
  purge repaint integration into `capture_policy_bindings.py` produced GREEN
  at 20,093 lines and 633 methods versus 20,099/633.
- Inspector cleanup removed the retired raw exchange Copy/Save prefixes,
  handlers, direct-call tests, and callable `asdict` clipboard/non-atomic file
  disclosure paths. The governed Export action is the only exchange boundary.

### Fix round 2 RED/GREEN

- Scoped Global Full acknowledgement: RED was 1 failed because a conversation
  Safe override masked the prospective effective global Full value and allowed
  persistence without the restart-aware acknowledgement. GREEN is 1/1: every
  Global + Full edit opens `GlobalFullCaptureConfirmation`, and cancellation
  leaves the mutation callback uncalled.
- Off-state prospective preview: RED was 1 failed because selecting dormant
  conversation Safe left the visible preview at dormant Full. GREEN is 1/1:
  the dialog retains the current Off/dormant state, adds the prospective
  selected dormant resolution, keeps radio state aligned, and does not mutate
  the snapshot.
- Production sentinel: the stricter rewrite first failed because its provider
  fake returned a pre-adapter Anthropic wire response at the production
  post-adapter `chat_api_call` seam, then exposed two incorrect test
  expectations for the canonical endpoint and API endpoint. GREEN is 1/1 with
  an Anthropic resolution through the real gateway/controller/store,
  `ChatPersistenceService`, in-memory ChaChaNotes SQLite, production exchange
  query, decoded cache blobs, and decoded full storage captures.

### Fix round 3 RED/GREEN

- Capture Off kill switch: RED was **1 failed** in the real Textual harness;
  cancelling the misleading `GlobalFullCaptureConfirmation` returned `None`
  and left the policy mutation uncalled when global Full was dormant. GREEN is
  **2 passed** for the new Off regression plus the retained masked
  conversation-Safe Global Full cancellation regression. Turning Off now
  applies the global disabled state without pushing an enable acknowledgement,
  while every enabled Global + Full write still requires the shared checkbox
  gate.

## Final gates

- Exact Task 4 privacy/UI matrix after fix round 3: **870 passed, 2 skipped,
  0 failed** in 432.61 seconds. Both skips were existing loopback-listener cases skipped
  because the sandbox denied listener creation.
- Production-shaped 80x24 policy/export/Inspector/live/imported/Settings gate:
  **108 passed** in 39.67 seconds.
- Settings/config/layout gate: **379 passed** in 313.04 seconds.
- Full capture-policy dialog file: **14 passed**; focused dialog/export
  re-review set: **20 passed**.
- Real SQLite/cache production sentinel focus: **1 passed**.
- Task 4 base-delta regression: **1 passed**.
- Ruff on every owned Python source and test: **passed**.
- `py_compile` on every owned production Python module: **passed**.
- `python -m tldw_chatbook.css.build_css`: **passed**; regenerated modular,
  widget-default, and screen CSS artifacts.
- `python -m tldw_chatbook.css.check_bundle_sync`: **passed** for all five
  generated bundles.
- Documentation boundary grep for Safe, Full, Anthropic, AGENTS, compression,
  WAL, backup, logical, 64 MiB, and 16 MiB: **passed**.
- `git diff --check` and staged-diff check: **passed**.
- Explicit Task 4 architecture delta: **20,093 lines / 633 methods**, below
  base `1b50778714` at **20,099 / 633**. The older repository-wide ceiling
  remains independently stale at 17,727/593; Task 4 neither raises nor worsens
  it, and its dedicated non-regression node passes.
- The repository-wide suite and the Impeccable detector were not run, as
  required. The controller owns the one permitted detector pass.

## Sentinel inspection

The corrected inspection drives an Anthropic provider resolution through
`ConsoleProviderGateway`, `ConsoleChatController`, `ConsoleChatStore`,
`ChatPersistenceService`, in-memory ChaChaNotes SQLite, the production
`get_message_exchanges` query, capture blob compression/decompression,
runtime/store/decoded-cache/decoded-storage owners, Redacted and Full export
projections, and a filesystem loguru sink. One Safe and one Full provider
exchange contain unique system, separately tagged AGENTS project/workspace, RAG,
tool-schema, tool-argument/result, ordinary semantic-secret, structured API
key, endpoint credential/query/fragment, structured path, and nested base64
sentinels.

Observed and asserted:

- Safe storage and Redacted export omit the tagged AGENTS/workspace body.
- Full storage and Full export retain the semantic system,
  AGENTS/workspace, RAG, tool, and ordinary-text sentinels.
- Structured API/tool credentials, endpoint userinfo/query/fragment, and raw
  nested base64 appear in none of the decoded owners, stored/exported
  projections, or logs. A direct whole-capture endpoint assertion would fail
  if a non-request field retained the configured credential-bearing URL.
- Binary content is represented by a deterministic `sha256:` stub.
- The configured filesystem log sink contains none of the sentinels.
- The selected filesystem export boundary separately proves its destination,
  body, and raw exception value never reach logs or traceback metadata.

## Files and authorized deviations

- Added the exchange exporter, shared capture-policy modal, governed exchange
  export modal, and their focused tests.
- Fix round 1 added `UI/Console_Modules/capture_policy_bindings.py`, the
  privacy-safe opt-in for `atomic_write_text`, and the explicit Task 4
  screen-baseline regression; no default behavior changed for other atomic
  writer callers.
- Updated the shared Trace export contract, Conversation Inspector, live and
  imported Trace screen, narrow `chat_screen.py` wiring, canonical F9 Settings,
  Console CSS source/generated bundles, both Console user-guide pages, config
  semantics regression, and the Task 4 Backlog note.
- Ownership was explicitly expanded by the controller to `tldw_chatbook/config.py`
  and `Tests/test_config_save_settings_semantics.py` for the import cycle.
- The controller also authorized staging the four mechanically regenerated
  screen/widget CSS artifacts required for honest bundle sync. They were not
  hand-edited.
- The fix-round CSS build changed only the generated modular timestamp; all
  five generated artifacts reproduce from their sources.
- Fix round 2 changed only the policy modal, its focused tests, and the real
  exchange sentinel; its CSS build again changed only the generated modular
  timestamp.
- Fix round 3 changed only the policy modal guard and its real Textual harness
  regression; the required CSS rebuild again changed only the generated
  modular timestamp.
- No dependency, second export enum, second policy owner, legacy Settings
  surface, or speculative abstraction was added. No generalizable new lesson
  was identified.

## Remaining review items

- Scoped re-review of fix round 3 remains outstanding. The controller-owned
  one-time Impeccable detector was already run and was not rerun.
- Task `TASK-22507.4` therefore remains **In Progress** with ACs unchecked.

## Final whole-branch correction

The final review reopened Task 4. Purge availability is now frozen from the
controller and freshly revalidated with immutable title/count/policy and
revision fences before confirmation and mutation; blocker reasons and
destructive-action limits are exact. Inspector status names the immutable
conversation title and armed next-send Full override, while Global detail
editing disables Inherit. Real 80x24/focus/changed-state regressions cover the
correction. See `final-review-fix-report.md` for exact RED/GREEN evidence. Task
4 remains **In Progress** with affected ACs #1 and #2 unchecked for independent
re-review.

# Task 4 implementation report

## Outcome

Implemented the ADR-089 shared Console capture-policy controls and governed
per-call exchange export in commit `d685a90009` (`feat(console): expose
governed full capture controls`). The Backlog task remains **In Progress** and
all acceptance criteria remain unchecked for independent review.

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

## Final gates

- Exact Task 4 privacy/UI matrix: **861 passed, 2 skipped, 0 failed** in
  374.40 seconds. Both skips were existing loopback-listener cases skipped
  because the sandbox denied listener creation.
- Production-shaped 80x24 policy/export/Inspector/live/imported/Settings gate:
  **101 passed**.
- Final exchange exporter/sentinel file: **6 passed**.
- Ruff on every owned Python source and test: **passed**.
- `py_compile` on every owned production Python module: **passed**.
- `python -m tldw_chatbook.css.build_css`: **passed**; regenerated modular,
  widget-default, and screen CSS artifacts.
- `python -m tldw_chatbook.css.check_bundle_sync`: **passed** for all five
  generated bundles.
- Documentation boundary grep for Safe, Full, Anthropic, AGENTS, compression,
  WAL, backup, logical, 64 MiB, and 16 MiB: **passed**.
- `git diff --check` and staged-diff check: **passed**.
- The repository-wide suite and the Impeccable detector were not run, as
  required. The controller owns the one permitted detector pass.

## Sentinel inspection

The durable inspection uses a real in-memory `CharactersRAGDB`, real capture
blob compression/decompression, in-memory and cache mirrors, Redacted and Full
export projections, and a filesystem loguru sink. One Safe and one Full
Anthropic-shaped exchange contain unique system, tagged AGENTS/workspace, RAG,
tool-schema, tool-argument/result, ordinary semantic-secret, structured API
key, endpoint credential/query/fragment, and nested base64 sentinels.

Observed and asserted:

- Safe storage and Redacted export omit the tagged AGENTS/workspace body.
- Full storage and Full export retain the semantic system,
  AGENTS/workspace, RAG, tool, and ordinary-text sentinels.
- Structured API/tool credentials, endpoint userinfo/query/fragment, and raw
  nested base64 appear in none of the stored/exported projections.
- Binary content is represented by a deterministic `sha256:` stub.
- The configured filesystem log sink contains none of the sentinels.

## Files and authorized deviations

- Added the exchange exporter, shared capture-policy modal, governed exchange
  export modal, and their focused tests.
- Updated the shared Trace export contract, Conversation Inspector, live and
  imported Trace screen, narrow `chat_screen.py` wiring, canonical F9 Settings,
  Console CSS source/generated bundles, both Console user-guide pages, config
  semantics regression, and the Task 4 Backlog note.
- Ownership was explicitly expanded by the controller to `tldw_chatbook/config.py`
  and `Tests/test_config_save_settings_semantics.py` for the import cycle.
- The controller also authorized staging the four mechanically regenerated
  screen/widget CSS artifacts required for honest bundle sync. They were not
  hand-edited.
- No dependency, second export enum, second policy owner, legacy Settings
  surface, or speculative abstraction was added. No generalizable new lesson
  was identified.

## Remaining review items

- Independent code/UX/privacy review and the controller-owned one-time
  Impeccable detector pass remain outstanding.
- Task `TASK-22507.4` therefore remains **In Progress** with ACs unchecked.

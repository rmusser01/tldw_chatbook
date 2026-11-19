# PR #3244 — CI follow-up and review

The published head `d909ce4bc3ddcaf3dbced880dfa49fdf93085d82` failed Perf Guard,
PR Fast Lane and Console P0. This follow-up fixes the reproduced branch regressions
without raising budgets or changing recovery admission policy. The PR remains draft;
a pushed fix still needs its own hosted checks.

ADR required: no
ADR path: N/A — existing ADR-097, ADR-150/161 and ADR-242/243 apply.
Reason: Restore the existing lazy-import, CSS and installed-migration contracts.
No new storage version, provider, dependency or speech boundary.

## Findings and changes

- Perf Guard measured UI-ready imports at 1034/1031, preimport at 564/558 and bare-type
  CSS rules at 276/270. Fast Lane failed the same CSS guard (2975 other tests passed).
  Defer portrait controllers, performance records and speech observers until their
  actual use. Template metadata no longer imports the two provider clients through
  the Media_Creation package facade; existing public exports still resolve to the
  original classes. A cold-process regression covers both import isolation and public
  export identity in the existing boot CI file. Fault-injection tests patch the PCM
  analyzer's defining class after its incidental sink-module alias is removed.
- Give six existing CSS subjects explicit widget classes and rebuild the consolidated
  defaults. Values, tokens and ordinary widget geometry remain unchanged. Buttons,
  dialog containers and inputs retain their existing owners and behavior.
- Console P0 failed every accepted v80 schema lineage during restricted restore.
  Its new portrait DDL lacked temporary authorizer authority; the predecessor gate
  and Canvas CHECK read scope also omitted the portrait-era historical catalogs.
  Admit only the installed v82→v83/v83→v84 objects while their own step executes,
  close that scope in `finally`, and retain exact full-catalog matching. Expand the
  existing pinned CI selector to versions 80–83 × four lineages × success/refusal/
  cancellation. Check exact successor catalogs, retained data, foreign keys and
  byte-identical rollback. Owner/step/source/temp-schema/unrelated-write negative
  controls run inside that same selector; no workflow change is needed.

## Verification

Counts overlap and must not be added into a distinct-test total. Source identities,
log hashes, results and the exact Console P0 command are retained in
[ci-follow-up/](../qa/portrait-compositing/2026-10-24/ci-follow-up/).

| Targeted run | Result |
|---|---|
| Full affected boot CI step, including new cold-import test | 29 passed |
| Exact failed sandboxed Console P0 step | 164 passed |
| Final response-rule/historical restore file, including scoped controls | 65 passed |
| Speech, sink, TTS and presentation gate | 308 passed |
| Post-cleanup speech/event/probe controls (overlap above) | 83 passed |
| Final radio geometry/CSS/token guards | 27 passed |
| Template catalog and lazy exports | 28 passed |
| Portrait migrations/region settings | 20 passed |
| Shared SQLite validation/dictionary catalogs | 86 passed; 2 baseline failures |
| Fresh installed authoring/Console journey | 1 passed, 41.16s |
| Fresh native F9 restore/open/component edit | 1 passed, 50.41s |

The final wheels both match **all 52 changed production files** relative to the
fetched dev base, including the Canvas/portrait/region SQL chain. Changed Python
compiles with Python 3.12; focused files pass formatting and exact legacy Ruff
comparisons add no diagnostics. Generated CSS and whitespace checks pass.
UI-ready imports are 1031/1031 and preimport is 558/558: both have zero module
headroom. No budget, snapshot or test expectation was relaxed.

The initial authoring/radio run had 93 passes and one two-row geometry assertion
failure. That original test passed in isolation; the complete radio/CSS run then
passed. The first combined installed run had one authoring timeout waiting for a
visible Console avatar and one successful native restore. The original authoring
journey passed isolated with unchanged timeouts, followed by a fresh final native
run. Their initial failures remain recorded; the cause is unconfirmed and no
product workaround or increased timeout was introduced.

Broader template/settings checks produced **27 failures and 32 setup errors**;
all 59 node IDs match unmodified dev exactly. The two shared validation failures
are RAG indexing schema-zero catalog mismatches and reproduce independently on
dev. These are not newly attributed portrait defects. No config/admission fence
was bypassed, runtime registry reset, fixture-wide state reset or golden-byte
regeneration was used. Earlier rebase evidence remains historical.

## Review and qualification limits

The inspected published head had no submitted GitHub reviews or inline review
threads. CodeRabbit explicitly skipped this draft; that is not approval. The
connected review tool requested GitHub connection, so authenticated `gh` supplied
CI logs and review metadata. Self-review covered deferred call sites, retained
runtime validation, public exports, selector coverage and migration scope/rollback;
no additional implementation finding remained after these corrections.

TASK-36555 stays **In Progress**, AC5 unchecked. Fresh actual configuration inventory
still finds **SwarmUI only**, unsupported by this app's reference-image path:
**zero live image requests, no fallback, `live_qualified=false`**. Historical external
ChatGPT patches and deterministic installed image/TTS/device fixtures do not qualify
an actual configured image adapter. Three real edits, selected retries and visual
semantic/alignment/lighting/seam review remain required with a supported adapter.
No full suite, live TTS or hardware/DAC qualification was run. No merge or automatic
review trigger is part of this follow-up.

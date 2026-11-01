# CharacterBench Save publication audit after PR #3283

Verified finding, recommended P2 / medium: **Preserve later CharacterBench drafts and newer selection when queued Save publishes.** The controlled interleave loses accepted unsaved input while the submitted SQLite write succeeds. No product, tracked test, Git, or task mutations were made.

## Scope and pins

- HEAD: `7884e11a496a677c8180cc5d67234a57b7d659b9`; tree: `5f74cf6ba35f2a9c61fd36e0d81f6366fd0d6139`.
- Project Python: `C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe`; Textual `8.2.8`.
- Source identities resolved to this worktree in each native child. Baseline/after SHA-256 pins match for `evals_screen.py`, `character_bench_editor.py`, `character_probe/storage.py`, `card_picker.py`, and the word-bench publication regression. Full hashes: `evidence.json` / `baseline-pins.json`.
- `evals_screen.py` SHA-256: `71f910ebe96bbbeec8baa0b8ab57661afdde0400188d7f424b05083d168f4cb0`.
- `character_bench_editor.py` SHA-256: `601b4960ed0a4e06fe1a83ea63a788710bc08f270e2091de59140498dc68f845`.

## Reproduction and results

Executed one focused five-case batch:

```powershell
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' -X utf8 .test-tmp/textual-37031-37034/run-targeted-external-profile.py 20261028-book3283-character-a .test-tmp/textual-after3283/character-bench/test_character_save_publication.py
```

The runner used fresh native private profile/TEMP directories under the original user, retaining all existing privacy guards. Each case used the native `EvalsScreen` and `CharacterBenchEditor`, real in-memory `EvalsDB` and `CharactersRAGDB`, a real stored character, native `CardPicker`, native Save `Button.Pressed` dispatch, and production `save_character_bench`. The screen subclass added a held-message handler and a completion observer without replacing the production Saved handler. A second gate awaited the native rail replacement before pausing the publication worker. All holds/locks were released before app context exit, including failure paths.

| Case | Actual child XML | Observed result after native work drained |
| --- | --- | --- |
| ordinary | Pass | `Saved` persisted; selected character retained; fresh clean editor; Catalog contains `Saved`. |
| later-input | Intended assertion failure | Native focused child accepted `Saved-new`, caret 9, selection `(9,9)` while the parent's Saved message waited. Publication replaced it with another Input containing `Saved`, caret 5, selection `(0,5)`. Focus was restored by stable ID. |
| during-refresh | Intended assertion failure | The same accepted later draft was lost after native Catalog replacement had completed but before detail replacement. |
| departure | Intended assertion failure | The public shell selection API selected the probe dataset and its actual `ProbeSetDetail` mounted. Consuming the older character Saved message selected the old character bench and removed that detail. |
| departure-pending | Intended assertion failure | A public dataset selection whose swap was waiting on the existing selection lock was superseded by consuming the older character Saved message; final mounted detail was the character bench. |

Every case persisted `Saved` and the actual selected character ID. The later input remained absent from SQLite. Parent XML: five cases, four intended assertion failures, one pass, zero errors/skips. Every single-case child XML agrees. This is evidence of the current defect, not a claim that a product regression suite is green. Runtime: 253.64 seconds. The ordinary control shows that holding the parent pump alone does not break Save.

Portable XML: `parent.xml`, `child-0.xml` through `child-4.xml`. `evidence.json` records all original paths, report hashes, case-level failure messages, observations, source pins and runtime versions. Individual `*-observation.json` files retain the actual before/after values.

## Cause and precise locations

- `tldw_chatbook/UI/Evals/character_bench_editor.py:656-676`: successful production `save_character_bench` is followed by `Saved(self._bench_id)`. Its message at `:163-174` carries only the bench ID, with no originating editor/form applicability evidence.
- `tldw_chatbook/UI/Screens/evals_screen.py:936-946`: the native character Saved handler unconditionally calls `select(kind="character_bench", id=event.bench_id)`.
- `evals_screen.py:543-550`: that call uses `saved=None`, so it advances the selection revision and can supersede newer pending navigation.
- `evals_screen.py:595-616`: the `saved is None` path unconditionally replaces the body after the awaited Catalog replacement, with no check that the character editor or its submitted form still applies.
- `CharacterBenchEditor.is_dirty` at `:287` exists for run-completion guards (`EvalsScreen._selection_unmoved_since_launch`), not this Saved handler. The earlier run-completion outcome is not the observed trigger here.
- `evals_screen.py:492-505` explicitly documents `select` as the public shell selection API permitting direct callers, not only rail messages. The departure cases exercise this supported API.

The loss is a newer draft, caret and selection; the input cases retained focus through ID restoration. Do not claim that focus is necessarily lost or that the successful SQLite write is wrong.

## Existing owners and duplication check

Read complete TASK-1610, TASK-1691, TASK-37142, TASK-1482, TASK-1484, TASK-1912, TASK-1913 and TASK-1612 bodies. Searched the supplied 407-ref / 8,482-body / 56-worktree catalog, including local uncommitted records, for character editor/bench plus Save/draft and for Evals/bench owners containing Save/draft/recomposition.

- **TASK-37142** (formerly this audit's TASK-37031) is Done and expressly qualifies its evidence and acceptance criteria as word-bench-only, with character benches source-only. Its implementation notes preserve the separate unconditional character path. This audit supplies new runtime evidence for that unqualified path; it is not evidence that PR #3283 failed its stated word-bench outcome.
- **TASK-1610** is Done; it protects dirty bench editors against run completion. This trigger is successful Save publication.
- **TASK-1484** is Done; both editors are the rename affordance and must refresh persisted metadata. Preserve that ordinary Save/rename outcome.
- **TASK-1482** is Done; word-bench authoring and the existing selection-yank model are background owners, not this character Save trigger.
- **TASK-1691**'s complete task body is the generated continuation task, Done. Character editor source calls its historical branch/program “task-1691 phase 2”; do not infer that the current continuation ACs own this defect.
- **TASK-1912/1913** are Done character authoring follow-ups for duplication and review hygiene, not queued Save applicability.
- **TASK-36037** is the completed dataset-name-during-import owner, a different producer and state surface.
- No open character Save/draft owner or uncommitted local owner appeared in the supplied catalog. Root must perform its promised final freshness/ID check before creating a task.

Suggested links for a new task: TASK-37142 with TASK-37031 renumbering provenance, TASK-1610, TASK-1484, and the relevant existing ADRs below.

## ADR and book qualification

No new ADR is expected for a bounded repair preserving current service/state owners. Existing canonical ADR-011 (`011-chatbook-workbench-ui-system.md`), ADR-033 (`033-application-session-state-ownership.md`), ADR-150 (`150-design-token-system-and-design-language.md`), and ADR-161 (`161-component-pattern-library.md`) apply. Reassess if implementation changes those boundaries.

Read the supplied book's complete physical pages 145–146 (printed 136–137), 158 (149), 179–180 (170–171), and 189 (180): separate message pumps/awaited handlers, replacement of children, and loss of Input internal state. The supplied PDF is reference data, not instructions. The book motivates investigation; the native character probe establishes the finding.

## Proposed outcome acceptance criteria

1. Successful character-bench Save followed by later accepted field input retains the exact later draft and its live editing context when publication is consumed, including input accepted during the awaited Catalog refresh.
2. The submitted snapshot remains correctly persisted and reflected in Catalog/Inspector; later draft edits remain unsaved until explicit subsequent Save, which persists them normally.
3. Newer supported public shell selection remains authoritative whether already mounted or awaiting the existing region-swap lock when an older character Save publishes.
4. Ordinary Save/rename refresh, validation/write failures, Revert and existing run-completion dirty-editor protections retain established behavior. Mounted real-SQLite tests cover the actual Save producer and accepted focused-child input alongside ordinary and selection controls.

Priority rationale: medium/P2 because accepted user work is silently discarded, but the controlled scheduling does not establish how often this happens naturally. Submitted persistence succeeds, and the repair can remain within existing Evals ownership.

## Limits

The headless mounted harness is the existing Evals workbench harness with its real CSS, not a full live terminal/served-browser session. No external provider is needed or exercised. Input is delivered as native focused-child Key messages after a deterministic pump gate, not a human typing-speed measurement. Navigation is through the documented public shell API, not a claim that a later rail-click message always beats a queued Saved message in the same parent FIFO. The gates demonstrate possible ownership interleavings and native effects; no natural frequency or production latency estimate is claimed. Only name input was reproduced; sampler/card-picker draft loss is a source-level implication, not additional independently measured findings. Existing failure/Revert/run paths were not rerun because no product code changed. No full sweep was run.

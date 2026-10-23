# Pending draft follow-ups: TASK-36454 through TASK-36458

Status: implementation and targeted local verification recorded; independent Superpowers review and PR CI pending.

## Scope and ownership

| Task | Repair | Existing decisions |
| --- | --- | --- |
| TASK-36454 | Prompt Save adopts committed identity/version, keeps newer raw metadata and Basic/Advanced blocks dirty, and keeps the next Save on that record | ADR-033 application state, ADR-040/049/076/086 |
| TASK-36455 | Dictionary Settings acknowledges submitted raw controls without reloading over a newer draft | ADR-033 application state, ADR-046/079/115 |
| TASK-36456 | First Basic block mounts re-read both live lanes after structural awaits, including clear and cross-lane input | ADR-033 application state, ADR-040/076 |
| TASK-36457 | Theme Save retains later raw palette/name/dark edits and protects the Saved-message handoff; actual saved-file identity remains separate | ADR-011, both applicable ADR-033 records, ADR-150/161 |
| TASK-36458 | Parakeet completion may finish consented activation/preference but selects the batch only when its original form and live controls still match | ADR-014/025/050/033 application state/011 |

The [audit](2026-10-23-textual-book-after-pr3219.md) and each task retain the existing-task/ADR links and source-bound findings. No new ADR: these repairs preserve the existing draft, persistence, artifact, consent and navigation boundaries. The [implementation plan](../plans/2026-10-23-textual-pending-draft-followups.md) records scope and review focus.

`prompt_save_ack.py` and `parakeet_install_selection.py` are narrow presentation helpers for existing owners. The former extracts the old post-save completion; the latter extracts the old install-result callback while preserving the original UI-thread delegate. Dictionary completion joins the existing Persona completion helper. The three affected size ceilings decrease to their actual measurements: Prompt controller 5,100, Library screen 37,525, Personas screen 16,554. No unrelated ceiling was raised.

## Regression evidence

Tests use enabled mounted child controls and explicitly witness accepted input before releasing held work. Prompt and Dictionary writes reach real SQLite services; Theme writes real TOML. Parakeet retains the actual install button, consent modal, thread worker and original UI publication, with dependency availability and external acquisition/preference seams qualified. It proves UI selection ownership, not downloads, package installation or inference.

| New regression module | Exact cases in required PR census |
| --- | ---: |
| `Tests/UI/test_library_prompt_pending_drafts.py` | 22 |
| `Tests/UI/test_dictionary_settings_pending_save_drafts.py` | 12 |
| `Tests/UI/test_theme_pending_save_drafts.py` | 13 |
| `Tests/UI/test_parakeet_install_selection_retention.py` | 9 |
| Total | 56 |

All 56 have passing local coverage across the final relevant selections, with iterations qualified below. They are appended as exact parameterized node IDs: the 664 inherited entries remain in their original order, floor 720, 13 shards, and unchanged time limits. The required CI will execute the full new set on the PR head; collection alone is not counted as execution.

Qualified RED controls reproduce later-key/clear/cross-lane loss in both Basic lanes; create/update false-clean acknowledgement and Advanced clobber; held older semantic publication losing actual `xy`; Dictionary name/description/raw budget/strategy/enabled clobber; Theme palette/name/dark loss and its delayed Saved handoff; Parakeet provider, precision and folder replacement. Matched ordinary and definite-failure controls distinguish normal successful completion and retry.

Prompt local evidence: first-mount final selection 8/8; corrected queued-body selection 2/2; create final selection passes ordinary/body/metadata, with its older queued-body fixture result superseded by the corrected 2/2 run; other update, failure and held-semantic cases pass. The first acknowledgement extraction omitted the existing action-capability refresh after creation: the second-Save regression caught it, the existing refresh was restored, and the affected create controls passed. An earlier queued injector incorrectly added another edit during the second Save; limiting it to the intended first attempt fixed that test without a production change. Initial missing-request/wrong-import/wrong-private-seam failures are fixture setup failures, not product evidence.

Dictionary: 10/10 initial GREEN plus 6/6 supplemental final-source controls cover the 12 distinct cases, including a reverted edit, raw invalid budget, queued notification and actual guard Stay actions. Theme: all 13 final cases pass, including Save-as, next Save, invalid raw palette/name and delayed Saved publication. Parakeet: all 9 new cases and 25 existing contracts pass, including queued folder input, changed batch, ABA choice and ordinary override.

## Compatibility and static checks

- Prompt block-editor controls: 29 pass. Five existing Prompt service/reader journeys and six recompose guards pass with the repository-supported bootstrap-profile qualification.
- Dictionary: eight existing widget controls and 17 existing screen contracts pass; the latter use the same supported bootstrap-profile qualification.
- Theme: 17 of 18 existing controls and five Reset controls pass directly. The remaining Save-as test assumes catalog completion after a fixed pause; it also fails on pristine HEAD before the pending scan lands. A bounded final-source scan witness preserves every original assertion and passes, with the new file selected after the real scan. The existing test is unchanged.
- Gate contracts: 178 pass. Affected owner ratchets and Prompt wiring: 19 pass. Task ID/readability checks pass across 5,697 files. New helpers/tests pass Ruff and formatting; nine existing production owners have no new Ruff diagnostics compared with dev (existing findings are retained).
- The diagnostic inventory change is one existing sanitized warning moved from LibraryScreen to its helper, with the same text, error-type metadata and `module=LibraryScreen` binding. No draft content, path, secret or persistent sink is added.

Unqualified standalone legacy UI selections hit `RecoveryRequired: raw_source_selection_changed` before their behavior. The supported `bootstrap_profile` marker (private compatibility plugin) or normal collection import context qualifies them without changing production admission guards. Dictionary's representative failure is reproduced with pristine Git HEAD module bytes. Parakeet's final 25-contract confirmation passes with normal collection context. These setup failures remain in the raw receipts.

A wider size-only check found nine unrelated baseline overages in the ingest controller, MCP workbench, change-review/LLM screens, Console transcript, app command providers/ingest queue/lifecycle, and API client. Each source file is byte-identical to dev; they were not repaired or repinned here. The affected Prompt-controller slack is repaired by lowering its own pin. The initial broad UI recompose setup errors are superseded by the six qualified passing guards above.

## Reproduction and delivery

The four new modules are independently runnable via targeted pytest; disposable private-profile wrappers are part of the repository test harness. Required PR selection is generated by `scripts/check_ui_pr_gate_census.py --shard <n> --shards 13`. No full local test sweep was run.

Original native commands, source snapshots, full logs and JUnit receipts are preserved under `.test-tmp/task-36360-36363/` and `.test-tmp/task-36454-36458/`; they are private working evidence and are not committed. The public tests, task notes, required CI logs and this report are the durable review surface.

Superpowers review of `7103ed34` found one Important/P1 issue: a two-lane Basic capture refreshed only the triggering Advanced card, leaving an already populated other lane stale. Switching to Advanced could then save its older text. The analogous Save-acknowledgement convergence used a blank block ID and refreshed no card. Both convergence loops now refresh every captured block, then recapture after structural awaits.

Three additional mounted controls reproduce the actual stale Advanced TextArea before the repair and pass afterward (3/3, 104.872 seconds), including both lane directions, stable child/block identity, mode-switch capture and real SQLite Save/version checks. The acknowledgement control holds actual Changed notifications at child publication while allowing DOM lifecycle messages; it does not claim a naturally occurring race frequency. Initial fixture attempts paused empty lanes, waited on a two-key Pilot sequence behind a held parent, or blocked lifecycle work with the canvas pump; those setup/timeout failures are excluded. Their receipts remain private. The original 717 census slots remain unchanged, with these three appended.

Focused follow-up review, neighboring Prompt verification and hosted CI remain pending before final task closeout.

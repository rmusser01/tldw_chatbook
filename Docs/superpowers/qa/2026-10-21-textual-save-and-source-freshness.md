# Textual save and source freshness verification

Implementation scope: TASK-36097 through TASK-36100. Base dev: `2a5afa03b31f60e00bdcd10a161e3e351fad023b`; task-filing baseline: `f793f3cea353dad71a5039b930f48b2a51cf55b9`.

[Plan](../plans/2026-10-21-textual-save-and-source-freshness.md). [Source-qualified book audit and existing ownership](2026-10-21-textual-book-after-pr3199.md). The book is reference material; its prose does not authorize repository actions.

## Behavior and decisions

- Character/Persona acknowledge the raw form captured by the widget's Save message. Actual persisted identity/version advance while newer fields and greetings remain dirty and participate in existing discard guards.
- Image/Video Settings acknowledge only the submitted draft, retain later raw values through configuration reload, and preserve the current editing context. A deliberate focus move wins over restoration.
- Study quiz/question creation clears each field only if its raw value still equals the submission. Existing normalized payloads and service policy boundaries remain unchanged.
- Audiobook source changes immediately invalidate pending chapter publication, including an empty source and the debounce gap. Existing dispatch ordering still protects a newer result. Programmatic same-source TextArea echoes retain their valid import dispatch.

No new ADR is required. Existing ADR-046, both ADR-033 records, ADR-011, ADR-150 and ADR-161 govern these repairs. ADR-044 remains secondary context, with its decision 2 superseded by ADR-176. No new storage schema, provider/runtime boundary, dependency, global draft store, synthesis operation or network effect is introduced.

## Native targeted evidence

Python 3.12.10 / Textual 8.2.8 on Windows. Every test invocation creates a fresh native private TEMP/TMP, bootstrap configuration root and retained basetemp using `platform_files.os.mkdir(..., 0o700)`. Private-profile marked tests accept `request` and run in their admitted child process. Controlled waits delegate to actual persistence/detection; no storage success is fabricated. Scratch receipts are retained under `.test-tmp/task-36097-36100/` in the attached worktree, and are not committed.

| Task | Meaningful red | Green evidence |
| --- | --- | --- |
| 36097 | `36097/red`: four create/update × Character/Persona false-clean assertions; `red-queued-dispatch`: two queued-submission assertions; `red-refresh-bounded`: one false-clean refresh assertion | `36097/green-2`: 10 passed, 495.65 s (source movement qualified); `final-persona-exact`: four Persona create/update/queued/refresh cases passed on final source, 169.35 s |
| 36098 | `36098/red-settled`: both media panels lose later edits | `36098/green-bounded`: original ten cases passed, 627.96 s; reviewed later-write/reload and expanded nondefault sibling cases pass in `green-reviewed-utf8` (four passes before intended keyboard-clear red); final `green-history-final` five unchanged/keyboard-clear/history cases pass, 144.37 s |
| 36099 | `36099/baseline-and-final-red-af691331`: exact final tests on original handler, two intended draft-loss failures and two passing ordinary/retry controls | `36099/final-green-6f0ebe75`: 10 passed, 77.24 s; four mounted regressions plus six existing local/scope service controls |
| 36100 | `36100/red-1`: two intended stale-source failures (clear/replace), two passing controls, 63.92 s | `36100/green-1`: four passed, 89.98 s; `import-and-parser`: same-source file-import echo plus nine real parser controls, ten passed, 38.22 s |

Study coverage includes independent/both-field/whitespace edits, persisted normalized data separately from live raw input, current selection, unchanged success and failed-create retry. Audiobook coverage observes actual old physical worker publication attempts, actual TextArea changes, editor/table contents and success notifications; it releases controlled threads before mounted teardown.

## Qualifications

Initial fixture/settlement attempts are retained separately and are not behavioral red evidence. Media tests received bounded message-dispatch settlement and completion waits after an observed real configuration reload exceeded a five-second helper deadline. No product delay or test admission guard was weakened.

Study legacy adjacent tests reproduced the same three `RecoveryRequired(raw_source_selection_changed)` failures against the original handler; a fourth test passed its body and failed worker retirement during teardown. The changed module and six focused service controls passed on final source. Media adjacent selection passed 124 controls with nine existing failures and 17 deselections. Two Video fixture failures stop at `RecoveryRequired(raw_source_selection_changed)`; five governance failures arise from native path separator assumptions, one from cp1252 decoding, and one from the unchanged Library panel size ceiling. The exact test/input participant sources are unchanged against dev (`36098/adjacent-qualification.json`, `adjacent-baseline-source.json`), and no baseline is weakened. Persona adjacent selection passed 24 controls and had four legacy mounted tests fail admission before their bodies. These failures are not reported as passes.

An intermediate broad module-size ratchet run had eight failures: seven unrelated files are byte-identical to the task baseline (`36100/unchanged-size-debt.json`), and Settings was over its ceiling before the scoped completion-helper extraction. Final changed-owner checks are recorded below. No unrelated size budget is increased.

STTS full-file Ruff has five inherited E721 diagnostics, and inherited formatter debt. Exact baseline/head diagnostic multiplicities and formatting replacement hunks are identical (`36100/static.json`). Its new regression module is lint/format clean. Other changed-owner static evidence is recorded with final integration.

Review also noted pre-existing staged-avatar attachment handling and long-import preview behavior outside the accepted raw-form/short-import scope. These are source-inferred limitations, not exercised upload/data-loss findings or claims fixed here.

No full local suite, live server, generated media, natural local service latency, production terminal/pointer geometry, or screen-switch lifecycle claim is made.

## PR integration and review

The independent Superpowers review identified expanded nondefault backend collapse, explicit Clear/retype history loss (including incremental typing and an unrelated equal global event), and accidental Unicode conversion during helper extraction. The source repairs preserve expansion/scroll/current focus, retain only explicit field history, and restore every unrelated Unicode byte. Failed and interrupted intermediate receipts are excluded from final-source claims. The reviewer verified the final bounded changes and source identities. Qodo is not used.

All 549 existing census entries retain their exact order. Thirty-five individual new regression nodes are appended (584 total), preserving thirteen shards and workflow time limits. Prior hosted UI4 body time was 994 seconds; native case distribution adds approximately 50 seconds there, with at most 198 seconds added on a lighter shard. These cross-platform figures are capacity estimates, not a hosted runtime certificate; the PR checks supply the actual final-source result.

Final integrated run: **92 passed, 24.21 s**, covering CI/census/shard/service contracts, the explicit-history regression and both changed-owner size ceilings. New test files and media helper pass scoped Ruff lint/format; the census checker is lint clean with identical pre-existing formatter hunks. Final diagnostics inventory: 689 owners, unchanged 1049/56/7717 diagnostic call totals and sixteen sinks. One literal warning moved from PersonasScreen to its completion helper; its text and logger binding are unchanged, with no new interpolated data or sink. The worker/DOM census transfers only the same two existing post-await lookups to that helper; unrelated pre-existing resolved census rows are retained.

[Source identities](2026-10-21-textual-save-and-source-freshness-evidence.json) pin the final product and regression files. Hosted CI links and the current-head verdict belong to the PR checks and final PR description; this report does not claim a future hosted result. The user requested a PR against dev, with CI and review feedback handled.

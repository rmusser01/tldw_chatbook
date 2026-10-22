# Textual source and pending-draft repairs — 2026-10-22

Implements TASK-36130, TASK-36131 and TASK-36132 from the post-PR3205 book audit. Original native baseline: dev `7700c63a688355a2df595941b8b2b4dc8a819074`. Rebased onto dev `e2cffbed60ae1bceb3eb26f295de15968aae9811` after PR #3184 merged; all ten implementation/test/architecture source fingerprints remain unchanged. These are corrections within existing owners; ADRs 011, 033 and 046 apply. No new ADR, service, schema, dependency, visual token or keybinding was required.

## Result

- Audiobook file, Notes and Conversation imports place complete source in the existing editable TextArea. Generate and chapter detection retain the tail; deliberate edits/clears retain their source authority.
- Character Save acknowledges its submitted attachment while adopting the persisted id/version. A later avatar remains visible and unsaved through Stay decisions, then persists to the same record on the next Save.
- Flashcards deck/card creates clear each field only when its live raw value still matches its submission. Later text, whitespace, caret/selection and moved focus remain; payload normalization and definite pre-write retry contracts stay intact.

## Native targeted evidence

| Area | Meaningful RED | Final new regressions | Adjacent checks |
| --- | --- | --- | --- |
| TASK-36130 | 3 source-loss failures, 4 controls pass | 7 pass, 69.176s | 5 freshness cases pass; 13 legacy admission setup errors qualified below |
| TASK-36131 | 2 attachment-loss failures, 2 controls pass | 4 pass, 114.888s | 22 pass, 278.110s |
| TASK-36132 | 2 draft-loss failures, 2 controls pass | 4 pass, 42.161s | 19 pass, 78.863s |
| Census/shard/service integration | Baseline 76 pass | 87 pass, 11.012s | Original 599 and rebased 603-selector censuses validate |

The three task-scoped spec/quality reviews approved the changes with no blocking findings. Whole-branch requesting-code-review and current-head hosted CI are PR-stage gates; their verdicts are recorded in the PR. Qodo is not requested.

All runs use native private profile/TEMP/TMP/bootstrap admission, real local persistence/services, and explicit UTF-8. Successful persistence is never fabricated. Generation requests are consumed before synthesis; no provider/network operation is installed in the new harnesses. No full local test sweep ran.

## Shared guards and CI coverage

Original native diagnostic, Textual worker, markup interpolation and timestamp-writer guards pass on matching before/after source hashes. These counters describe the original native baseline; current-head hosted CI verifies the rebased dev graph and pins. No inventory pins, module/boot ceilings, shard count or timeouts were raised. Original diagnostic inventory was 690 owners, 1051 TASK-492 calls, 56 TASK-31551 calls, 7719 TASK-494 calls and 16 sink files. Worker guard reports no new synchronous targets, post-await lookups or wait-for-dismiss entry points.

Initially all 15 new concrete regression nodes were appended to the 584-entry prefix: 599 selectors. The rebase preserves dev's expanded 588-entry prefix and appends the same 15 nodes: **603 selectors** over the same thirteen shards. Every dev selector and shard slot is unchanged; the targeted 87 census/shard/service checks pass again after conflict resolution. Measured native added work is 9.592–28.129 seconds per shard; this is measured test duration, not a hosted-time prediction. The source checker floor advances with coverage.

New test files and the changed Character widget/Flashcards handler pass scoped Ruff lint/format. STTS retains the exact five baseline E721 diagnostics and format debt; PersonasScreen retains its 24 baseline E721 diagnostics and one format hunk. The screen remains at the unchanged 16,561-line ceiling. Baseline/current source/AST comparisons qualify this inherited debt without suppression or broad reformatting.

## Qualifications and retained evidence

Thirteen selected legacy audiobook cases stopped in autouse setup with `RecoveryRequired(raw_source_selection_changed)` before their bodies. No matched-baseline run was performed; those bodies are not certified passing. The maintained real-service regressions and five source-freshness cases did execute. Existing Pydantic/pydub/pytest-asyncio warnings remain; captured RED thumbnail noise is distinct from the meaningful avatar-byte failures.

The avatar screen received AST-equivalent line/comma formatting during GREEN; test source was stable and no semantic change followed. An initial Flashcards candidate hit SQLite's unique deck-name rule because test inputs repeated a name; numbered inputs fixed that fixture and passed the final run. An interrupted redundant RED is non-evidence. A shared scan overlapped its temporary baseline restoration, so that scan is excluded as final-head evidence; the final shared guard run occurs after implementation handoff with matching source hashes.

Mounted tests do not claim terminal pointer UAT, live remote services/providers, audio synthesis, large-book performance benchmarks or uncertain-write/exactly-once behavior. Existing list/status/replacement-panel tasks retain their scope.

Exact commands, JUnit/log hashes, source identities, review identities and census positions are in [the evidence manifest](2026-10-22-textual-source-and-attachment-drafts-evidence.json). Raw native receipts remain locally under `.test-tmp/task-36130-36132/`; complete task reports/reviews remain under the owned `.superpowers/sdd/2026-10-22-textual-source-and-attachment-drafts/` workspace. The earlier post-PR3205 audit is historical evidence of the repaired failures.

Rebase integration on [PR #3207](https://github.com/rmusser01/tldw_chatbook/pull/3207) retained both lesson additions and resolved only the shared census floor/tail. The original source review remains applicable by exact fingerprints and range-diff; the scoped integration review and hosted checks are recorded in the PR.

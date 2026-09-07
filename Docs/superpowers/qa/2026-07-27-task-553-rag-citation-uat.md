# TASK-553 local RAG citation UAT — 2026-07-27

## Outcome

Passed after the TASK-553.17 repair.

The rendered application searched a real local note, staged the exact evidence
in Console, generated the correct answer with `[S1]`, persisted one canonical
trace, exposed `Sources (1)`, displayed the literal cited chunk, and opened the
exact note in Library. After stopping and restarting the application, the
persisted answer still exposed `Sources (1)` and the Sources modal still loaded
the same chunk.

The first run exposed a real production-composition defect: a fresh profile had
a `fingerprint_key_id`, but no keyring secret, so Console saved the answer
without its trace. TASK-553.17 fixed only that first-use key provisioning
boundary. No server, sync, export, import, or additional provenance system was
added.

## Environment

- Revision: `a73b9b46ff86c1ff189f291a95ccd49bb01d7b2c`
  (`origin/dev` at UAT start)
- App: rendered Textual application served locally and driven in a headed
  Chromium session
- Profile: fresh isolated config and data directory; the real HOME was retained
  so the production macOS keyring backend was available
- Retrieval corpus: one real note persisted through `CharactersRAGDB`
- Generation provider: local llama.cpp at `127.0.0.1:9099`
- Citation setting: `[rag_citations].canonical_writes_enabled = true`
- Server dependency: none; `tldw_server` was not used

Seeded note facts:

- Title: `Zephyr Orchard launch checkpoint`
- Verification code: `marigold-73`
- Launch review: Thursday at 14:30 UTC

## Rendered user journey

| Step | Result |
| --- | --- |
| Open Library and search `marigold-73` | Pass; the exact note and matching text were returned |
| Select the result as evidence | Pass |
| Hand the selected evidence to Console | Pass; RAG changed to on and one source was staged |
| Ask for the code and review time | Pass |
| Generate against the staged source | Pass; answer returned `marigold-73`, Thursday at 14:30 UTC, and `[S1]` |
| Persist canonical trace and owner | Pass; one trace, owner, answer payload, and evidence reference were stored |
| Show `Sources (1)` on the answer | Pass |
| Inspect the exact cited chunk | Pass; the modal showed the literal note text |
| Open the original note in Library | Pass; the existing exact-ID route opened `Zephyr Orchard launch checkpoint` |
| Restart the application and reopen the answer | Pass; `Sources (1)` and the modal remained available |

Initial failure screenshots:

- [Library search result](task-553-citation-uat-search-result-2026-07-27.png)
- [Staged Console source](task-553-citation-uat-console-staged-source-2026-07-27.png)
- [Correct generated answer with `[S1]`](task-553-citation-uat-generated-answer-2026-07-27.png)
- [Persisted answer after reconnect, without a Sources footer](task-553-citation-uat-persisted-answer-no-footer-2026-07-27.png)

Passing rerun screenshots:

- [Generated answer with `Sources (1)`](task-553-citation-uat-fixed-answer-sources-2026-07-27.png)
- [Sources modal with the exact literal chunk](task-553-citation-uat-fixed-sources-modal-2026-07-27.png)
- [`Open in Library` resolved to the exact note](task-553-citation-uat-fixed-library-open-2026-07-27.png)
- [Persisted answer with `Sources (1)` after app restart](task-553-citation-uat-fixed-restart-sources-2026-07-27.png)

## Persistence evidence

Before the repair, ordinary messages persisted but all four canonical citation
tables remained empty. On the passing rerun, the same fresh-profile journey
produced:

```text
rag_citation_traces          1
rag_message_trace_owners     1
rag_answer_attempt_payloads  1
rag_trace_evidence_refs      1
```

The app was then stopped, the local server restarted, and the persisted
conversation reopened through the rendered UI. The footer and exact chunk both
remained readable. The temporary keychain entry and isolated profile were
removed after the UAT.

## Scoped automated verification

Verification remained scoped to the touched citation path:

- Citation identity, service composition, builder, repository, terminal
  persistence, Console boundary, footer/modal, and transcript selection:
  **437 passed**
- Final touched identity/factory modules after self-review: **29 passed**
- Filtered Library citation source and exact-open checks:
  **8 passed, 259 deselected**
- Ruff on the four touched Python/test files: passed
- `git diff --check`: passed

The only warning was the repository's existing `requests` dependency-version
warning.

---
id: TASK-21700
title: >-
  RAG diagnostics render the user search query and whole model replies
status: Done
assignee: []
created_date: '2026-08-24'
labels:
  - privacy
  - diagnostics
  - rag
priority: medium
---

## Description

> **Title corrected 2026-08-24.** This was filed as "…into a persistent sink". That framing was
> **wrong**, and the implementation disproved it by measurement: the shipped
> `PrivateRotatingFileHandler` carries a **default-deny** `PersistentDiagnosticFilter`, so none of
> these statements reach the log file. The provable blast radius is the **live terminal and the
> in-app Logs view**, not a file a user attaches to a bug report. The finding still stands — the
> sweep found three sites *worse* than the one originally filed — but the severity is narrower than
> the title claimed, and the title should not keep asserting something that was measured false.


`RAG_Search/simplified/rag_service.py` logs a cache hit at INFO level with the first 50
characters of the user's search query interpolated into the message. Search queries are user
content. The diagnostic-privacy work in this repo (TASK-15103/15600, and the persistent
diagnostic inventory that guards it) exists to keep exactly this class of value out of
persistent sinks.

Pre-existing — it was noticed while re-pinning the inventory after an unrelated RAG refactor
re-wrapped the line, and the statement text is byte-identical to what was there before. Filing
rather than fixing in that commit, because the fix is a behaviour change and deserves its own
review.

## Acceptance Criteria

- [x] It is determined and recorded whether this statement actually reaches a persistent sink at the shipped default log level, or is filtered before it lands
- [x] If it does reach one, the query text is removed, hashed, or length-only — the diagnostic keeps its debugging value without carrying user content
- [x] The whole `RAG_Search` tree is swept for the same shape: any diagnostic interpolating a query, a document body, a chunk, or a filename
- [x] Anything found is fixed in the same pass, or filed with a reason it is safe
- [x] A test or census guard pins the outcome so a future refactor cannot reintroduce it
- [x] If the conclusion is that this is acceptable, that reasoning is written down — an unreviewed "it's fine" is what let it sit this long

## Evidence (verified on dev 1daa47f0a, 2026-08-24)

`tldw_chatbook/RAG_Search/simplified/rag_service.py:1340`:

```python
logger.info(
    f"[{correlation_id}] Cache hit for query: '{query[:50]}...'"
)
```

The inventory checker flags added statements with exactly this prompt — *"does it interpolate
user content, a secret, a path, or a URL?"* — and this one does. It was previously at line 1315
with identical text, so the refactor moved it rather than introducing it.

Note the truncation to 50 characters limits the volume but not the kind: a 50-character prefix of
a search query is still the user's words.

## Implementation Plan

1. Re-confirm both cited statements on the branch base (line numbers drift).
2. Establish the SINK reality empirically before deciding severity: install the
   real `PrivateRotatingFileHandler`, emit this exact statement from a real
   `RAG_Search.simplified` module through the real loguru bridge, read the file
   back. Record the answer either way.
3. Sweep the whole `RAG_Search` tree by AST, not grep, for diagnostics that
   interpolate a query, a body, a chunk, a filename, or a path.
4. Fix the user-content sites; keep debuggability (a handle, not silence).
   Record the reason for every site left alone.
5. Add a census guard in the spirit of the inventory checker, plus real
   behavioural tests through the production seams.
6. Review each drifted inventory row with `--statements` before `--write`.

## Implementation Notes

### AC#1 — does it reach a persistent sink? No. Measured, not assumed.

The shipped `PrivateRotatingFileHandler` (`Logging_Config._configure_private_
file_logging`) carries `PersistentDiagnosticFilter`, which is default-DENY: a
Chatbook record is admitted only when it carries the
`_tldw_metadata_only_record` marker, and only `log_persistent_metadata` sets
that. A loguru record cannot carry it, because `_forward_loguru_to_standard`
rebuilds `extra` from scratch — deliberately, so no caller can
`logger.bind()` its way past the schema.

Probed on this base with an isolated `HOME`/`TLDW_CONFIG_PATH`: the real sink
installed at its default level (`file_log_level=INFO`), the cache-hit statement
emitted from a real module under `tldw_chatbook/RAG_Search/simplified/`, plus a
control `persist_event`. The log file contained the control record and
**neither the query nor the phrase "Cache hit for query"**.

The Logs screen's "Copy all"/"Copy visible" export reuses the same
`PersistentDiagnosticFilter` object and replaces every non-metadata message
body with `***REDACTED***`, so the clipboard route is closed too. The live
in-app Logs view and the terminal do show the line (`redact_log_line` collapses
home paths and secrets there, but a query is neither).

**Blast radius that can be proven: the live terminal and the in-app Logs
view.** Not a log file a user attaches to a bug report. The fix-at-the-sink
prior art this repo already has (ADR-029's admission boundary; TASK-19555's
`redact_log_line`) applies here and was *already applied* — that is why this
was a defence-in-depth fix and not an incident.

### AC#2/#3/#4 — the sweep, and what was changed

Swept by AST over all 408 `RAG_Search` diagnostics. Twelve statements rendered
user content; all twelve were fixed, and three of them were worse than the one
filed:

| site | level | was | now |
|---|---|---|---|
| `rag_service.py` cache hit | INFO | `query[:50]` | `query_fp` + `chars` |
| `rag_service.py` FTS5 failure | ERROR | the **whole** query | `query_fp` + `chars` + `{e}` |
| `simple_cache.py` ×8 | DEBUG | `query[:50]` | `query_fp` + the cache `key` |
| `reranker.py` parse failure | ERROR | `response[:200]` | type + `{e}` + `chars` + `response_fp` |
| `reranker.py` unexpected format | WARNING | the **whole** response dict | `keys=[...]` |

The last row was found by the census after both greps had missed it (neither
regex included "response"), and it dumped an entire provider response — the
model's text about the user's query and 500 characters of their document — at
WARNING, untruncated. `pointwise` is the DEFAULT reranking strategy, so this
is live product code.

The replacement handle is `Utils/log_sanitizer.content_fingerprint`. Its
docstring states the guarantee precisely: it removes plaintext, it is **not**
secrecy against someone who already holds the log. It is deliberately
unsalted, because a per-run salt would destroy the across-restart correlation
the handle exists to provide. Truncation was strictly worse than the
replacement in both directions: it kept the words *and* lost the identity, since
two long queries sharing a prefix printed identically.

Left alone, with reasons:

* **Paths** (`db_path`, `persist_directory`, `config_file`, `cache_dir`,
  `path.name` in `config_profiles.py`) — the path IS the diagnostic ("database
  not found at X"); hashing it destroys the line. They reach no persistent
  sink, and the one route off the machine (the Logs export) already runs
  `redact_user_paths`, which strips the account name — the identifying part.
  This is the same fix-at-the-sink argument, already shipped.
* **`{e}` in the FTS5 and reranker lines** — kept. The exception is the
  diagnostic; the query never was. What reaches SQLite is `escaped_query`,
  per-token quoted and pinned by `test_fts5_query_escaping.py`.
* **Identifiers and counts** (`doc['id']`, `doc_id`, `chunk_id`, `len(texts)`,
  `model_name`, `collection_name`, `profile_id`, `type(e).__name__`) — code-side
  handles, not user words.
* **No document or chunk BODY was found interpolated anywhere in the tree**;
  every `doc`/`chunk`/`text` hit was a length or an id.

### AC#5 — the guard

`Tests/RAG_Search/test_rag_diagnostic_privacy.py` (17 tests):

* a **census** over all of `RAG_Search`, allowing `len(value)` and
  `content_fingerprint(value)` and rejecting every other rendering — it is what
  found the `reranker.py` dump;
* a vacuity guard (`test_census_actually_scans_the_tree`), because an empty
  finding list is the PASS condition and a blinded scanner would pass forever;
* classifier boundary cases, including the exact statement this task was filed
  for;
* **real behavioural** tests: the sync and async cache paths through
  `SimpleRAGCache`, and the reranker parse-failure path through
  `PointwiseReranker` with a signature-bound fake at the single provider seam.
  Each asserts both halves — no user words, AND a usable handle still present,
  so a future "fix" that just deletes the value fails too.

Mutation-tested: 14 deliberate breaks, all detected (see PR notes).

### Files

`tldw_chatbook/Utils/log_sanitizer.py` (new `content_fingerprint`),
`tldw_chatbook/RAG_Search/simplified/rag_service.py`,
`tldw_chatbook/RAG_Search/simplified/simple_cache.py`,
`tldw_chatbook/RAG_Search/reranker.py`,
`Tests/RAG_Search/test_rag_diagnostic_privacy.py` (new),
`Docs/security/production-diagnostic-inventory.json` (3 rows, all privacy
improvements; counts unchanged at 12/68/19, no sink-topology change).

### Out of scope, found on the way

* The same census over the rest of `tldw_chatbook` reports 90 candidate sites.
  The notable ones: `UI/Console_Modules/workspace.py` logs the Console
  conversation-browser **search query** with `!r` at EXCEPTION level;
  `UI/MediaWindow_v2.py` and `LLM_Calls/LLM_API_Calls.py` dump whole provider
  responses/`response.text` at ERROR. (Many of the 90 are classifier false
  positives on `response.status_code`.)
* `reranker.py`'s parse-failure handler references `response` in an `except`
  that can be reached before `response` is bound (`_call_llm` raising
  `ValueError`), which would raise `NameError` inside the handler. Pre-existing;
  untouched.
* Importing `tldw_chatbook.RAG_Search.reranker` as the first RAG module raises
  `ImportError` from a `reranker -> simplified/__init__ -> enhanced_rag_service_v2
  -> reranker` cycle. All three edges exist at the base commit; pre-existing.

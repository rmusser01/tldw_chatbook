---
id: TASK-19558
title: >-
  Security primitives the repo already owns are not adopted at five seams
  (FTS5 quoting, defusedxml, log_sanitizer, path redaction, risk tags)
status: Done
assignee: []
created_date: '2026-08-21 20:08'
labels:
  - security
  - db
  - tech-debt
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 2 (security & privacy) — Tier 3 items
**#7, #8, #9, #10, #11**. Grouped because they share one disposition: *the
correct primitive exists in this repo and these call sites do not use it.*
All re-verified at this branch base.

**1. FTS5 quoting — including three sites where the protection is a dead
store.** This is the sharpest of the group. Several `ChaChaNotes_DB.py` search
methods compute a quoted term and then **bind the raw one**:

```
11322:        safe_search_term = f'"{content_query}"'
   ...
11335:        params_list: List[Any] = [content_query]      # ← raw, not safe_

 7112:        safe_search_term = f'"{search_term}"'
   ...
 7122:        cursor = self.execute_query(query, (search_term, limit))   # ← raw
```

`safe_search_term` is used only in the error message. It **reads as protection
in code review and is not protection at all** — the same shape also at
`:9253`. Separately, the quoting that *is* applied never doubles embedded `"`
characters (`:13293`, `:12045`, `:12189`), permitting column-filter injection
into the MATCH expression and raising on ordinary input. Consequences already
visible: the live Library notes filter **swallows the exception and returns
silently wrong results**, and the Study flashcard search box raises a bare
`OperationalError`. Equivalent sites exist in the Evals, Media and RAG
searches.
The repo already owns `Utils/fts5_match_forms.quote_fts5_token` — currently
used only by `RAG_Search/simplified/rag_service.py`.

**2. OPML import uses stdlib etree.**
`Subscriptions/watchlist_opml_service.py:3` — `import xml.etree.ElementTree as
ET`, while **every sibling parser uses `defusedxml`**, which is a core
dependency. The lane probed this: the exposure is billion-laughs entity
expansion, **not** XXE. One-line fix. CONFIRMED live.

**3. `log_sanitizer._is_sensitive_log_key` computes a normalized key and never
uses it for the suffix rules.** `Utils/log_sanitizer.py:31-32` builds
`normalized` (lowercased, hyphens→underscores) but passes the **raw** key to
`is_sensitive_config_key`, using `normalized` only for an exact-membership
check. Proven by probe to return False for `Ocp-Apim-Subscription-Key`,
`X-Subscription-Token`, and bare `key`. This is **LATENT** — the module has
exactly one call site in the whole package
(`Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py:82`) —
but it is a defect in a primitive that TASK-19555 may want to lean on.

**4. Agent-supplied paths are echoed at WARNING.** `validate_path` prints the
full path unless `redact_paths=True`, and only 9 of 30 call sites pass it.
`Tools/local_tool_impls.py:53` omits it, so model-supplied paths are echoed.

**5. Local-tool risk-tag asymmetry.** `fs_read`/`fs_list`/`fs_glob`/`fs_grep`,
`web_*` and `watchlists_*` declare `tags=()` in
`Agents/local_tool_provider.py` (lines 1054, 1084, 1202, 1234, 1257, 1300…)
while the equivalent builtins declare `("reads",)`. The documented rationale
for the split is untrusted MCP-supplied tags — which **does not apply to local
tools, whose tags we author ourselves**. Untagged tools are not floored to
`ask`.

## Acceptance Criteria

- [x] Every FTS5 search path binds the value it actually quoted — no call site
      computes a `safe_*` term and then binds the raw one
- [x] FTS5 term quoting doubles embedded `"` characters, via the existing
      `Utils/fts5_match_forms.quote_fts5_token` rather than a new local helper
- [x] A search term containing `"` and a column-filter expression returns
      correct results (or a clean user-facing error) instead of silently wrong
      results in the Library notes filter or a bare `OperationalError` in the
      Study flashcard box
- [x] The Evals, Media and RAG search sites are fixed in the same sweep
      (RAG was already correct — it is the primitive's one pre-existing
      caller; Evals and Media are fixed)
- [x] A test fails if a `safe_*`-style dead store is reintroduced at any FTS5
      call site
- [x] `watchlist_opml_service.py` parses with `defusedxml`, and a test pins
      that an entity-expansion OPML payload is rejected
- [x] `_is_sensitive_log_key` applies its suffix rules to the normalized key;
      `Ocp-Apim-Subscription-Key`, `X-Subscription-Token` and bare `key` are
      recognised as sensitive, pinned by test
      (the first two were repaired by TASK-19555 before this task started;
      bare `key` was still live and is fixed here)
- [x] `Tools/local_tool_impls.py` passes `redact_paths=True`; the remaining
      `validate_path` call sites that handle untrusted paths are swept
- [x] The local-tool risk-tag decision is made explicitly: either the read-only
      local tools carry `("reads",)` like their builtin equivalents, or the
      rationale for the asymmetry is written down where the next reader will
      find it

## Implementation Plan

1. Read `Utils/fts5_match_forms.quote_fts5_token` and establish whether the
   primitive is itself correct before adopting it anywhere.
2. Census every hand-rolled FTS5 quoting spelling and every raw MATCH bind
   across `DB/`, `Library/`, `Notes/` and the UI layer; separate the sites
   that take PLAIN user text from the ones that take a caller-BUILT MATCH
   expression, because those need opposite treatment.
3. Reproduce both user-visible consequences at the branch base through the
   real search methods against a real throwaway database, and prove the
   dead store by mutation (mutate the computed value, show behaviour is
   unchanged; then mutate the fixed value, show it changes).
4. Adopt the primitive at every site; give the plain-text methods whose
   callers legitimately supply expressions an explicit `fts_match_query`
   seam, matching `search_notes`' existing shape.
5. Fix defusedxml, path redaction, and the log-sanitizer bare-`key` gap.
6. Settle the risk-tag question on measured behaviour rather than on the
   finding's premise.
7. Write two AST censuses (no re-spelled escape, no `safe_*` dead store) and
   an XML-parser census, each with bite-proofs in both directions.
8. Run the branch-relevant suites plus a repo-wide `--collect-only`.

## Implementation Notes

**`quote_fts5_token` was already correct.** It has doubled embedded quotes
since TASK-17755 (`'"{}"'.format(token.replace('"', '""'))`). Nothing in the
primitive needed fixing; the whole of finding 1 was non-adoption. Two thin
additions were made for the shapes the call sites actually needed, both
built on that one escape rather than beside it: `quote_fts5_phrase`
(a same-object alias — a whole phrase and a single token are the same FTS5
string literal, so a second implementation is exactly how the broken
spellings got written) and `quote_fts5_prefix` (`f"{quote_fts5_token(t)}*"`,
which four sites had spelled out longhand).

**The dead store, proven by mutation at base (72a82bc56).** Replacing the
computed `safe_search_term` with `"ZZZ_MUTATED_NEVER_MATCHES_ANYTHING_ZZZ"`
in all three methods left their results byte-identical
(`['Zed the Hunter'] / ['Talk about dragons'] / ['hello world']`). The same
mutation applied to the FIXED code returns `[] / [] / []`. The value now
reaches the query; before, it reached only the error message.

**Both user-visible consequences, measured at base.** Library notes filter:
`search_notes('alpha" OR title:"Other')` returned **2 rows** on a corpus
where one note contained "alpha" and the other contained neither term — the
closing quote ended the intended literal and the remainder ran as a live
FTS5 column filter; and `search_notes('foo"bar')` raised
`unterminated string`, which `library_screen._run_library_notes_filter`
swallows in `except Exception`, so the filter box silently did nothing.
Study flashcards: `list_flashcards(q='foo"bar')` raised a bare
`sqlite3.OperationalError` out of `flashcards_handler.refresh_cards`, which
nothing on that path catches. After the fix both return `[]`, and
`search_keywords('alpha"beta')` — which raised at base — now returns the
keyword actually named `alpha"beta`.

**Sites changed (`DB/ChaChaNotes_DB.py`, by post-change line):** import
74-78; `search_character_cards` 7688-7747 (dead store + new
`fts_match_query` seam); `_fts_prefix_match_expression` 7786;
`search_conversations_by_title` 9852-9880 (dead store);
`search_conversations_by_content` 9898-9972 (raw bind + new
`fts_match_query` seam); `search_messages_by_content` 12076-12106 (dead
store); `search_keywords` 12810-12824; `search_keyword_collections`
12966-12971; `_library_note_fts_query` 13090;
`_library_conversation_fts_query` 13369; `search_notes` 14068-14077;
`list_flashcards` 16253-16271; `search_flashcards` 17520-17526.

Other production files: `Utils/fts5_match_forms.py` (the two additions),
`DB/Evals_DB.py` (`search_tasks`, `search_datasets` — the latter wrapped the
RAW query with no doubling), `DB/Client_Media_DB_v2.py` (`search_media_db`'s
plain-text branch + `_library_fts_query`), `DB/Prompts_DB.py`
(`search_prompts`' plain-text branch, `search_prompts_by_text`,
`_library_prompt_fts_query`), `DB/Subscriptions_DB.py`,
`Library/library_fts_query.py`, `Library/library_local_rag_search_service.py`
(its conversations seam now uses the explicit `fts_match_query` seam its
notes/media/prompts siblings already used), `Notes/file_notes_replica.py`,
`UI/CCP_Modules/ccp_character_handler.py`, `UI/Screens/personas_screen.py`,
`UI/Console_Modules/prompts.py`.

**The seam distinction is the load-bearing design decision.** A method that
takes PLAIN user text now quotes it; a caller that genuinely owns an FTS5
expression passes it through an explicit `fts_match_query` parameter. Two
methods gained that parameter (`search_character_cards`,
`search_conversations_by_content`); `search_notes` and the media/prompts
siblings already had it. Without the distinction, quoting either breaks the
Library's widened keyword search or leaves the user-facing boxes injectable.

**Finding 2 (OPML).** `Subscriptions/watchlist_opml_service.py` now parses
with `defusedxml.ElementTree.fromstring` behind the same try/except shape
its five siblings use; stdlib ElementTree is retained for `export()`, which
only BUILDS a tree (defusedxml has no `Element`/`SubElement`). Measured at
base: a **573-byte** billion-laughs OPML parsed successfully and produced a
**300,000-character** outline name. After: `EntitiesForbidden`. The task's
framing is confirmed — it is entity expansion, not XXE, and a test re-proves
that stdlib ElementTree leaves a `SYSTEM` entity unresolved rather than
taking that on trust.

**Finding 3 (log_sanitizer) was two-thirds already fixed.** TASK-19555
landed the hyphen-normalization repair on dev before this task started, so
`Ocp-Apim-Subscription-Key` and `X-Subscription-Token` were already
recognised (probed, not assumed). Bare `key` was still `False`, and bare
`key` is how Google's Custom Search credential travels
(`?key=<API key>`) — the whole URL passed through `sanitize_string`
unchanged. Fixed by adding `key` to `_LOG_ONLY_SENSITIVE_FIELDS`, not to
`is_sensitive_config_key`, which also drives config encryption and the
Privacy & Security protected-field count. The cost is stated in the code:
a benign `key=<value>` label loses its value in diagnostics too.

**Finding 4 (path redaction) swept two choke points, not one.**
`Tools/local_tool_impls._resolve_in_workspace` (the ADR-032 local tool
family) AND `Utils/path_validation.validate_path_multi` — the choke point
for the in-process builtin file tools, which called `validate_path` once
per root and so logged a model-supplied path several times per probe. Both
refusal messages are unchanged: they are built by the caller from its own
argument and are the model's recovery route. The remaining 16 un-redacted
call sites take paths the USER chose in a file picker (OCR backends,
character-card import, prompt import), where redaction would remove
actionable diagnostics from the person who typed the path; that boundary is
stated rather than swept silently.

**Finding 5 (risk tags): the finding's premise is wrong, and adding the tag
would have been the defect this task exists to remove.** Local tools are
resolved by `permission_store.resolve_effective_state` (wired through
`UnifiedControlPlaneService.gate_tool_test`), whose floor set is
`HIGH_RISK_TAGS = {"mutates", "process"}`. `"reads"` lives in
`BUILTIN_HIGH_RISK_TAGS`, consulted only by `resolve_builtin_state`, which
serves `agent:builtin` and never `local:__local__`. So tagging `fs_read`
`("reads",)` would floor nothing — a marking that reads as protection and
provides none, i.e. the `safe_search_term` shape. The AC's second option was
taken: the rationale is written at the `LocalToolSpec` definition site (where
the next reader looks), and three tests in
`Tests/Agents/test_local_tool_provider.py` DEMONSTRATE the mechanism —
`("reads",)` leaves an inherited allow at `allow` while `("mutates",)` floors
it to `ask`, and a fresh permission store leaves every local tool at `ask`
anyway because `local:__local__` has no server entry. Moving the floor means
widening the MCP resolver's tag set, which was considered and rejected once
already (TASK-845) because MCP tags are server-supplied.

**The durable outcome: three AST censuses, each mutation-checked both ways.**
`Tests/Utils/test_fts5_quoting_adoption_census.py` — (1) no
`.replace('"', '""')` outside `Utils/fts5_match_forms.py`, with one
exemption keyed on `(module, function)` so an FTS5 helper cannot be parked
in `sql_validation.py` and inherit it; (2) no `safe_*`/`quoted_*`/`escaped_*`
local whose every read is inside a logging call. Census 2 run against the
real base file rediscovers exactly the three dead stores and nothing else,
and that check ships as a test (`test_dead_store_census_rediscovers_the_
three_base_defects`, reading the base blob from git).
`Tests/Subscriptions/test_watchlist_opml_entity_expansion.py` — (3) no
module in `Subscriptions/` calls a stdlib-etree parse entry point without a
defusedxml import. Scope is stated honestly rather than allowlisted: a
repo-wide version of census 3 currently reports **seven** other unhardened
parsers (`Evals/eval_runner.py`, `Local_Ingestion/XML_Ingestion.py`,
`Media/local_media_reading_service.py`,
`Research_Interop/academic_providers.py`, `Utils/file_extraction.py`,
`Web_Scraping/Article_Extractor_Lib.py`,
`Web_Scraping/Article_Scraper/crawler.py`) — a real, separate population
outside this task, deliberately NOT allowlisted, so widening the census is
what turns them red.

**Born-red evidence.** The three new test files run against base
72a82bc56: **18 failed + 1 collection error**; on this branch: **0 failed**.
(One of the 18, `test_the_primitive_actually_doubles_embedded_quotes`, fails
at base only because the two new helper names do not exist there — the core
`quote_fts5_token` behaviour it asserts was already correct.)

**Tests changed rather than added, and why.** Four test doubles gained the
new `fts_match_query` parameter (`Tests/Library/test_library_rag_scope.py`,
`Tests/UI/test_personas_workbench.py`), now asserting on the expression that
reaches SQLite rather than on the positional argument.
`Tests/Prompts_DB/test_prompts_db_legacy.py::test_search_with_invalid_fts_
syntax_raises_error` was INVERTED: it asserted that typing `invalid "syntax`
into the prompt search box raises `DatabaseError`. That was never a
contract, it was the symptom — the raw query was bound to MATCH. It is now
`test_search_with_a_typed_quote_is_matched_literally`, plus a new test that
the caller-built-expression seam still works.

**Test counts.** `Tests/DB` + `Tests/ChaChaNotesDB` + `Tests/Subscriptions`
+ `Tests/Utils` + `Tests/Library`: baseline 5544 passed / 4 skipped / 0
failed → after **5600 passed / 4 skipped / 0 failed**. `Tests/Evals`,
`Media`, `Media_DB`, `RAG_Search`, `RAG`, `Prompts_DB`, `Study_Interop`,
`Notes`, `Character_Chat`, `Agents`, `MCP`, `Tools`: **10099 passed /
16 failed**, the failure set byte-identical to the pre-existing baseline
(MCP documentation-contract inventory, a media migration red, an Evals
import census, a RAG_Search inflection test). Affected UI files
(`test_ccp_handlers`, `test_chat_search_enhanced`, `test_console_prompt_
picker`, `test_console_prompts_controller`, `test_media_viewer_prompt_
search_15477`, `test_study_flashcards_screen`, `test_library_prompts_canvas`,
`test_personas_workbench`): **775 passed / 2 failed / 1 collection error**,
identical to the same set at base (both `test_library_prompts_canvas`
failures and the `test_console_prompt_picker` collection error reproduce on
base). Repo-wide `--collect-only -q`: **56,915 tests collected**, 1 error —
`Tests/UI/test_library_file_notes_workspace.py`, the known dev red
(TASK-20972).

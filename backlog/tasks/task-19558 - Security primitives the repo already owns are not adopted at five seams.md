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

## Implementation Notes — review round (2026-08-23)

Independent review returned **fix-then-ship** with six items. The disposition
of finding 5 and the inverted Prompts test were independently confirmed; the
rest of this section is what changed.

**1 — the headline the first round shipped and did not name: a multi-word
recall narrowing at eight seams.** Quoting each seam's whole query as ONE
FTS5 phrase closed the injections and also made the words CONTIGUITY-bound.
This repo had already learned that once: `RAG_Search/simplified/rag_service.
_escape_fts5_query`'s docstring records TASK-3995 discovering that
whole-query phrase quoting "is strictly stronger than AND-of-terms, not
equivalent to it". Round one re-created that defect while removing a
different one.

Measured on one corpus (two records per seam: words adjacent / words split),
`dragon lore`:

| seam | base 72a82bc56 | round 1 | now |
|---|---|---|---|
| `search_character_cards` | 2 | **1** | 2 |
| `search_conversations_by_title` | 2 | **1** | 2 |
| `search_conversations_by_content` | 2 | **1** | 2 |
| `search_messages_by_content` | 2 | **1** | 2 |
| `list_flashcards` | 2 | **1** | 2 |
| `search_flashcards` | 2 | **1** | 2 |
| `Prompts_DB.search_prompts` | 2 | **1** | 2 |
| `Prompts_DB.search_prompts_by_text` | 2 | **1** | 2 |

The other half of the bar, on the same corpus and the same run: every
injection probe still returns **0 rows at all 14 seams** — column filter
(`dragon" OR title:"Other`), bare quote (`foo"bar`), and a typed operator
(`dragon OR zzznomatch`, which really executed under the base raw bind and
returned 2-3 rows). Recall was restored without trading the closure back.

The fix routes the plain-text branch through the repo's existing form:
`fts5_query_tokens` + per-token `quote_fts5_token`, joined by FTS5's
implicit AND — added to `Utils/fts5_match_forms` as
`build_and_match_expression` / `build_and_match_query`, and
`rag_service._escape_fts5_query` now builds its expression with the same
function instead of its own copy of the join.

**The rule applied, stated once rather than left to be inferred from
thirteen call sites: a seam that bound its query RAW (FTS5 implicit AND)
gets `build_and_match_query`; a seam that already bound a quoted PHRASE
keeps `build_phrase_match_query`.** That restores every regression round one
introduced and introduces no unmeasured behaviour change of its own —
`search_notes`, `search_keywords`, `search_keyword_collections` and both
Evals seams matched phrases before this task too, so they still do.
`test_the_phrase_seams_are_deliberately_left_as_phrases` pins that as a
decision, and shows the AND form would have returned 2 rather than 1.

A consequence stated rather than buried: the base docstrings' "Supports FTS
query syntax" promise is retired at these seams — a typed `dragon*` or `OR`
is now literal text. That is the point of the fix, and the three docstrings
that still advertised the old contract were retired in `51a42ca98`.

**2 — E1, a NUL byte.** `sqlite3` hands a bound TEXT parameter to SQLite as
a C string, so the value is truncated at the first NUL **after** quoting:
`"a\x00b"` arrives as `"a` and FTS5 raises `unterminated string`. No correct
quoting survives it — the closing quote is past the truncation point — and
raw binds escaped only by luck, the truncated `a` still being a valid
bareword. Nine seams went rows→raise. `Notes/file_notes_replica.search` had
guarded this since it was written; that rule is now
`fts5_query_is_searchable` and every seam shares it.

**3 — E2, `None`.** Quoting `None` raised a bare `AttributeError`, not even
wrapped in `CharactersRAGDBError`. Fixed in the same predicate, plus
`isinstance` guards ahead of the two `.strip()` calls that ran before it.
`Evals_DB.search_tasks(None)` raised `TypeError` at base too and is swept
here as the last seam in the family with that shape.

After both: the 126-cell probe matrix has **zero** raising cells; base had
17 and round one had 19.

**4 — the XML census now exists.** Round one scoped it to `Subscriptions/`
and claimed the other seven parsers were "not allowlisted so widening turns
them red" — describing a widening that had not happened, so nothing could
red and only memory stopped an eighth. It is now repo-wide with
`_KNOWN_UNHARDENED: dict[str, str]`, a **register of open defects** naming
each of the seven and what reaches it, plus
`test_known_unhardened_entries_are_still_unhardened` and
`test_the_register_matches_the_measured_population_exactly` so an entry
cannot outlive its defect. Bite-proof both ways, by real in-tree mutation: a
synthetic eighth parser reds two tests; hardening
`Research_Interop/academic_providers.py` reds two others.

**For filing (out of scope here, four take untrusted input today):**
`Evals/eval_runner.py:1842` (parses MODEL OUTPUT — prompt-injection
reachable), `Research_Interop/academic_providers.py:218` (remote Atom feed),
`Web_Scraping/Article_Scraper/crawler.py:398` and
`Web_Scraping/Article_Extractor_Lib.py:1063` (fetched sitemaps; a byte cap
is no defence — amplification is the point), plus
`Local_Ingestion/XML_Ingestion.py` (same threat shape as the OPML importer),
`Media/local_media_reading_service.py` (`iterparse` still expands internal
entities) and `Utils/file_extraction.py`.

**5 — guard-limitation honesty, and a third census.** Census 1's docstring
now states plainly that it fires on the CORRECT escape hand-rolled
(`.replace('"', '""')`) and is **blind to the spelling that actually caused
this task** (`f'"{x}"'`, which contains nothing to match). A repo-wide
detector for that shape was tried and rejected on measurement: four false
positives (SQL identifier quoting, UI copy) and zero true ones. What is
available structurally is **census 3** — a module that binds a parameter to
`... MATCH ?` must import `Utils/fts5_match_forms`; seven modules do and all
seven import it. It catches BOTH spellings in a new file, proven by an
in-tree mutation with the broken one. Its blind spot is stated too: a new
seam inside one of the seven already-importing modules is not covered.
Census 2's docstring now names `logger.opt(exception=True).error(...)` — this
repo's house style — as a real evasion of its logging-root walk.
`test_dead_store_census_rediscovers_the_three_base_defects` no longer
`skip`s: it raises with instructions. It keeps a PINNED revision rather than
`git merge-base HEAD origin/dev`, because merge-base moves on rebase and
would eventually resolve to a commit where the defects are already fixed —
turning the one test that proves the census detects anything into a failure
whose obvious "repair" is deletion.

**6 —** the `test_personas_workbench` stub comment now says what is true:
when `fts_match_query` is supplied, `search_term` is unused, including by
the error message.

**Born-red at the reviewed HEAD `51a42ca98`:** 11 of the new tests fail
there (6 recall seams, the phrase-seam decision, injections-under-AND, both
NUL tests, the None sweep). Census 3 and the widened XML census pass at that
HEAD by design — they harden rather than fix, and saying otherwise would be
overclaiming.

**Merge check against current dev (`ae018308b`).** Verified by merging this
branch into a throwaway worktree rather than by reading the textual merge:
the only conflict is `lessons-testing-evidence.md` (both sides append), and
`ChaChaNotes_DB.py` auto-merges across dev's `messages_fts` rebuild
(TASK-21100). On the merged tree `Tests/DB` + `ChaChaNotesDB` + `Utils` +
`Library` + `Media_DB` + `Prompts_DB` = **5121 passed / 1 failed**, that one
being a pre-existing baseline red — so the clean textual merge is a clean
semantic merge too. Also measured there: dev's TASK-21160 fixes the
`config_profiles`↔`simplified` cycle, so the priming import in
`Tests/Utils/test_agent_path_redaction_adoption.py` becomes redundant on
that dev and is annotated to be dropped then; it is still required at this
branch's base.

**Test counts (review round).** `Tests/DB` + `ChaChaNotesDB` +
`Subscriptions` + `Utils` + `Library`: **5618 passed / 4 skipped / 0
failed** (round 1: 5604). `Evals`, `Media`, `Media_DB`, `RAG_Search`, `RAG`,
`RAG_Eval`, `Prompts_DB`, `Study_Interop`, `Notes`, `Character_Chat`,
`Agents`, `MCP`, `Tools`: **10457 passed / 16 failed**, the failure set
byte-identical to the recorded 72a82bc56 baseline. Affected UI set: **775
passed / 2 failed / 1 collection error**, identical to that set at base.
Repo-wide `--collect-only -q`: **56,933 collected**, 1 error
(`test_library_file_notes_workspace.py`, TASK-20972).
`scripts/preflight.sh`: all green.

## Implementation Notes — review round 2 (Qodo on PR #2006, 2026-08-23)

Qodo returned five findings on the reviewed HEAD `cfc9e2604`. Three were
fixed, two declined with evidence. (CodeRabbit posted only a "review skipped —
auto reviews are disabled on non-default base branches" notice; nothing to
action there.)

**Finding 1 (High, bug) — `search_media_db` forced zero rows. FIXED, and it
was the same defect class round 1 was convened to fix.** When no MATCH
expression could be built, the code appended `"0"` to a `" AND ".join`ed
condition list, so the whole query returned nothing even though the LIKE legs
beside it could still express the intent. Measured on a five-row corpus, the
merge-base and this branch loaded side by side in one process (`git show
<merge-base>:…/Client_Media_DB_v2.py` imported under a dotted name inside the
real package, so its relative imports resolve without a second worktree);
re-run unchanged after the final rebase onto `12931e1a3`:

| input | dev (merge-base) | branch as reviewed | now |
|---|---|---|---|
| `!!!` | RAISE DatabaseError | 0 rows | **1** — "Alert!!! urgent dragon" |
| `   ` | 0 rows | 0 rows | **5** — whitespace-only = empty search |
| `""` | 0 rows | 0 rows | **1** — 'Quotes "" doubled' |
| `-` | RAISE DatabaseError | 0 rows | **1** — "well-known dashes" |
| `***` | RAISE DatabaseError | 0 rows | **1** — 'The "gold" standard' |
| `` (empty) | 5 rows | 5 rows | 5 rows |
| `dragon` (control) | 1 row | 1 row | 1 row |
| `  dragon  ` (control) | **0 rows** | 0 rows | **1** |

The last row is a pre-existing narrowing the strip incidentally fixes: the
LIKE leg is AND-ed with the FTS leg, so `%  dragon  %` vetoed the row `MATCH`
had already found.

The fix branches on WHY the builder returned empty rather than on the fact
that it did. A caller-owned `fts_match_query` that came out blank stays hard
false — `""` means "no rows" by the shared builders' contract, and the
title/content LIKE legs are deliberately not built in that branch, so dropping
the condition would return the entire table. A NUL stays hard false too: SQLite
truncates the bound parameter at the NUL, so `%dragon\x00lore%` reaches LIKE as
`%dragon`, which is WIDER than what was typed (it returned the dragon row at
the merge-base). Only punctuation-only text drops the leg — along with the FTS
JOIN and relevance ordering, which are now added inside the same branch that
adds the MATCH rather than unconditionally.

**The closure was not traded for the recall.** Re-run at `search_media_db`
after the fix, every probe in this branch's existing set still returns 0 rows:
`dragon OR zzznomatch`, `alpha" OR title:"Other`, `foo"bar`, `dragon" OR
"lore`, `title:dragon`, `dragon NEAR lore`, `dragon*`, and `dragon\x00lore`
— while `dragon lore` still returns its row. Nine new tests in
`Tests/DB/test_fts5_quoting_search_seams.py` pin all of it, including the
caller-owned-empty-expression case and the NUL case in both directions.

**Finding 5 (Medium) — docstrings claiming "literal phrase" at AND-form seams.
FIXED, and swept beyond the three Qodo named.** The sweep was mechanical: an
AST pass over every production file this branch touches, listing each function
that calls an `fts5_match_forms` builder, the FORM it actually builds, and
what its docstring claims. Corrected:

- `ChaChaNotes_DB.search_conversations_by_title` — "literal phrase" → AND (Qodo)
- `ChaChaNotes_DB.search_conversations_by_content` — three claims, incl. the
  `fts_match_query` "whole-phrase quoting" line → AND (Qodo)
- `prompt_scope_service.search_prompts` — "quoted as a literal FTS5 phrase by
  `PromptsDatabase.search_prompts`" → AND (Qodo)
- `ChaChaNotes_DB.search_messages_by_content` — "literal phrase" → AND (**not
  named by Qodo**; same wrong claim, same file)
- `ChaChaNotes_DB.search_character_cards` — its `fts_match_query` line said the
  parameter "replaces the whole-phrase quoting of `search_term`" (**not named**)
- `Client_Media_DB_v2.search_media_db` — the parameter's inline comment said
  "quoted as a literal FTS5 phrase" while the body builds AND (**not named**)
- `Utils/fts5_match_forms` module docstring — still said "**What is NOT here:
  the AND forms**", which TASK-19558 itself made false (**not named**). Replaced
  with the three forms and the rule for picking one.
- Four `_library_*_fts_query` helpers (ChaChaNotes ×2, Media, Prompts) —
  stated the quoting but never the join form; now say AND-of-quoted-tokens.
- `Prompts_DB.search_prompts_by_text` / `search_prompts_by_content`,
  `Evals_DB.search_tasks` / `search_datasets`, `file_notes_replica.search` —
  had no form claim at all; each now names its form (AND, PHRASE, PHRASE).

Verified after the edits by re-running the same AST sweep: no seam whose
builder is an AND form still says "phrase", and no phrase seam says AND.

**Finding 2 (Medium, rule) — `search_keyword_collections` docstring. FIXED.**
Full Google-style docstring with Args/Returns/Raises, and it states the form:
this seam is deliberately a PHRASE (it bound one before the task), unlike its
AND-form neighbours.

**Finding 3 (Medium, rule) — `defusedxml` bypasses `optional_deps`. DECLINED,
with evidence.** `defusedxml` is a CORE dependency: `pyproject.toml`
`[project] dependencies`, annotated `# engine xml security parsing (Q9:
core)`. `Utils/optional_deps.py` names it only inside the `ebook` extra's
aggregate availability check and publishes no import accessor for it; its
`get_safe_import` users are torch / transformers / huggingface_hub / aiohttp,
all genuinely optional. All six existing defusedxml call sites in the app
(`Subscriptions/security.py`, `Subscriptions/monitoring_engine.py`,
`Tools/web_tool_impls.py`, `Web_Scraping/WebSearch_APIs.py`,
`Chunking/engine/strategies/json_xml.py`,
`Event_Handlers/Chat_Events/chat_image_events.py`) import it directly under
exactly this `try/except ImportError` shape. Routing this one through the
helper would make it the odd one out, not the consistent one. The reasoning is
recorded at the import site so the next round does not re-raise it.

**Finding 4 (Medium, rule) — `search_character_cards` lacks `transaction()`.
DECLINED, on a census of its siblings.** Of the 26 read-only search/list
methods in `ChaChaNotes_DB.py`, 21 call `execute_query()` / `get_connection()`
with no `transaction()` — including every nearest sibling of this one
(`search_conversations_by_title`, `search_conversations_by_content`,
`search_messages_by_content`, `search_notes`, `_search_generic_items_fts`).
The 5 that do wrap are all PAGED Library seams
(`search_conversations_page`, `list_library_notes_page`,
`search_library_notes_page`, `list_library_conversations_page`,
`search_library_conversations_page`), and they wrap for a stated reason: the
row page and the exact total must be read from one snapshot. `execute_query`
is the designed read seam — it documents `commit=False` and joins an outer
`transaction()` when one exists — and `search_character_cards` is a single
statement with no cross-statement invariant. This branch did not change its
transaction shape; wrapping only this one would be inconsistency, not
compliance.

**Diagnostic inventory: held, then regenerated once it was only ours.** On
the first rebase (onto `736359202`) the rebuild reported four drifted rows —
`chachanotes_fts_backfill.py` +3, `app.py` +3, `ChaChaNotes_DB.py` +3 (all
dev's TASK-21100) and `Console_Modules/workspace.py` same-count-changed-digest
(dev's TASK-21118). Read with `--statements`: **zero** diagnostic statements
had changed in any of them on our side, so regenerating would have absorbed
dev's unreviewed drift into this PR, and the pin was left alone. Dev then
merged `12931e1a3` ("re-pin the inventory after TASK-21100's FTS backfill"),
which conflicted with ours; the conflict was resolved to dev's re-pinned
baseline and the rebuild rerun. It now names exactly two rows, both ours and
both read before writing: `Subscriptions/watchlist_opml_service.py` +1 (the
defusedxml-fallback `logger.warning`, a static string with no interpolation)
and `ChaChaNotes_DB.py` same-count-changed-digest (two `logger.error` strings
that now interpolate `match_expression` / `safe_search_query` where they used
to interpolate `safe_search_term` / `search_query` — the same user-query text
either way, no secret, path or URL newly exposed). `./scripts/preflight.sh`:
all five derived-artifact checks green.

**Test counts (round 2, rebased onto `12931e1a3`).**
`Tests/DB` + `Tests/ChaChaNotesDB` + `Tests/Utils` + `Tests/Subscriptions` +
`Tests/Prompts_DB`: **3593 passed / 2 skipped / 0 failed** (the same five
directories split as 2415/1 and 1178/1 on the intermediate base; round 1 was
2388/1 and 1178/1).
`Tests/Media_DB`: **78 passed / 1 failed** —
`test_reading_progress_reopens_through_versioned_migration`, a DEV red:
the v5→v6 migration (`ALTER TABLE UnvectorizedMediaChunks ADD COLUMN
chunk_engine_version`, dev commit `33c1ea0f8`, 2026-08-19) is not idempotent,
so a test that rewinds `schema_version` to 2 on a database that already has
the column dies on "duplicate column name". Neither this branch's 3-dot diff
nor its working tree touches a single line of migration code.
Repo-wide `--collect-only -q`: **57,069 collected**, 1 error
(`Tests/UI/test_library_file_notes_workspace.py`, TASK-20972 — known dev red).

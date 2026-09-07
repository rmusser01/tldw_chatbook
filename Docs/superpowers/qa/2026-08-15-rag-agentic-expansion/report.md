# TASK-16174 Phase E — answer-level oracle run (2026-08-15)

**Result: tool-OFF 0/8, tool-ON 7/8.** The gated `expand_document` tool is
what turns a label-only retrieval hit into an answer; without it the agent
retrieves the right document, says so, and cannot answer.

Artifacts in this directory: `questions.toml` (the pre-registered question
set, oracles and their corpus verification), `oracle_run.py` (the run),
`run-artifacts.json` (every query, tool call, tool result, answer, and the
spend, verbatim).

## What was run

Real `AgentService` loop -> real `chat_api_call` -> `api.anthropic.com`, over
a real, isolated installation of the retrieval stack built by the RAG-eval
harness (`Tests/RAG_Eval/harness/ingest.build_eval_runtime`, the whole
172-document fixture corpus written through the production writer APIs and
indexed through the production indexing helper; 166 of 172 indexed — the 6
prompts have no vector index by design, `UNINDEXED_SOURCE_TYPES`).

> **Two figures here are CONSOLE-OBSERVED, not archived** (final review,
> finding 8). "166 of 172 indexed" is printed by `oracle_run.py`'s
> `index_summary` and never persisted to `run-artifacts.json`, and the
> smoke-run cost below was likewise read off the console; the run's console
> log was not committed and cannot now be reconstructed. Both are stated as
> observations, not as artifact-backed numbers. Everything else in this
> report is recomputable from `run-artifacts.json` (the final review
> re-derived the spend, the byte costs and the 0/8 → 7/8 score
> independently, and every one reproduced exactly).

| | |
|---|---|
| Model | `claude-haiku-4-5`, temperature 0, non-streaming, `max_tokens=1024` |
| Retrieval route | `plain` — `config.search.default_search_mode = "plain"` sends the tool's `mode="rag"` request down `LibraryLocalRagSearchService._search_keyword`, the four-seam scope-aware keyword path |
| Questions | 8, fixed, pre-registered in `questions.toml` |
| Scoring | mechanical fact-oracle regex against the final answer text; no LLM grader |
| Arms | **OFF** = shipped default (`[tools] expand_document_enabled = false`, `allowed_tools = ("search_library_rag",)`); **ON** = gate flipped true, settings force-reloaded, provider rebuilt, `allowed_tools` gains `"expand_document"` |
| Everything else | identical: same corpus, index, route, system prompt, model, temperature, question order |

**Why `plain`.** That route is what produces the LABEL-ONLY rows this arc
exists for: `Matched media · document` / `Matched conversation · 1 message`,
`chunk_id: ""` (`_media_row`/`_conversation_row`). Under `hybrid` the engine
returns chunk text for every seam and there is nothing label-only to expand —
a different question. Pre-registered in `questions.toml`, not chosen after
seeing a number.

**Why the oracles avoid titles.** The agent sees the row's TITLE and the
label. Every oracle was grep-verified to appear in exactly one corpus
document's BODY, in no title anywhere, and not in its own question text
(re-runnable: `oracle_run.py --verify-oracles`).

## The table

| question | target (label-only row) | oracle | tool-OFF | tool-ON | ON tool calls |
|---|---|---|---|---|---|
| `q1-quillon-access` | `media-quillon-antenna` (media) | `/rescue certification/` | miss | HIT | search_library_rag, search_library_rag, expand_document |
| `q2-pellucid-lag` | `media-pellucid-gauge` (media) | `/lowest decade/` | miss | HIT | search_library_rag, search_library_rag, expand_document |
| `q3-ashgrove-seal` | `conv-ashgrove-pump` (conversation) | `/shimming/` | miss | HIT | search_library_rag, expand_document |
| `q4-obsidian-bearing` | `media-obsidian-lathe` (media) | `/brinelling/` | miss | HIT | search_library_rag, expand_document |
| `q5-dunnock-cooling` | `conv-dunnock-row-cooling` (conversation) | `/blown sand/` | miss | HIT | search_library_rag, search_library_rag, expand_document |
| `q6-gatehouse-ups` | `conv-gatehouse-power` (conversation) | `/(?:ninety\|90)\s*minutes/` | miss | miss | search_library_rag, search_library_rag |
| `q7-filling-head-mtbf` | `media-filling-head-reliability` (media) | `/(?:nine hundred\|900)\s+(?:running\s+)?hours/` | miss | HIT | search_library_rag, search_library_rag, search_library_rag, expand_document |
| `q8-larkspur-lubrication` | `media-larkspur-turbine` (media) | `/starved a bearing/` | miss | HIT | search_library_rag, expand_document |
| **TOTAL** | | | **0/8** | **7/8** | |

## Spend

| arm | model calls | input tokens | output tokens | USD |
|---|---|---|---|---|
| OFF | 25 | 62,663 | 2,620 | $0.0758 |
| ON | 29 | 87,033 | 2,770 | $0.1009 |
| **run total** | 54 | 149,696 | 5,390 | **$0.1766** |

A one-question smoke run (both arms) preceded it at **$0.0222**
(console-observed; the smoke run wrote no artifact — see the note above),
so the whole exercise cost **$0.199**, of which the $0.1766 above is
artifact-backed. Priced at `claude-haiku-4-5` list rates
($1.00/MTok in, $5.00/MTok out); no cache reads or writes were reported.
The ON arm costs ~33% more — expansion buys the answer, it does not come
free.

## Attribution: the one ON miss, and the four cleanest OFF misses

Every arm's FIRST search query was byte-identical to the other arm's on all
eight questions (temperature 0), so the arms are comparable at the point
retrieval begins.

**`q6` (the single ON miss) is a RETRIEVAL failure, not an expansion
failure.** Both of its searches returned `status: "empty"` in BOTH arms —
the model never had a row to expand. The four-seam keyword path ANDs every
query token (`build_fts_match_query`), and the model's own phrasings
("gate house UPS incomer drops live duration", "gate house UPS battery
backup time") over-constrain. The document IS reachable on this route: the
no-model probe retrieved it at rank 1 for `gate house UPS`. Conditioned on
the target row actually being returned, the ON arm expanded and answered
**7 of 7**.

**Four OFF questions saw the label-only row and still scored zero** — q1,
q3, q4, q8. This is the sub-result that does not depend on any search-
persistence difference between the arms, and the OFF answers say the
mechanism out loud, unprompted:

> "I found a reference to a 'Quillon-6 antenna mast survey' document in
> your Library. However, I can only see that it's a media document and
> cannot access its full contents through the search results." (q1, OFF)

> "I found a document titled 'Obsidian-3 lathe spindle teardown' in your
> library, but the search results are only showing the title without the
> detailed content." (q4, OFF)

On the other three (q2, q5, q7) the ON arm ran one more search than the OFF
arm and that is what surfaced the row. Disclosed as a confound in the ON
arm's favour: knowing a hit can be opened plausibly makes the agent persist.
It does not touch the four-question sub-result above.

## Mechanism evidence

- **The gate is the real one.** The OFF arm's built-in catalog is
  `['calculator', 'get_current_datetime']`; after the config rewrite +
  `load_settings(force_reload=True)` the ON arm's is
  `['calculator', 'expand_document', 'get_current_datetime']`. The tool
  appears because `BuiltinToolProvider.__init__` read `[tools]
  expand_document_enabled`, not because the script added it.
- **The risk-tag floor fired live.** `risk_tags=("reads",)` floors the tool
  to `ask`, and the run recorded exactly **7 approval rounds in the ON arm
  (one per `expand_document` call) and 0 in the OFF arm**. A headless run
  auto-approved them; a real user sees one card per call. That friction is
  a fact, not a defect.
- **The payload identity is what the agent used.** All 7 calls passed
  `source_type` + `source_id` verbatim from `_project_row`'s TASK-16174
  identity keys — e.g. `{'source_type': 'media', 'source_id': '3'}`,
  `{'source_type': 'conversation', 'source_id': '287e6e5b-...'}`. No
  `note_id`/`doc_id` fallback was needed and no call guessed. Task 3b's
  concern (a policy actionable only by inference would measure inference)
  does not apply to this run.
- **Every expansion returned `status: "ok"`.** No `not_found`, no `error`.

## TASK-3b's two pre-registered suspects: neither manifested

- **(a) `chunk_start` absent from the payload, so a chunked row expands from
  the HEAD.** Not observed. Every projected row in all 8 probes carried an
  empty `chunk_id` (`chunked_rows` was `[]` for all 8) — the `plain` route
  emits no chunked rows at all, so no expansion was mis-anchored. The
  suspect remains live for the `semantic`/`hybrid` routes, which this run
  does not exercise.
- **(b) a semantic `source_id` that is a vector-store point id, so a
  declared-expandable row returns `not_found`.** Not observed, for the same
  structural reason: the four-seam path's `source_id` is the real database
  id. Zero `not_found` results, live or in the probe.

Both are therefore **unrefuted rather than refuted**, and remain the first
things to check if the tool is ever measured on a semantic/hybrid route. A
follow-up task carries them (TASK-16588).

> **Fix-wave update (2026-08-15, after the final review).** Suspect (a)'s
> *cause* is now fixed, though the *measurement* is not: `_project_row`
> emits `chunk_start` whenever a row's provenance carries a usable anchor,
> so a chunked hit no longer expands from the document head by
> construction. That closes TASK-16588 AC#1 at the unit level only — this
> run still cannot exercise it, because the `plain` route emits no chunked
> rows, so AC#3's route measurement stands. Suspect (b) is untouched.

## Disclosed limits

- N = 8, one corpus, one model, one run per arm, temperature 0. No
  confidence interval is claimed.
- Oracle-fact inclusion is not answer quality: an answer can contain the
  fact and still be bad, and a correct paraphrase that avoids the oracle
  string scores as a miss.
- One route (`plain`). The semantic/hybrid routes are unmeasured here.
- Two runtime tools (`search_run_log`, `run_log_slice`) are offered
  regardless of `allowed_tools` — they are appended as `runtime_schemas`
  after the allow-list filter in `_run_one`. The OFF arm called
  `search_run_log` on q3 and q4; it reads the run log, never Library
  content, so it cannot have supplied an oracle fact, and the OFF arm
  scored zero regardless. **`q3` OFF ended `status: "stuck"`** (empty final
  text) after spending its steps on those calls — recorded, not scored
  differently.
- The embedding model is read (read-only) from the developer's real
  HuggingFace cache; downloads were blocked by forcing
  `huggingface_hub.constants.HF_HUB_OFFLINE = True`.

## Isolation, verified

- Scratch `HOME` / `XDG_CONFIG_HOME` / `XDG_DATA_HOME` / `TLDW_CONFIG_PATH`
  / `TLDW_TEST_MODE=1`, all set BEFORE any application import.
- `[database] media_db_path / chachanotes_db_path / prompts_db_path` in the
  scratch config point at the harness's scratch DBs, and the run asserts
  `config.get_*_db_path()` resolves to exactly those three files — this is
  what makes `expand_document` (which resolves handles through
  `config.get_*_db_lazy()`) read the eval corpus rather than the
  developer's own library.
- Real config `~/.config/tldw_cli/config.toml` sha256
  `42e2f42de95915f59fcc7c36751c2cc041cb0e4ddda9b324668683a270d7ecc0`
  **before and after — unchanged.**
- `tldw_chatbook` import provenance asserted inside the worktree.
- The API key was read from the git-excluded repo-root file at call time,
  passed as `chat_api_call(api_key=...)`, and never printed, logged, or
  written to any config.
- No TUI was launched, so `css/tldw_cli_modular.tcss` was never
  regenerated (`git status` clean apart from this directory).

## Reproducing

```
.venv/bin/python Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/oracle_run.py --verify-oracles
.venv/bin/python Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/oracle_run.py --dry-run
.venv/bin/python Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/oracle_run.py --live --confirm-billable
```

`--dry-run` builds the whole runtime and probes retrieval with zero model
calls; `--live` refuses to run without `--confirm-billable`.

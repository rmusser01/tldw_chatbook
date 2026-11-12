# S10 + S26 — Chunking/Embeddings/RAG_Admin; Workflows (+ the strays)

**Coverage S10** (70 files / 26,734 lines): read in full **1** · sampled **11** · mechanical only **58**.
**Coverage S26** (22 files / 7,676 lines): read in full **18** · sampled **3** · mechanical only **1**.

> **Scope fact that governs the S10 numbers — verified by the lead.**
> **38 of S10's 70 files (14,398 lines, 56%) are vendored from `tldw_server`** under
> `Chunking/engine/VENDOR_MANIFEST.toml`: *"NEVER hand-edit files listed here (spec §5.2). Re-sync:
> `python Helper_Scripts/sync_chunking_engine.py`"*. Manifest vs tree: 38 listed, 39 present, the only
> non-vendored file is `engine/__init__.py`. **Four findings below land in vendored files and must go upstream,
> not into a local PR**, and the lead's repo-wide D4 pass must exclude `Chunking/engine/**`.

> **S26 exists because the review's scope table omitted `Workflows/` entirely.** Note the package is
> **reviewed-never but tested-heavily** — `Tests/Workflows/` holds 17 modules / 6,978 lines plus 6
> `Tests/UI/test_workflows_*.py`. That is the most likely explanation for how clean it reads.

## Findings

### P1 [D1] — The OpenAI-compatible embedding backend makes raw `requests` calls to a config-supplied URL with no egress policy
- Where: `Embeddings/Embeddings_Lib.py:530` (`requests.Session()`), `:566-568` (`session.post(embeddings_endpoint,…)`),
  endpoint built at `:546-550` from `OpenAICfg.base_url` (`:258-260`).
- Evidence: `grep -n "egress" tldw_chatbook/Embeddings/*.py` → **NONE**. `base_url: Optional[HttpUrl]` validates URL
  *shape* only. Reachability traced: `EmbeddingFactory._build:728-742` dispatches `provider == "openai"` →
  `_openai_embedder`; `RAG_Search/simplified/embeddings_wrapper.py:420-432` builds
  `{"provider": "openai", …, "base_url": base_url}`; `config.py:4993` **ships `provider = "openai"` as a documented
  `[embedding_config.models.*]` option.** So a user-editable TOML value becomes an unguarded outbound destination.
- Why it matters: every other network path in this repo (50 importers of `Utils/egress.py`) is screened for
  link-local/loopback/private-range targets before the socket opens; this one is not, **and it carries an
  `Authorization: Bearer` header when a key is configured alongside the base URL.**
- Size: S · Confidence: verified (reachability traced; not exercised live)
- Already covered: **no** — TASK-32806 `.1`–`.8` contains no egress item.

### P2 [D1] — Six `logger.*` calls in `Embeddings_Lib.py` use printf-style args that loguru silently discards, including both retry/failure paths
- `:571, 581, 588, 647, 656, 889`. Proven against the venv's loguru:
  `logger.debug("openai_embed[%s] %d texts in %.3fs via %s", "m", 3, 0.1, "http://x")` renders
  **`openai_embed[%s] %d texts in %.3fs via %s`**.
- Why it matters: `:581` and `:588` are the `except requests.RequestException` handlers — the endpoint, the attempt
  number **and the exception `exc` itself** are all dropped, so a failing embedding backend logs a line with literal
  `%s` and no cause. · Size: S · Confidence: verified

### P2 [D1] — Eleven `logger.*(…, exc_info=…)` calls in vendored `Chunking/` produce no traceback
- `engine/chunker.py:172, 1583, 1715, 1828, 1912, 1920, 1931, 1980, 2252`; `engine/strategies/semantic.py:425`;
  `engine/strategies/structure_aware.py:251`. loguru probe: `logger.debug("boom", exc_info=True)` →
  `record["extra"] == {'exc_info': True}`, `record["exception"]` **unset**, no traceback emitted. The app's only
  sink format has neither `{extra}` nor `{exception}`.
- Why it matters: every one is an `except:` that discards the failure (tokenizer override failures, strategy
  `llm_call_func`/`language` wiring failures, structured-overlap extraction failure). **The largest such
  concentration outside `UI/`.** Fix is `logger.opt(exception=True)` — **all 11 are vendored → upstream.**
  · Size: S · Confidence: verified

### P2 [D1] — `SecurityLogger`'s dedicated audit-log sink can never match its own filter, and nothing installs it anyway
- `engine/security_logger.py:52-59` (sink with `filter=lambda r: "security" in r["extra"]`, format
  `{extra[event_type]}`) and `:91-99` (`logger.<level>(msg, extra=extra)`).
- Evidence: loguru probe — `logger.warning(msg, extra={"security": True, …})` yields
  `record["extra"] == {'extra': {'security': True, …}}`, so **the filter evaluates False**. Separately,
  `grep -rn "configure_security_logging" tldw_chatbook/` → **only its own definition**; every construction goes
  through `get_security_logger()` with `log_file=None`, so `logger.add(...)` never runs.
- Why it matters: **the XXE / ReDoS / oversized-input audit trail this module exists to produce has two independent
  reasons to be empty.** The message text still reaches the general app log (not a total blackout), but the
  structured `event_type` reaches no formatter — and if anyone wires the sink up, `{extra[event_type]}` would
  `KeyError`. Vendored → upstream. · Size: S · Confidence: verified
- Pinning test: `Tests/Architecture/test_security_logger_write_surface.py` pins the *write surface*, not delivery.

### P2 [D2] — Every keystroke in the Workflows editor re-parses and re-serializes the whole document ~6 times on the event loop
- `UI/Screens/workflows_screen.py:702-724 field_changed` → `project()` `:711` → `controller.edit_field:230-244` →
  `DocumentService.edit_field:589-644` (`_document` + `_serialize` + `_document` **again**) → `controller.validate()`
  → `render_draft`. Events from `editor.py:731 @on(Input.Changed)` / `:717 @on(TextArea.Changed)` — **per keystroke,
  no debounce.**
- **Measured** against `DocumentService` directly (one keystroke = `project` + `edit_field` + `dependency_issues` +
  `project`):
  ```
  10 steps / 3 KiB      ->  0.3 ms        100 steps / 1.9 MiB  -> 14.6 ms
  100 steps / 31 KiB    ->  2.1 ms        100 steps / 15.3 MiB ->   85 ms
                                          500 steps / 14.4 MiB ->   86 ms
  ```
  `MAX_DOCUMENT_BYTES = 16 MiB`, `MAX_DOCUMENT_STEPS = 500` — **85 ms is the admitted ceiling, not synthetic.**
  Draft *persistence* is debounced (`draft_session.py:16 DRAFT_DEBOUNCE_SECONDS = 0.5`); the
  parse/serialize/validate chain is not.
- Size: M · Confidence: **verified (measured)**
- Pinning test: `Tests/UI/test_workflows_projection_performance.py` pins `validate()` at **≤2 decodes** and a
  no-change re-render at **0 decodes** — **the repo has deliberately bounded two of the legs.** The un-pinned
  remainder is the screen's own `project()` and `edit_field`'s 2 decodes + 1 full encode, **which is exactly where
  the measured cost lives.**

### P2 [D2] — `step_label` rebuilds and sorts the entire 130-row discovery catalog once per step, on two surfaces, per refresh
- `UI/Workflows_Modules/controller.py:353-356`, called per step at `navigator.py:101`, `editor.py:360,619` and per
  neighbour at `editor.py:346,460,496`. `len(DISCOVERY) == 130`, `len(_LABELS) == 5`. **Measured: 2.95 ms per
  navigator refresh over 100 steps, and the overview list pays it again — ~6 ms per refresh**, on top of the
  document cost above. **The information wanted is a 5-row dict lookup.** · Size: S · Confidence: verified (measured)

### P2 [D2] — The workflow picker searches the database on every keystroke, with no debounce, against a full `workflow_heads` scan
- `UI/Workflows_Modules/library.py:294-297` → `DocumentService.list_workflow_summaries` → `_workflow_summaries:938-985`.
  With a non-empty query the `json_extract`/`json_type` SELECT runs **without `LIMIT`** and `fetchmany`-loops until
  the page fills — a full scan when the query matches few rows. The worker is correctly
  `@work(exclusive=True, group=…)` and off-loop, **but `exclusive=True` cancels the awaiting coroutine, not the
  thread already inside `to_thread`** — so an 8-character query queues 8 full scans.
- Why it matters: **this contradicts the repo's own established pattern.** The known-deliberate list records
  "Library has no per-keystroke DB search" and "browser-search debounce with cancellation token" as the convention;
  this surface has neither. · Size: S · Confidence: inferred (scan behaviour read, not measured)

### P2 [D3] — `get_common_embedding_models()` is unreferenced, and it ships a `trust_remote_code=True` model with no revision pin
- `Embeddings/Embeddings_Lib.py:938-1035`; the unpinned entry is `qwen3-embedding-4b` at `:1030-1037`.
  `grep -rn "get_common_embedding_models" tldw_chatbook/ Tests/` → only its own `def` and its `__all__` entry.
  **The drift is self-documenting:** its sibling `stella_en_1.5B_v5` (`:1021-1029`) carries
  `revision="4bbc0f1e…"` with the comment `# Pinned for security`; `qwen3-embedding-4b` has `trust_remote_code=True`
  and **no `revision`**.
- **Retired as a live RCE** — nothing can select that entry today. **It remains a loaded gun:** the helper is
  exported in `__all__`, and the moment anything wires it into a model picker, choosing that model executes whatever
  Python is at the HEAD of the Hub repo. · Size: S · Confidence: verified

### P3 [D1] — The Workflows editor renders "timeout unsets", and its own test asserts the typo
- `UI/Workflows_Modules/editor.py:468-478` — `timeout = … or "unset"` then `f"… / timeout {timeout}s"`.
  `Tests/UI/test_workflows_projection_performance.py::test_editor_field_values_and_summaries_use_prepared_projection`
  asserts `" · retry unset / timeout unsets"`. **The pinning test records the defect as the requirement, so a naive
  fix goes red and looks like a regression.** · Size: S · Confidence: verified

### P3 [D1] — The corrupt-tags warning drops the template name it exists to identify
- `Chunking/chunking_interop_library.py:801-805` — printf-style `%s` discarded. The operator is told a template is
  corrupt and **not which one**, with no other signal since the exception is swallowed. **Local file**, unlike the
  other 11. · Size: S

### P3 [D3] — `RAGAdminScopeService`'s off-loop hop is per-method opt-in; 20 of its 22 backend calls run the (entirely synchronous) local service inline
- `_call_off_loop` used at `:218`, `:323`; plain `await self._maybe_await(...)` at 20 other sites. The heaviest is
  `apply_template` (`:345`), which runs a full `Chunker.chunk_text()` on the calling thread.
- **Retired as a live defect:** tracing the consumers → only `list_templates(mode="local")` and
  `get_diagnostics(mode="local")` have callers, **and both are the two that were offloaded. TASK-32804.12 is
  complete for the live surface.** What remains is structural: the file's default is "inline", so the next UI wiring
  of any other method silently reintroduces sqlite/Chroma/chunking on the loop. · Size: M · Confidence: verified

### P3 [D3] — A vendored engine file carries a local hand-edit with no carry-forward record; the next re-sync reverts it
- `Chunking/engine/security_logger.py`, changed by `c255ec3936` (task-32803.1) to import
  `Utils.timestamps.utc_now_iso`. `git log d60ebe1d0a..HEAD -- Chunking/engine/` → **exactly one commit, that one.**
  `Tests/Architecture/test_vendor_pin_consistency.py` pins the *commit sha*, not per-file content.
- Why it matters: **the naive-`utcnow()` fix that ADR-173's guard exists to enforce lives in a file the sync script
  overwrites, with nothing recording that it must be re-applied.** · Size: S · Confidence: verified

### P3 [D3] — `assign_select_value` reads Textual's private `Select._options`
- `Widgets/select_values.py:44`. The module's docstring justifies mirroring Textual's internal check, so the coupling
  is deliberate — **but it is to a private attribute, not to the `_legal_values` it names.** This helper exists
  specifically to stop `InvalidSelectValueError` from killing the app; **a Textual upgrade that renames `_options`
  turns the guard itself into the crash.** · Size: S · Confidence: inferred

## The ruling on the open question blocking TASK-32808.6's ADR
**`RAG_Admin`'s fail-direction is correct. Standardize on "do not thread unless positively confirmed safe."**
Full argument and blast-radius caveat are reproduced in `report.md` §"Two corrections to this report's own Phase 3".

## Candidate triage
**RETIRED — the lead's dispatch hypotheses:**
- **`expressions.py` reaching `eval`/`exec`/`compile`/an unsandboxed template: retired.**
  `grep -rnE "\b(eval|exec|compile)\s*\(|jinja|Template\(|__import__|pickle|subprocess|os\.system"` over both
  packages → **only `re.compile` at `expressions.py:7,8`.** The grammar is `{{ a.b.c }}` dotted dict lookups only;
  every other delimiter (`{%`, `{#`) is refused; `_resolve` walks plain dicts; every leg is byte-budgeted.
  **A correct, closed evaluator.**
- **Chunker-version contract broken by an unbumped behaviour change: retired.**
  `git log d60ebe1d0a..HEAD -- Chunking/engine/` → one commit, a timestamp fix. No chunk-boundary code changed since
  the pin; `ENGINE_VERSION = "parity-1@385afa95"` matches the manifest's `upstream.commit`. **Contract intact.**
- **New `CREATE INDEX`/`CREATE TABLE` (the CLAUDE.md gotcha-1 obligations): retired — NONE** anywhere in
  `Chunking/ Embeddings/ RAG_Admin/ Workflows/ UI/Workflows_Modules/`. DDL lives in `DB/Workflows_DB.py`.
- **Embeddings downloaded-artifact size/checksum: retired.** Every download goes through
  `AutoTokenizer/AutoModel.from_pretrained`, which delegates integrity to `huggingface_hub`'s etag/sha verification.
  **The real artifact-trust gap is `trust_remote_code=True` without a `revision`** (P2), not checksums.
**RETIRED:** `fetchall_no_limit` `local_rag_admin_service.py:641` — a `GROUP BY chunk_engine_version` aggregate
bounded by the number of engine versions (currently 2); **its docstring pins the exact SQL text to a CI index-plan
pin, so re-spelling it would break the pin. Do not touch.**
`mutable_class_attr` `tokens.py:277` — a deliberate cross-instance failure cache guarded by a lock, documented in
place, and vendored. `except_exception_pass` ×2 — a best-effort NLTK corpus fetch and a
`# pragma: no cover - regex on str cannot fail`. `re_compile_in_def` ×22 — **21 of 22 vendored**; the one local row
compiles a per-recipe user pattern that must not be cached across recipes.
`raw_mkdir`/non-atomic write `engine/templates.py:546,719` — **confirmed as a TASK-32808.5 non-adopter but out of
local scope** (vendored; fix upstream).
`chunk_generator` ×4, `_coerce_bool_option` ×2, `tokenizer` ×2 — **retired as D4 targets: all members are vendored.**
`_maybe_await` — `rag_admin_scope_service.py:82` is a member and its sync call sites *would* be the instance the
lead wants, **but none is UI-reachable today.** Reported as P3 instead.
Dotted `get_cli_setting`, `run_worker(exclusive=True)` without `group=`, loguru+stdlib in one file — **all three
greps empty across both slices.**
**CONFIRMED:** task-32861 `set_status_line` — 17 importers vs 23 `_set_status` definitions; 6 non-adopters named.
*(Caveat the reviewer flagged: `Audio/diarizer_local.py`'s `_set_status` may not be a Textual status line at all —
S07 independently confirmed it is an integer progress callback on a non-Textual class, so the real count is 5.)*

## D4 observations for repo-wide Phase 3
1. **`Chunking/engine/**` must be excluded from the census.** Any hit whose members are all inside it is not
   actionable; a hit that *mixes* vendored and local members can only move the local copy.
2. **Duplicate-JSON-key rejection, 17 copies — helper exists, ignored.** Canonical home
   `Utils/input_validation.py:616`. **Constraint the lead needs before consolidating: each copy raises its own domain
   exception (`InvalidDraft`, `RecoveryRequired`, `ValueError`, `InvalidTemplateError`), so the shared helper needs
   an injectable exception type — a straight swap changes what callers catch.** TASK-32855's territory.
3. Scope-service scaffold — `RAG_Admin` contributes 6 members; **the new input is the fail-direction ruling, which
   the shared scaffold must encode as its default.**
4. **`RAG_Admin ↔ Evaluations_Interop` normalizer pair is a genuine 2–3 copy cluster outside the scaffold:**
   `_model_dump` ×3, `_dump_model` ×2, `_safe_int` ×2, `_coerce_json_mapping`/`_parse_template_payload` ×2.
   **Four helpers, one obvious home, both packages local and editable. Highest-yield S10 D4 target.**
5. `cancel@UI/Workflows_Modules/library.py:125` — a 3-line `event.stop(); self.action_cancel()`. **Not worth a
   helper; close that census row.**

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| The per-keystroke editor cost reproduces in the live app, not just against `DocumentService` | app must not be booted; **and `Tests/UI/test_workflows_*.py` all error at setup with `RecoveryRequired` (the ADR-126 gate) in this clean worktree** | run them from a worktree with the recovery participants installed, then a live capture with a 500-step document |
| The picker's per-keystroke scan is expensive at realistic library sizes | no populated store; scan cost read, not measured | seed N heads, time `list_workflow_summaries(query="zzz")` for N ∈ {100, 1000, 10000} |
| The Embeddings egress bypass is exercised by a real request | would need an `openai`-provider embedding model configured | `EmbeddingFactory({… "provider":"openai","base_url":"http://127.0.0.1:1/v1" …}).embed(["a"])` and observe no `check_url_or_raise` |
| `Workflows/draft_session.py` (618) and `UI/Workflows_Modules/run_controls.py` (603) hold no further defects | sampled/grep-scanned only. **Both are lifecycle-critical** — `draft_session` owns durable-write retention across screen lifetimes; `run_controls` owns the run-setup form feeding `WorkflowSession.prepare`'s security checks | full read, specifically `run_controls.py:245-320` (user-typed `Path` and model selections before `prepare`) and `draft_session.py:140-618` (flush/close/quit ordering) |
| `Chunk_Lib.py` (1,745 lines, 116 `legacy` markers) holds no further defects | read ~150 targeted lines. **It is the largest local (non-vendored) file in S10 and the shim between the app and the vendored engine** | full read; `git log --oneline -- …/Chunk_Lib.py \| head -20` to find what the 116 markers defer |

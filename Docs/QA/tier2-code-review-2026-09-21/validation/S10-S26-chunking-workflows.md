# S10 + S26 validation — Chunking/Embeddings/RAG_Admin; Workflows

Validated at worktree HEAD `d0face3ebe` (detached at `origin/dev`) against slice
`qa/tier2-code-review-2026-09-21/slices/S10-S26-chunking-workflows.md`.

## 1. P1 — OpenAI-compatible embedding backend makes raw `requests` calls with no egress policy
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Embeddings/Embeddings_Lib.py:530` (`requests.Session()`), `:566-568` (`session.post`), no line drift.
- Proof: `grep -n "egress" tldw_chatbook/Embeddings/*.py` → no hits. `RAG_Search/simplified/embeddings_wrapper.py:420-432` builds `model_config["base_url"] = base_url` from a caller-supplied `base_url` when `model_name.startswith("openai/")`, reaching `_openai_embedder` via `EmbeddingFactory._build` (`elif spec.provider == "openai": ... fn = _openai_embedder(spec)`). `config.py:4992-4997` ships a documented `[embedding_config.models.*]` example with `provider = "openai"` and a user-set `base_url`.
- Note: none — matches the review exactly, no drift.

## 2. P2 — Six printf-style `logger.*` calls in `Embeddings_Lib.py` silently discarded by loguru
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Embeddings/Embeddings_Lib.py:571,581,588,647,656,889` — exact match, zero line drift.
- Proof: `awk 'NR==571||NR==581||NR==588||NR==647||NR==656||NR==889'` shows all 6 are `logger.debug/error/warning/info` calls with `%s`/`%d` args. Independent loguru probe: `logger.debug("openai_embed[%s] %d texts in %.3fs via %s", "m", 3, 0.1, "http://x")` → sink message `'openai_embed[%s] %d texts in %.3fs via %s\n'` (literal, unsubstituted).
- Note: none.

## 3. P2 — Eleven `logger.*(..., exc_info=...)` calls in vendored `Chunking/` produce no traceback
- Verdict: CONFIRMED (vendored — needs upstream/carry-forward, not a local PR)
- Site now: `chunker.py:172,1583,1715,1828,1912,1920,1931,1980,2252`; `strategies/semantic.py:425`; `strategies/structure_aware.py:251` — all 11 lines confirmed present verbatim.
- Proof: loguru probe — `logger.debug('boom', exc_info=True)` → `record['exception'] is None`, `record['extra'] == {'exc_info': True}`. App's only sink format (`Logging_Config.py:611`) is `"{time}...{message}"`, no `{exception}`/`{extra}` placeholder. `chunker.py`, `strategies/structure_aware.py`, `strategies/semantic.py` are all listed in `Chunking/engine/VENDOR_MANIFEST.toml`.
- Note: none — matches review.

## 4. P2 — `SecurityLogger`'s audit-log sink can never match its own filter, and nothing installs it
- Verdict: CONFIRMED (vendored — needs upstream)
- Site now: `tldw_chatbook/Chunking/engine/security_logger.py:52-59` (filter), `:91-99`/`:220-234` (`get_security_logger()` default `log_file=None`).
- Proof: probe — `logger.warning('SECURITY EVENT: x', extra={'security': True, 'event_type': 'y'})` → `record['extra'] == {'extra': {'security': True, 'event_type': 'y'}}` (loguru's `extra=` kwarg is a format kwarg, not `record["extra"]` merge), so the sink's `filter=lambda r: "security" in r["extra"]` never matches. `grep -rn "configure_security_logging" tldw_chatbook/` → only its own `def`; both production constructors (`chunker.py:330`, `strategies/json_xml.py:721`) call `get_security_logger()` with default `log_file=None`, so `logger.add(...)` never runs at all.
- Note: none — both independent failure modes confirmed.

## 5. P2 — Workflows editor re-parses/re-serializes the whole document ~6× per keystroke, no debounce
- Verdict: CONFIRMED
- Site now: `UI/Screens/workflows_screen.py:702-724 field_changed` (unchanged), `controller.py:230-249 edit_field`, `Workflows/document_service.py:593-644 edit_field` (`_document` decode → `_serialize` encode → `_document` decode again, exactly as claimed), fired from `editor.py:734 @on(TextArea.Changed)` / `:751 @on(Input.Changed)` with no debounce visible in either handler.
- Proof: read of `document_service.py:607` (`_document(raw_json)`), `:629` (`raw = _serialize(document)`), `:630` (`_document(raw)` — second decode to re-validate). `DRAFT_DEBOUNCE_SECONDS = 0.5` (`draft_session.py:17`) governs only persistence, not this chain.
- Note: none.

## 6. P2 — `step_label` rebuilds/sorts the full 130-row discovery catalog per step, per refresh
- Verdict: CONFIRMED
- Site now: `UI/Workflows_Modules/controller.py:352-355 step_label`, calling `discover()` (`Workflows/catalog.py:346-372`), used at `navigator.py:101`, `editor.py:346,360,460,496,619`.
- Proof: `len(DISCOVERY)==130`, `len(_LABELS)==5` (measured via venv import). `discover()` builds a full `entries` list and `sorted(...)` on every call; `step_label` calls `discover()` and linear-scans it with `next(...)` for every step.
- Note: none.

## 7. P2 — Workflow picker searches the DB on every keystroke, no debounce, unbounded scan
- Verdict: CONFIRMED
- Site now: `UI/Workflows_Modules/library.py:298 search_page` (`@on(Input.Changed, "#workflow-page-search")`, calls `self.load_page(0, event.value)` unconditionally), `load_page` at `:251` (`@work(exclusive=True, group="workflow-choice-page")`), `Workflows/document_service.py:960-1003 _workflow_summaries` (non-empty `query` branch runs `cursor.execute(sql)` with **no** `LIMIT`, then `fetchmany`-loops).
- Proof: read confirms no debounce timer anywhere in `search_page`; `asyncio.to_thread(self.loader, offset, query)` at `library.py:275` — `exclusive=True` cancels the awaiting Textual Worker task, not the already-dispatched thread-pool call inside `to_thread`, matching the claim.
- Note: none.

## 8. P2 — `get_common_embedding_models()` unreferenced; ships `trust_remote_code=True` with no revision pin
- Verdict: CONFIRMED
- Site now: `Embeddings/Embeddings_Lib.py:938-1035`, unpinned entry `qwen3-embedding-4b` at `:1030-1036`.
- Proof: `grep -rn "get_common_embedding_models" tldw_chatbook/ Tests/` → only the `def`, its `__all__` entry, and a doc example in `Config_Files/EMBEDDING_DEFAULTS_README.md`. `stella_en_1.5B_v5` carries `revision="4bbc0f1e9df5b9563d418e9b5663e98070713eb8"  # Pinned for security`; `qwen3-embedding-4b` has `trust_remote_code=True` and no `revision=` key at all.
- Note: none.

## 9. P3 — Workflows editor renders "timeout unsets", and its own test asserts the typo
- Verdict: CONFIRMED
- Site now: `UI/Workflows_Modules/editor.py:468-478 _step_summary` — `timeout = ... or "unset"` then `f" ... timeout {timeout}s"`.
- Proof: `Tests/UI/test_workflows_projection_performance.py:111` — `assert editor._step_summary(0, "execution") == " · retry unset / timeout unsets"`, matching literally.
- Note: none.

## 10. P3 — Corrupt-tags warning drops the template name it exists to identify
- Verdict: CONFIRMED
- Site now: `Chunking/chunking_interop_library.py:801-805` — `logger.warning("Chunking template %s has a corrupt tags column; ...", row["name"])`.
- Proof: same loguru printf-discard mechanism as finding 2. File is not in `Chunking/engine/VENDOR_MANIFEST.toml` (not under `engine/`) — this is a local, directly-fixable file, unlike findings 3/4.
- Note: none.

## 11. P3 — `RAGAdminScopeService`'s off-loop hop is per-method opt-in; 20 of 22 backend calls run inline
- Verdict: CONFIRMED
- Site now: `RAG_Admin/rag_admin_scope_service.py` — `_call_off_loop` used at `:218`, `:323` (unchanged); `_maybe_await(` count = 26 total (includes `def` + its one internal use inside `_call_off_loop`).
- Proof: `grep -c "_call_off_loop("` = 2 (def + one internal call... actually 2 call sites at 218,323 plus the def = distinct); `grep -c "_maybe_await("` = 26.
- Note: **The review's reachability evidence is incomplete, though its P3/"not live" conclusion still holds.** Tracing all callers of `app.rag_admin_scope_service` (`grep -rln`) finds **three** UI call sites, not two: `Widgets/Library/library_ingest_canvas.py` (`list_templates`, offloaded via `_call_off_loop`), `Widgets/Library/library_search_rag_panel.py` (`get_template_diagnostics`, offloaded), and `Widgets/Library/library_rechunk_run.py` (`rechunk_legacy_media`, **NOT** offloaded — its body at `rag_admin_scope_service.py:565` calls `await self._maybe_await(method(...))` directly, inline). This third caller is real and live. However it does not reintroduce a UI-freeze risk in practice: `library_rechunk_run.py:_run` is dispatched via `self.app.run_worker(self._run, thread=True, ...)` and calls `asyncio.run(launch(...))` **inside that background thread**, so the inline sqlite/Chroma/chunking work runs on a private per-thread event loop, never the Textual app's own loop. Net effect: the review's "only list_templates and get_diagnostics have callers" statement is factually wrong (a third exists), but the disposition (structural defect, not a live event-loop freeze today) is still correct for a different reason than the review gave.

## 12. P3 — A vendored engine file carries a local hand-edit with no carry-forward record
- Verdict: CONFIRMED
- Site now: `Chunking/engine/security_logger.py:14` (`from tldw_chatbook.Utils.timestamps import utc_now_iso`), `:76` (used).
- Proof: `git log --oneline d60ebe1d0a..HEAD -- tldw_chatbook/Chunking/engine/` → exactly one commit, `c255ec3936 feat(time): timestamp-writer format guard, in preflight + CI (task-32803.1 AC#2/#3)`. `Tests/Architecture/test_vendor_pin_consistency.py` docstring confirms it pins only the commit SHA (`PIN`/`upstream.commit`), never per-file content.
- Note: none.

## 13. P3 — `assign_select_value` reads Textual's private `Select._options`
- Verdict: CONFIRMED
- Site now: `Widgets/select_values.py:44` (`if _offers(select._options, candidate):`).
- Proof: Textual's own `Select._validate_value` (installed venv, `textual/widgets/_select.py:589`) checks `value not in self._legal_values` — a *different* private attribute than `_options`, though `_legal_values` is itself derived from `_options` (`_select.py:534-537`) so the check is functionally equivalent today. The module's own comment at `select_values.py:23-24` names `_legal_values` as what it mirrors, while the code reads `_options` — confirming the finding's "not to the `_legal_values` it names."
- Note: none.

TOTALS: confirmed=13 fixed=0 wrong=0 demoted=0 promoted=0

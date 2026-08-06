# RAG Truth PR-T2 — Paid-Moment Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The product stops taking the user's money silently. RAG Answer names its provider and model before the call and its real cost after; the context/cost surfaces count the staged evidence they will actually send instead of reporting zero; the paid mode is marked as paid; and the three-way provider-config split that let the critique spend money while being told it had no provider is resolved to one truth.

**Architecture:** No new cost infrastructure — the expensive half already exists and is provider-agnostic (`Chat/provider_usage.py`, `LLM_Calls/pricing_catalog.py`, both pure, zero Console imports). Task 1 promotes the money/token *formatting* out of `console_cost_tracker`'s privates into a shared module so both surfaces speak identically. Tasks 2-3 stop `library_rag_answer_service.py:464` from discarding `raw`'s provider/model/usage and render it in the answer region's own footer. Task 4 marks the paid mode in the quiet line that already reserves its row. Tasks 5-6 fix the "sums only the rows it owns" bug class: swap the char-ratio placeholder for the real tokenizer that already exists unused, then feed staged evidence into both the estimate and the chip (as an *estimated* row, so the chip's existing `~` marker does the honesty). Task 7 normalizes the provider-config namespaces once in `config.py` and makes the Library gate ask the same question Console asks.

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest.

**Branch/worktree:** `feat/rag-truth-paid-moments` in `.worktrees/rag-truth-pr2`, base `afebcad5f` (the PR-T1 merge). All paths relative to the worktree root. Full seam evidence: `scout-map.md` in the SDD workspace — every line ref verified at `afebcad5f`; **re-verify before editing, never edit blind** (three plan premises were refuted against live code in PR-T1; expect the same rigor here).

## Global Constraints

- **Never claim a number you cannot source.** The house precedent is `UI/Evals/inspector.py:428-446`: it refuses to say "local · no cost" for an unresolvable target because that "is a claim about money this code has no basis for," and prints `cost unknown — …` instead. Match that register: unknown pricing says so; it never renders `$0.00` or omits the line.
- **Never store dollars** — store usage, price at render time from `pricing_catalog` (existing discipline; `ModelPricing.as_of` carries staleness).
- **Quiet register, verb-first, second person.** Exemplars: `_ORIGIN_SENTENCES`/`_UNKNOWN_ORIGIN_SENTENCE`/`_EMPTY_STATE_COPY` (`UI/MCP_Modules/mcp_inspector.py:131-163`), `LIBRARY_RAG_*_COPY` (`Library/library_rag_state.py:71,127`). No marketing, no hedging.
- **Escaping is terminal; `markup=False`** on every new Static (the cost chip's own test pins this: `Tests/Chat/test_console_status_chips_cost.py:318`).
- **Targeted tests only** (owner ruling): each task's gate = the test files it touched + `pytest Tests/ --collect-only -q | tail -3`. Never a full suite. **The machine is heavily contended — ONE test file per foreground Bash command with a generous timeout; never end a turn waiting on a background command.**
- venv-only pytest (`source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate`); pytest under `Tests/` is the ONLY python entry point (a bare `python3 -c` importing app modules writes the LIVE config). `git stash` FORBIDDEN.
- **Protected oracles:** pre-existing tests change ONLY where a task names the change as deliberate, and every such change is called out in that task's report. Tasks 4, 5 and 6 each have authorized, named re-baselines — nothing else.
- **CSS:** `#library-rag-answer*` rules exist in BOTH `css/tldw_cli_modular.tcss` (the bundle) and `css/components/_agentic_terminal.tcss` (the source). Never hand-edit the bundle: edit the source and run `python3 -m tldw_chatbook.css.build_css`. `Tests/UI/test_css_class_coverage_contract.py` will fail the branch on any composed-but-unstyled class.
- **Backlog IDs:** next free is **2503** (138-worktree scan at plan time); re-scan at ship.

## File Structure

| File | Responsibility in this PR |
|---|---|
| `tldw_chatbook/Chat/cost_display.py` (new) | T1: shared money/token formatting + the one-line provenance builder |
| `tldw_chatbook/Chat/console_cost_tracker.py` | T1: delegate its privates to the shared module; T6: staged row flows through estimated-entry path |
| `tldw_chatbook/Library/library_rag_answer_service.py` | T2: `LibraryRagAnswer` carries provider/model/usage; capture from `raw` before discard |
| `tldw_chatbook/Library/library_rag_state.py` | T3: panel state carries the answer provenance; T4: paid-mode quiet-line copy |
| `tldw_chatbook/Widgets/Library/library_search_rag_panel.py` | T3: answer-region footer + in-flight provider line; T4: quiet line + tooltip |
| `tldw_chatbook/UI/Screens/library_screen.py` | T3: pass resolved provider/model into panel state |
| `tldw_chatbook/Chat/console_session_settings.py` | T5: real tokenizer; T6: staged text folded into `used_tokens` |
| `tldw_chatbook/UI/Screens/chat_screen.py` | T6: supply staged text to estimate + synthetic cost row |
| `tldw_chatbook/config.py` | T7: one normalization of provider credentials |
| `tldw_chatbook/Chat/provider_readiness.py` / `library_rag_answer_service.py` | T7: the Library gate asks the same question Console asks |
| `backlog/tasks/`, `Docs/User_Guide/` | T8 |

---

### Task 1: One shared vocabulary for money and tokens

**Files:**
- Create: `tldw_chatbook/Chat/cost_display.py`
- Modify: `tldw_chatbook/Chat/console_cost_tracker.py:212-249` (`_format_amount`, `_format_tokens` → delegate)
- Test: `Tests/Chat/test_cost_display.py` (new); `Tests/Chat/test_console_cost_tracker.py` must stay green UNMODIFIED

**Interfaces:**
- Produces: `format_cost_amount(value: Decimal | float | None) -> str` and `format_token_count(n: int | None) -> str` (byte-identical output to the current privates — that is the whole point), plus `build_provenance_line(*, provider: str, model: str, usage: ProviderUsage | None, cost: Decimal | None, pricing_known: bool) -> str` returning the one-line form used by both surfaces: `provider · model · $0.0031 (1,240 tok)` when priced, `provider · model · 1,240 tok · pricing unknown` when not, and `provider · model` when there is no usage yet.
- Consumes: `ProviderUsage` (`Chat/provider_usage.py`) only. **No Console imports** — this module must be importable from Library without dragging Console in (verify with a test that imports it in isolation).

- [ ] **Step 1: Read** `console_cost_tracker.py:212-249` and its callers, plus `Tests/Chat/test_console_cost_tracker.py` to learn the exact formatting contract (thousands separators, decimal places, None handling).
- [ ] **Step 2: Write failing tests** in `Tests/Chat/test_cost_display.py`: each formatter reproduces the tracker's output for a table of values INCLUDING the edge cases the tracker's own tests pin (zero, None, sub-cent, large); `build_provenance_line` for the three shapes above; and an import-isolation test asserting `Chat.cost_display` has no `Console` import (inspect the module's imports).
- [ ] **Step 3: Run to verify failure.** `pytest Tests/Chat/test_cost_display.py -q`
- [ ] **Step 4: Implement** the module; make the tracker's privates thin delegates (keep their names so the tracker's internal call sites and tests are untouched).
- [ ] **Step 5: Gate.** `pytest Tests/Chat/test_cost_display.py -q` then `pytest Tests/Chat/test_console_cost_tracker.py -q` (must be green with ZERO edits) then the collect sweep.
- [ ] **Step 6: Commit** — `feat(chat): shared cost/token display vocabulary`

---

### Task 2: The answer service keeps what the provider told it

**Files:**
- Modify: `tldw_chatbook/Library/library_rag_answer_service.py:196-219` (`LibraryRagAnswer`), `:464-488` (the capture seam), `:469-474` (empty-response failure)
- Test: `Tests/Library/test_library_rag_answer_service.py` (additive; the ready/abstain/fail assertions at `:522-634` may need field additions — name any as deliberate)

**Interfaces:**
- Produces: `LibraryRagAnswer` gains `provider: str = ""`, `model: str = ""`, `usage: ProviderUsage | None = None`. Populated on ready, abstained, AND the empty-response failure (a call that cost money and returned nothing still cost money — the usage must survive).
- Consumes: `raw` at `:464` — `raw.get("model")` (the provider's actual model, which is the ONLY app-side source since `resolve_library_rag_answer_provider` deliberately returns `model=None` and the handler picks its own default at `LLM_API_Calls.py:1227`), and `ProviderUsage.from_provider_payload(raw.get("usage"), provider=..., model=...)`.

**Context:** `chat_api_call` returns the handler result unmodified (`Chat/Chat_Functions.py:912,932`); the Anthropic normalizer sets `"model"` (`LLM_API_Calls.py:1791`) and `"usage"` (`:1799`); the OpenAI shape does the equivalent at `:297`. `extract_response_content` reads only `choices[0].message.content` — so today everything else is dropped.

- [ ] **Step 1: Write failing tests**: a stub chat returning a realistic Anthropic-shaped payload (model + usage) yields a `LibraryRagAnswer` carrying provider, the payload's model (NOT the configured endpoint name), and a `ProviderUsage` with the real token counts; same for the abstained path; same for an empty-response failure (usage preserved). A payload with NO usage key yields `usage=None` and does not raise. **Do not invent payload shapes** — copy the real ones from `Tests/Chat/test_provider_usage.py`.
- [ ] **Step 2: Verify failure.** `pytest Tests/Library/test_library_rag_answer_service.py -q -k "provenance or usage or model"`
- [ ] **Step 3: Implement** the dataclass fields and the capture at `:464` (before `extract_response_content` discards `raw`), threading through all three return sites. Containment: a malformed/absent usage payload must degrade to `usage=None`, never raise — the answer still renders.
- [ ] **Step 4: Gate.** `pytest Tests/Library/test_library_rag_answer_service.py -q` then the collect sweep.
- [ ] **Step 5: Commit** — `feat(library): RAG Answer keeps its provider, model and token usage`

---

### Task 3: The answer region says what it cost

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_search_rag_panel.py:231-398` (`library_rag_answer_children`: in-flight branch `:296-307`, region footer at `:398`)
- Modify: `tldw_chatbook/Library/library_rag_state.py:1545-1635` (`LibraryRagPanelState.from_values` carries the provenance)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:3857-3862` (pass resolved provider/model into panel state)
- Modify: CSS source `css/components/_agentic_terminal.tcss` (`#library-rag-answer*` block ~`:5077-5095`), then regenerate the bundle
- Test: `Tests/Library/test_library_rag_state.py`, `Tests/UI/test_product_maturity_gate16_library_search_rag.py` (additive)

**Interfaces:**
- Consumes: Task 1's `build_provenance_line`; Task 2's `LibraryRagAnswer.provider/model/usage`; `pricing_catalog.get_pricing_catalog().cost_for_usage(...)` priced at render time (never stored).
- Produces: (a) **before the call** — the in-flight line names the provider that is being billed (`Asking <provider>…` replacing the bare `Generating answer…`); (b) **after the call** — a footer Static `#library-rag-answer-provenance` inside the answer region's own `Vertical`, rendering Task 1's provenance line, `markup=False`; (c) unknown pricing renders the `pricing unknown` form, never `$0.00`.

- [ ] **Step 1: Write failing tests** at the state level (panel state carries provider/model/usage through to the region's children) and one gate16-style pilot asserting the footer text after a stubbed answer; plus a test that an answer with `usage=None` renders the no-usage form and does NOT render a dollar figure.
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement.** Keep the region's existing four status branches intact; the footer is additive and appears only when there is something true to say. Style the new id in the CSS SOURCE, regenerate the bundle, and confirm `Tests/UI/test_css_class_coverage_contract.py` stays green.
- [ ] **Step 4: Gate.** Each touched test file in its own foreground command, then `pytest Tests/UI/test_css_class_coverage_contract.py -q`, then the collect sweep.
- [ ] **Step 5: Commit** — `feat(library): RAG Answer shows provider, model and cost`

---

### Task 4: Mark the paid mode as paid

**Files:**
- Modify: `tldw_chatbook/Library/library_rag_state.py` (quiet-line copy; `mode_label` region ~`:1015`)
- Modify: `tldw_chatbook/Widgets/Library/library_search_rag_panel.py:417-464` (the always-present quiet line), `:481-492` (tooltip)
- Test: **AUTHORIZED RE-BASELINE** — `Tests/UI/test_library_shell.py:1479-1533` pins both tooltip strings verbatim; `:1568`, `:1611`, `:2500-2540`; `Tests/UI/test_product_maturity_phase1_empty_setup_states.py:382-396`; `Tests/UI/test_product_maturity_gate16_library_search_rag.py:1119-1120`. Update ONLY the strings this task deliberately changes; every other assertion in those tests stays byte-identical, and each change is named in the report.

**Interfaces:**
- Produces: in RAG Answer mode with a provider configured, the quiet line states plainly that running will call a paid provider and names it — one sentence, house voice, e.g. `RAG Answer sends your question and the evidence to <provider>. Search stays local.` In Search mode the quiet line keeps its current ready-state behavior (empty, reserving its row — the layout rationale at `:421-425` is load-bearing; do not regress it). The tooltip gains the same fact in its own register.
- **Do not** add a confirmation dialog or a blocking gate — the owner's product is keyboard-fast; this is a statement, not a speed bump.

**Context:** today the ONLY provider-adjacent copy on this surface is the *blocked* branch (`library_rag_state.py:1046-1051`) — configure a provider and every signal disappears, the exact inversion of what's needed.

- [ ] **Step 1: Write failing tests**: quiet line carries the paid sentence in Answer mode with a ready provider; stays empty (row reserved, height 1) in Search mode; the blocked branch's existing copy is unchanged; tooltip contains the fact.
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement**, then update the pinned strings in the named test files — strings only.
- [ ] **Step 4: Gate.** Each named test file in its own foreground command, then the collect sweep.
- [ ] **Step 5: Commit** — `feat(library): the paid mode says it is paid`

---

### Task 5: Count tokens with the real counter

**Files:**
- Modify: `tldw_chatbook/Chat/console_session_settings.py:1077-1091` (`_estimate_tokens_locally` internals; keep the signature — it is an injectable `TokenCounter`, `:70`)
- Test: **AUTHORIZED RE-BASELINE** — `Tests/Chat/test_console_session_settings.py:745-790` asserts an exact label (`"123 / 456 tokens"`) via an injected counter and should be UNAFFECTED (it injects); anything asserting numbers from the *default* counter will move. Re-baseline only those, name each in the report.

**Interfaces:**
- Consumes: `Utils/token_counter.py:137-161` `estimate_tokens(text, model, provider)` — custom tokenizer → tiktoken → conservative chars floor, explicitly never a whitespace word count.
- Produces: `_estimate_tokens_locally` delegates to it, preserving its current signature and its None/empty handling. The fake `len(messages) * 10` overhead is retired (or justified in a comment if the real counter needs a per-message allowance — say which you chose and why).

**Context:** today the estimator is a char-ratio placeholder (`CONSOLE_TOKEN_CHAR_RATIOS` at `:99-103`, `del model` at the top) while a real tokenizer sits unused in the same repo. Swapping it changes every number both the chip and the settings modal display — that is the point, but it means the re-baseline must be deliberate and named.

- [ ] **Step 1: Write failing tests** comparing the estimator against `estimate_tokens` for a table of realistic inputs (short, long, code, unicode) and asserting the model/provider arguments are actually honored (the placeholder ignored `model` entirely).
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement**; run the two affected suites and re-baseline ONLY assertions whose numbers moved for this reason. If a number moves in a way you cannot explain from the tokenizer swap, STOP and report — that is a real find, not a re-baseline.
- [ ] **Step 4: Gate.** `pytest Tests/Chat/test_console_session_settings.py -q`, then `pytest Tests/Chat/test_console_cost_tracker.py -q` (it imports the estimator), then the collect sweep.
- [ ] **Step 5: Commit** — `fix(console): estimate tokens with the real counter, not a character ratio`

---

### Task 6: Count the evidence you are about to send

**Files:**
- Modify: `tldw_chatbook/Chat/console_session_settings.py:721-780` (`build_console_context_estimate` gains staged text/tokens and folds it into `used_tokens`)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:4052-4088` (estimate call site), `:9440-9448` (cost snapshot call site)
- Test: **AUTHORIZED RE-BASELINE** — `Tests/Chat/test_console_session_settings.py:691-708` (`test_context_estimate_counts_messages_and_staged_sources`) currently pins the label-only behavior this task must change. Update it; name it.

**Interfaces:**
- Consumes: the staged text the send will ACTUALLY carry. Two candidates, in order of honesty: `capture_console_staged_evidence_for_chat`'s `result.context` (`Event_Handlers/Chat_Events/chat_rag_events.py:1601-1641`, consumed at `chat_screen.py:5305-5317`) — the exact prompt block; or `evidence_bundle_from_launch(launch).references[].snippet` (`Chat/console_display_state.py:253-267`). **Read both and choose the one that is true at estimate time without making a provider call or a DB round trip; say which and why.** Note `EvidenceReference` already records `snippet_truncated` and `original_snippet_char_count` (`Chat/citation_evidence_models.py:65-66`) — a 942 KB source's real size is already captured.
- Produces: (a) the context estimate's `used_tokens` includes staged evidence, with the existing `"; N sources staged"` suffix retained; (b) the cost chip receives staged evidence as a synthetic ESTIMATED row (`role="user"`, `content=staged_text`, `usage=None`) via its duck-typed row contract (`console_cost_tracker.py:445-447`), so it prices at the input rate and flips `has_estimated_entries` → the existing `~` prefix. The `~` is the honesty marker for an unsent estimate; do not invent a new one.

**Context:** `ConsoleStagedSource` (`Chat/console_chat_models.py:195-201`) carries no text — the snippet is discarded when it is built (`chat_screen.py:4645-4662`). Whatever source you choose, the estimate must not silently report zero for content it will send.

- [ ] **Step 1: Write failing tests**: an estimate with staged evidence counts more than the same estimate without it, and the delta tracks the staged text's real size; the label keeps its sources suffix; the cost chip shows the `~` estimated marker when staged-but-unsent evidence is present; a 942 KB-class source does not produce a zero.
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement**, keeping the estimate pure (no I/O in the builder — pass the text in).
- [ ] **Step 4: Gate.** `pytest Tests/Chat/test_console_session_settings.py -q`, `pytest Tests/Chat/test_console_cost_tracker.py -q`, `pytest Tests/UI/test_console_staged_context.py -q`, then the collect sweep.
- [ ] **Step 5: Commit** — `fix(console): staged evidence counts toward the context and cost estimates`

---

### Task 7: One truth for "is a provider configured?"

**Files:**
- Modify: `tldw_chatbook/config.py:881` region (normalize `[API] <provider>_api_key` into `api_settings.<provider>.api_key` when the modern key is absent) and `:1044-1046` (derive the legacy `<provider>_api` dicts from the SAME normalized value)
- Modify: `tldw_chatbook/Library/library_rag_answer_service.py:179-193` (`library_rag_answer_provider_ready` asks `resolve_provider_readiness`)
- Test: `Tests/Chat/test_provider_readiness.py` (additive), `Tests/Library/test_library_rag_answer_service.py:699-751` (provider-resolution pins — additive; do not change what they assert about endpoint resolution)

**Interfaces:**
- Produces: a key in `[API] anthropic_api_key` makes BOTH readers agree — `provider_readiness` sees it (Console stops showing "Connect a provider") and `LLM_API_Calls.py:1218-1219` still spends with it. `library_rag_answer_provider_ready()` returns True only when credentials actually resolve, not merely when an endpoint NAME is non-empty.
- Consumes: `Chat/provider_readiness.py:140-204` (`_provider_settings_for_key`, `provider_config_key`, `_valid_api_key`).

**Context (the harm this closes):** the critique spent real money through the Library path while Console showed a blocking "Connect a provider" wall, because `[API]` lands only in `anthropic_api` (`config.py:1044`) — the namespace that spends — while Console gates on `api_settings.<provider>.api_key` (`provider_readiness.py:192`). A THIRD namespace makes it worse: the Library's own gate reads `config.default_api_endpoint` and never checks credentials at all. No test pins the legacy mapping or the absence of a bridge — that gap is why this shipped.

- [ ] **Step 1: Write failing tests**: with ONLY `[API] anthropic_api_key` set, `resolve_provider_readiness("anthropic", cfg)` reports configured; the legacy `anthropic_api.api_key` still resolves (spending path unbroken); `library_rag_answer_provider_ready()` is False when an endpoint is named but no credential resolves, True when both do; a modern `api_settings` key still wins where both exist (precedence stated explicitly in the test name).
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement** the single normalization; do not add a second code path — both readers must consume the same normalized value.
- [ ] **Step 4: Gate.** `pytest Tests/Chat/test_provider_readiness.py -q`, `pytest Tests/Library/test_library_rag_answer_service.py -q`, then the collect sweep. **Do not** run a live app probe (it writes the real config).
- [ ] **Step 5: Commit** — `fix(config): one truth for provider credentials across Console and Library`

---

### Task 8: Docs and backlog

**Files:** `Docs/User_Guide/library/search-and-rag.md` (it already promises cost copy this surface never delivered — `:122`, `:141`, `:284`), `Docs/User_Guide/console/context-and-rag.md`, `backlog/tasks/`

- [ ] **Step 1:** Document what shipped: RAG Answer names its provider/model and its cost; the paid mode is marked; the context/cost estimates count staged evidence; a `[API]` key now satisfies both Console and Library. Correct any doc text this PR falsified. Stamp per each file's existing convention.
- [ ] **Step 2:** Fresh cross-worktree ID scan (next free was 2503 at plan time). File Done records for what shipped and To Do tasks for anything deliberately left (e.g. streaming-path usage capture for Library if it ever streams; any re-baseline whose cause you could not fully explain).
- [ ] **Step 3: Commit** — two commits, docs and backlog separately.

---

### Task 9: Whole-branch review, live verification, ship

- [ ] **Step 1: Whole-branch review** (strongest available model — note Fable hit a usage limit during PR-T1; prefer Opus). Composition watch-items: T5's tokenizer swap changes T6's inputs AND the chip's totals — are the re-baselines consistent with each other? T2/T3's provenance vs T7's readiness (can the footer name a provider the gate says isn't configured?). T1's shared formatter used by two surfaces — identical output proven both places? Point it at the ledger's deferred minors for triage.
- [ ] **Step 2: ONE fix wave + one scoped re-review.** Residuals adjudicated (parked with rulings, or escalated if load-bearing).
- [ ] **Step 3: Live verification** (scratch profile; the PR-T1 recipe is proven — copy the three DBs + chromadb BEFORE first launch, `[first_run] setup_started/completed = true`, session-suffixed tmux socket, python char-index for click columns). **This one needs a real paid call** — the repo-root `anthropic-api-key.txt` is deliberately provided for agent use; budget ONE RAG Answer call. Scenarios: (1) Answer mode shows the paid sentence before running; (2) the in-flight line names the provider; (3) after the answer, the footer shows provider · model · real cost from real usage; (4) with pricing unknown, it says so and shows no dollar figure; (5) staged evidence moves the context estimate off zero and the chip shows `~`; (6) a `[API]`-only key satisfies Console (no "Connect a provider" wall) AND spends correctly.
- [ ] **Step 4: Docs stamp** with the live-check commit.
- [ ] **Step 5: Ship** — merge latest origin/dev (regenerate the CSS bundle on conflict, never hand-merge), targeted gates, fresh ID re-scan, push, PR, merge on verified, confirm `.merged`.

---

## Self-Review Notes

- Coverage: F1 → T2+T3; F2 → T1 (reuse, not rebuild); F3 → T5+T6; F4 → T4; split-brain (a) → T7; docs/backlog → T8; ship → T9.
- Type consistency: `build_provenance_line`'s signature (T1) is what T3 calls; `LibraryRagAnswer`'s three new fields (T2) are what T3 renders; `_estimate_tokens_locally`'s signature is preserved across T5 so T6's call sites don't move.
- Three authorized re-baselines (T4 strings, T5 default-counter numbers, T6 label-only staged behavior) — each names its files and forbids collateral edits.
- Known unknowns delegated with read-first instructions, not placeholders: which staged-text source is true at estimate time (T6 Step 1), whether the per-message overhead survives the tokenizer swap (T5), the exact formatting contract of the tracker's privates (T1 Step 1).

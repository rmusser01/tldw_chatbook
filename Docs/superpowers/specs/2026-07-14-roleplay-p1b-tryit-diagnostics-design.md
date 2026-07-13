# Roleplay P1b — Try-it diagnostics (design)

**Status:** Implemented (P1b).
**Program:** Personas→Roleplay redesign. P0 MERGED (#619); **P1a MERGED (#622)**. P1 = Dictionaries mode, decomposed P1a→P1d. This is **P1b**: the Try-it pane's upgrade from before/after diff to the north-star verify surface — which entries fired, which near-missed and why, and the token-budget picture.
**Research input:** `Docs/superpowers/research/2026-07-13-server-dictionaries-port.md` (server surfaces `entries_used`/`replacements`/`token_budget_exceeded`; it has no near-miss reporting — that part is ours).
**Worktree/branch:** `.claude/worktrees/personas-redesign`, `claude/roleplay-p1b-diagnostics` off dev `9aa86bf9` (includes P1a).

## Problem

P1a's Try-it shows *what changed* (word-diff via `process_text`) but not *why*: which entries fired, how many times, which matched-but-were-skipped (and at which pipeline stage), and how the token budget played out. The engine (`Chat_Dictionary_Lib.process_user_input`) returns a bare `str`, discarding everything the pipeline knows. P1b adds an **additive** diagnostics-returning path and renders it in Try-it. The chat-time `str` contract stays byte-compatible.

## Engine reality (ground-truth read — drives the design)

`process_user_input(user_input, entries, max_tokens=5000, strategy="sorted_evenly") -> str` (`Chat_Dictionary_Lib.py:503-660`) runs seven sequential stages. Facts the design depends on, verified in code:

- **Every stage is identity-preserving**: each takes a `List[ChatDictionary]` and returns a filtered/sorted subset of the *same objects* (`match_whole_words`, `group_scoring`, `filter_by_probability`, the `apply_timed_effects` loop, `enforce_token_budget`, `apply_strategy`). → Per-entry skip reasons are derivable from **stage-boundary set-diffs by `id(entry)`**, with an `{id(e): input_index}` map built once. **No helper function changes.**
- **Skip stages** (an entry that matched at stage 1 can drop out at): `group_scoring` (best-of-named-group = longest raw_key wins), `probability` (per-run `random.randint(1,100) <= probability`), `timed_effects` (delay/cooldown vs `last_triggered`), `token_budget` (walk-and-stop; the first non-fitting entry **breaks the loop**, dropping everything after it too).
- **`apply_strategy` is sort-only** (unknown strategy returns input unchanged) — it is NOT a skip stage.
- **The budget stage runs BEFORE the strategy sort** (`:603` vs `:623`): survival under budget pressure depends on *stored/match* order, not display order. The diagnostics report what actually happened; the quirk itself is documented here and its fix (priority-aware budgeting) belongs to **P1c**, not P1b.
- **`alert_token_budget_exceeded` is dead code in this pipeline**: it runs on the post-trim list, whose total is ≤ budget by construction, and `warnings.warn` doesn't raise — so the pipeline's `except TokenBudgetExceededWarning` is vestigial. → `budget_exceeded` must be **derived from truncation** (≥1 entry dropped at the budget stage), never from the alert. The refactor keeps the vestigial catch **verbatim** (behavior-preserving; no opportunistic cleanup in this PR).
- **`apply_timed_effects` does not mutate on check** (explicit comment `:380-384`); `last_triggered` is set only in the replacement loop, on in-memory objects. `process_text` loads entries fresh per call and `last_triggered` is never persisted → **Try-it runs are side-effect-free** for chat behavior.
- **`skipped:timed_effects` is practically unreachable today**: BOTH the chat path and `process_text` load entries fresh per call, and `last_triggered` is never persisted — so delay/cooldown checks always pass on a `None` last-trigger. The diagnostics keep the stage for completeness (the engine supports it; lib tests exercise it by pre-setting `last_triggered`), but the UI reason copy will essentially never render in practice. Do not "fix" the persistence as part of P1b.
- **Zero-replacement survivors are real**: every stage-survivor matched the *original* text, but replacements apply sequentially — an earlier entry's replacement can consume a later entry's key (`apply_replacement_once` then finds nothing). Cause is nameable: *"text changed by an earlier entry."*
- **Per-entry token cost** = `len(entry.content.split())` — reuse `calculate_token_usage([entry])`, don't re-implement.
- **There is no local "iterations" concept** (the server's field): replacement is a per-entry `max_replacements` loop. The honest local total is `total_replacements`. Do not invent an `iterations` field.
- **Chat-time call sites** (must be behavior-identical after the refactor): `Chat/Chat_Functions.py:1024` (outgoing message) and `:1281` (post-generation pass).

## Goal / Acceptance

- **AC1 — additive engine path.** New `process_user_input_with_diagnostics(...) -> tuple[str, DictionaryProcessDiagnostics]` containing the moved pipeline body; `process_user_input` becomes a wrapper that calls it and returns only the string — **same signature, same return type, same behavior** (pinned by a wrapper-contract test).
- **AC2 — honest diagnostics.** Per matched entry: `input_index`, `pattern` (raw_key), `status` (`fired` | `skipped:group_scoring` | `skipped:probability` | `skipped:timed_effects` | `skipped:token_budget` | `skipped:strategy_error` — defensive, only reachable if `apply_strategy` raises; the UI falls back to rendering the raw status string | `no_replacement`), `replacements`, `token_cost`, `content_preview` (first 40 chars of the entry content — the fired line renders it), and `applied_order` (`int` for entries that reached the replacement loop — their 0-based position in the post-strategy application sequence — else `None`). **The UI's fired list renders in `applied_order`**, so "no replacement — text changed by an earlier entry" points at entries visibly ABOVE it; near-misses follow in `input_index` order. Totals: `matched`, `fired`, `skipped`, `total_replacements`, `tokens_used` (**budget-stage accounting**: sum of `token_cost` over entries that survived the budget stage — including `no_replacement` survivors, which consumed budget without firing), `token_budget`, `budget_exceeded` (truncation-derived). Never-matched entries are omitted.
- **AC3 — service pass-through.** Local `process_text` uses the diagnostics path and adds ONE additive response key `"diagnostics": {...}` (existing keys byte-identical), with per-entry records enriched to carry the positional entry id (`local:chat_dictionary_entry:<dict_id>:<index>`). **Enrichment is append-time id tracking, not index arithmetic:** `process_text` builds its engine input list by iterating dictionaries and applying an optional `group` filter — so `input_index` equals the stored index ONLY for Try-it's exact call shape (single id, no group). The service builds a parallel `entry_ids: list[str]` as it appends each entry (recording that entry's dictionary id + stored index at append time); enrichment is then `entry_ids[input_index]`, which stays correct under group filters and under the `dictionary_id=None` all-dictionaries path (where records simply carry each entry's own dictionary's id).
- **AC4 — Try-it renders the story.** Below the existing diff: a **summary strip** (`{fired} fired · {skipped} skipped · {tokens_used}/{token_budget} tokens`, amber budget flag when `budget_exceeded`), compact **fired** lines (`pattern → replacement · ×{replacements} · {token_cost} tok`), and dim **near-miss** lines with human reasons (`skipped: lost group scoring` / `probability roll` / `cooldown or delay` / `token budget` / `no replacement — text changed by an earlier entry`). Probability copy notes that re-running may differ. Results area scrolls (`overflow-y: auto` + max-height — the P1a clipping lesson).
- **AC5 — graceful degrade.** A `process_text` response without the `diagnostics` key renders exactly P1a's diff-only view plus a dim "diagnostics unavailable" note (future server mode; defensive).
- **AC6 — no collision / no contract drift.** Only Personas-owned files + the two Character_Chat dictionary files change. Shared-shell files untouched. The chat-time `str` contract and every existing `process_text` response key unchanged.

## Architecture

**Files**

- **Modify** `Character_Chat/Chat_Dictionary_Lib.py` — add `@dataclass DictionaryProcessDiagnostics` (entry records + totals + `to_dict()`); add `process_user_input_with_diagnostics()` containing the moved body of `process_user_input`, instrumented purely by stage-boundary diffs (`id()`-keyed; duplicate objects in the input are first-wins — pathological, the service never produces them) and the existing replacement counters; shrink `process_user_input` to the wrapper. Google-style docstrings.
- **Modify** `Character_Chat/local_chat_dictionary_service.py` — `process_text` calls the diagnostics path, maps `input_index` → entry id, and adds the `"diagnostics"` key via the dataclass's `to_dict()`.
- **Modify** `Widgets/Persona_Widgets/personas_dictionary_tryit.py` — `render_result(original, processed, diagnostics: dict | None = None)` (additive param, default preserves P1a callers); renders summary strip + fired + near-miss Statics; new DOM ids `#personas-dict-tryit-summary`, `#personas-dict-tryit-fired`, `#personas-dict-tryit-nearmiss`; structure-only CSS with scroll containment.
- **Modify** `UI/Screens/personas_screen.py` — `_handle_dictionary_tryit_run` passes `response.get("diagnostics")` through to `render_result`.
- **Tests** — lib: `Tests/Character_Chat/test_chat_dictionary_lib_diagnostics.py` (new; direct engine tests); service: extend `test_local_chat_dictionary_service.py`; UI: extend `Tests/UI/test_personas_dictionaries.py` (fake service emits the real diagnostics shape).

## Data flow

Try-it Run → screen `process_text({"text", "dictionary_id", "token_budget": max_tokens})` → local service loads the dict (stored entry order) → `process_user_input_with_diagnostics(text, entries, max_tokens=token_budget, strategy=dict.strategy)` → `(processed_text, diagnostics)` → service enriches entry ids → response `{text, processed_text, dictionary_id, source, diagnostics}` → screen calls `tryit.render_result(text, processed, diagnostics)` → diff + summary + fired + near-misses.

Chat path: `Chat_Functions` keeps calling `process_user_input(...)` — the wrapper — and receives the identical string.

## Error handling

- The pipeline's existing per-stage `except → entries = []` fallbacks are preserved verbatim; when one fires, the stage-diff honestly reports every then-surviving matched entry as `skipped:<that stage>`.
- Diagnostics assembly itself must never break processing: the core collects into plain structures with no additional I/O; if `to_dict()`/enrichment fails at the service layer, log + omit the `diagnostics` key (AC5 degrade), never fail the substitution response.
- UI: malformed/partial diagnostics dict → render what parses, skip what doesn't (each section guards independently); the diff never depends on diagnostics.

## Testing

- **Lib (deterministic):** one fixture dictionary per skip stage — group loser (two same-group entries, shorter key loses), probability 0 (skipped) and 100 (fired) — never rely on intermediate probabilities; cooldown (pre-set `last_triggered`); budget truncation (tiny `max_tokens`, assert both the too-big entry AND post-break entries report `skipped:token_budget`, `budget_exceeded is True`); sequential-consumption `no_replacement` (entry A's replacement eats entry B's key); totals arithmetic; never-matched entries absent.
- **Wrapper contract:** `process_user_input(args) == process_user_input_with_diagnostics(args)[0]` for a mixed fixture (pins the chat path; probability fixed at 100/0 for determinism).
- **Service (real DB):** `process_text` response carries `diagnostics` with enriched ids; existing keys byte-identical; a diagnostics-assembly failure (monkeypatched `to_dict` raising) omits the key but still returns `processed_text`.
- **UI:** summary/fired/near-miss render from the fake's realistic diagnostics; budget flag shows when `budget_exceeded`; AC5 degrade (fake returns no `diagnostics` key → diff-only + unavailable note); fake shape mirrors the real `to_dict()` output (the P1a fake-divergence lesson — the fake must not invent a friendlier shape).
- Follow the P1a harness patterns (`test_personas_dictionaries.py` fixtures, `size=(200, 60)` for detail-area clicks).

## Scope / non-goals

- **P1b does NOT**: add per-entry enabled/case/priority or priority-aware budgeting (**P1c** — including the budget-before-strategy order quirk documented above), Attachments/Stats/Versions/import-export (**P1d**), a live per-turn chat indicator (digest Stage 2 — explicitly out), or any change to chat-time behavior.
- **Forward:** P1c's validation panel can reuse the diagnostics dataclass's entry-record shape for its `{code, field, message}` list; P2 (Lore Test-Match) inherits this exact stage-diff pattern for world-book triggers.

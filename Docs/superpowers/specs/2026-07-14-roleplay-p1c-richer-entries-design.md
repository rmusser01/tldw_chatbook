# Roleplay P1c — richer entries + validation (design)

**Status:** Implemented (P1c).
**Program:** Personas→Roleplay redesign. P0 #619, P1a #622, P1b #625 MERGED; Prompts→Library landed (4-mode strip). This is **P1c**: per-entry `enabled`/`case_sensitive`/`priority`, the structured validation panel, and the unified-ordering fix for the documented budget-before-strategy quirk. P1d (tabs + portability) follows.
**Worktree/branch:** `.claude/worktrees/personas-redesign`, `claude/roleplay-p1c-entries` off dev `a2de830c`.

## Problem

Dictionary entries can't be individually disabled, are always case-insensitive for literal keys (hardcoded `re.IGNORECASE`), and have no priority — so under token-budget pressure, survival follows hidden *stored* order (the quirk documented in the P1b spec's Engine reality). Authoring mistakes (a regex that silently degrades to a literal, duplicate patterns, probability-0 entries) surface nowhere. P1c adds the three per-entry fields end-to-end (model → engine → service → form → Try-it), unifies ordering under a priority knob, and gives the Entries tab a warn-not-block validation panel with jump-to-entry.

## Decisions locked in brainstorm

- **Priority governs BOTH budget survival and application order** (one knob, server-style): entries order by `-priority`, tie-broken by the dictionary's strategy sort. Legacy-safe: default `priority=0` + stable sorts → strategy order for everyone until a priority is actually set.
- **Validation is warn-not-block, lives in the Entries tab**, recomputed on load and after every mutation; selecting a finding jumps the entries-table cursor. The engine already tolerates every finding (bad regex degrades to literal), so blocking would misstate severity. (Spec review dropped the originally-floated inspector count-mirroring — see AC5.)
- **Disabled entries filter AFTER matching** → they appear in Try-it as `skipped:disabled` near-misses (visibility beats pre-filtering; the extra match cost for disabled entries is negligible).

## Ground truths (verified in code this session)

- `ChatDictionary.to_dict()`/`from_dict()` round-trip with `.get` defaults (`Chat_Dictionary_Lib.py:133-155`) → adding `enabled=True` / `case_sensitive=False` / `priority=0` as constructor kwargs + dict keys is **fully backward-compatible with existing `entries_json` rows**; no DB schema bump (entries_json is opaque JSON).
- **Update-merge round-trip is name-aligned:** `update_entry` merges `to_dict()` output with API payloads through `_entry_from_payload`. The three new fields use the SAME name in both vocabularies, so `_entry_from_payload` reading `data.get("enabled", True)` / `data.get("case_sensitive", False)` / `data.get("priority", 0)` serves both the stored-merge and API-payload paths — no dual-name fallback machinery.
- Literal-key case-insensitivity is hardcoded at exactly two sites: `match_whole_words` (`:455`) and `apply_replacement_once` (`:498`), both `re.IGNORECASE` on the escaped-key pattern. Regex keys derive flags from their own `/pat/i` form (`_compile_key_internal`) — `case_sensitive` is **ignored for regex entries** (validation warns).
- `ChatDictionary.matches()` (`:128-134`) is **dead code** (no callers; pipeline uses `match_whole_words`) — deliberately untouched by the case change.
- `group_scoring` picks a named group's winner by **longest raw key only** (`:335`) — without a fix, a `priority=100` entry loses its group to a longer-keyed sibling, contradicting the one-knob principle.
- The Duplicate flow's payload is a **hardcoded field list** (`personas_screen.py:1661-1672`) — the three new fields must be added there or duplication silently strips them.
- The regex-form wrap rule (`type=="regex"` → `/pat/`) and slash-flag parsing live in `_entry_from_payload` (service) and `_compile_key_internal` (engine). The validation module must **probe through these** (construct a `ChatDictionary` and inspect `is_regex`), never re-implement the parsing.
- P1b's diagnostics instrumentation (stage-boundary diffs, `skip_reason_by_id`, `applied_order`) directly extends to the new stages; the status enum grows `skipped:disabled`.
- The P1b behavior-preservation constraint bound *that* refactor. **P1c amends the pipeline and helpers deliberately** — the changes below are the spec'd behavior deltas; everything not listed stays byte-identical.

## Goal / Acceptance

- **AC1 — model.** `ChatDictionary` gains `enabled: bool = True`, `case_sensitive: bool = False`, `priority: int = 0` (constructor, `to_dict`, `from_dict` with defaults). Old stored entries parse unchanged. The service seam (`_entry_from_payload`/`_entry_to_response`) round-trips all three under the same API names; the response's hardcoded `"enabled": True` becomes the real value.
- **AC2 — engine.** Pipeline (new order): match → **disabled filter** (`skipped:disabled` near-miss) → group scoring (**winner = max(priority, then raw-key length)**) → probability → timed effects → **strategy sort, then stable sort by `-priority`** → token budget (walk-and-stop, unchanged mechanics, now walking the unified order) → replacements (no post-budget re-sort). `case_sensitive=True` literal entries match and replace case-sensitively (both sites); `False` keeps today's `IGNORECASE`. Diagnostics stage-diffs follow the new sequence.
- **AC3 — chat-path impact (explicit).** For legacy data (all `priority=0`, all enabled, `case_sensitive=False`): matching, filtering, group winners, and application order are **identical** (stable sorts + length tie-breaks reproduce today's outcomes). The ONLY observable legacy change: under budget pressure, survival follows the strategy order instead of stored order — the documented quirk, fixed deliberately. The P1b wrapper-contract + iterable-degradation tests stay green unchanged; budget-order tests update to pin the new ordering.
- **AC4 — validation.** New Personas-owned module `Widgets/Persona_Widgets/personas_dictionary_validation.py`: `validate_entries(entries: list[dict]) -> list[ValidationFinding]` (dataclass: `code`, `field`, `message`, `entry_id`). Codes: `invalid_regex` (regex-form pattern that degrades to literal — probed via `_entry_from_payload`→`ChatDictionary.is_regex`), `duplicate_pattern` (exact same `pattern`+`type` twice), `probability_zero` (can never fire), `case_flag_on_regex` (`case_sensitive` set on a regex entry — ignored at runtime). Pure function, no I/O.
- **AC5 — Entries tab.** Form gains Enabled Switch, Case-sensitive Switch, Priority Input (any integer; integer-ness validated like the existing numeric fields). Table gains a `pri` column and shows disabled rows dimmed with an `off` marker. A compact validation `OptionList` (`#personas-dict-validation`) under the table lists findings as `[code] pattern — message`; selecting one moves the table cursor to that entry. Recomputed on `load_dictionary`/`update_entries`. Validation is **wholly Entries-tab-local** — the inspector is untouched (no dictionary path calls `show_validation` today, and mirroring a count that sits directly under the table would add a message + handler for nothing).
- **AC5b — pay the clipping debt.** P1c adds height to the Entries tab, so the P1a-ledgered Important (tab needs ~19 rows, no scroll escape, buttons unreachable by mouse at 80×24) is paid NOW: the Entries `TabPane` content gets a scrollable container (`overflow-y: auto` + sensible max-height), verified by a geometry test that the button row and validation list are reachable at a constrained size.
- **AC6 — Try-it.** Reason map gains `skipped:disabled` → "skipped: disabled". Diagnostics enum in the P1b spec's AC2 sense grows the same status.
- **AC7 — Duplicate integrity.** The Duplicate payload carries all TEN entry fields (existing seven + the three new); the duplicate round-trip test asserts every field survives (promoting the P1a-deferred assertion gap to mandatory).
- **AC8 — deferred P1b minors.** The UI test fake's budget branch aligns to real walk-and-stop (`break` semantics) and mirrors the new ordering/fields; `content_preview` whitespace-flattens (`" ".join(content.split())[:40]`) so multi-line content can't break the fired-line layout.
- **AC9 — scope guard.** Only Personas-owned + Character_Chat dictionary files change. No shared-shell files, no DB schema version change, no server-service changes beyond the shared seam file if trivially required (report if so).

## Architecture / files

- **Modify** `Character_Chat/Chat_Dictionary_Lib.py` — model fields; `group_scoring` winner key; `match_whole_words` + `apply_replacement_once` conditional flags (`0 if entry.case_sensitive else re.IGNORECASE`); pipeline reorder in `process_user_input_with_diagnostics` (disabled filter + unified ordering + removed post-budget sort) with stage-diffs updated; `skipped:disabled` status.
- **Modify** `Character_Chat/local_chat_dictionary_service.py` — seam round-trip for the three fields (payload + response).
- **Create** `Widgets/Persona_Widgets/personas_dictionary_validation.py` — `ValidationFinding` + `validate_entries` (probes via `_entry_from_payload`).
- **Modify** `Widgets/Persona_Widgets/personas_dictionary_detail.py` — form fields, table column + disabled treatment, validation OptionList + jump, recompute hooks; `form_payload`/`load_dictionary`/`_fill_form_from_entry` extended.
- **Modify** `Widgets/Persona_Widgets/personas_dictionary_tryit.py` — reason map entry; `content_preview` flatten happens engine-side (see below), widget unchanged otherwise.
- **Modify** `Chat_Dictionary_Lib.py` `_finalize` — `content_preview=" ".join((candidate.content or "").split())[:40]` (the flatten lives where the preview is built).
- **Modify** `UI/Screens/personas_screen.py` — Duplicate payload field list ONLY. The validation panel is wholly widget-owned: the Detail widget recomputes on `load_dictionary`/`update_entries` (both already receive the full entry list) and posts nothing new to the screen; the inspector count rides the screen's existing selection/save refreshes calling `show_validation`.
- **Tests** — engine (`test_chat_dictionary_lib_diagnostics.py` + a new `test_chat_dictionary_lib_entries.py` if cleaner), service round-trip, validation module unit tests, UI (`test_personas_dictionaries.py`) form/table/panel/jump/duplicate/Try-it-disabled.

## Data flow / error handling

Unchanged shapes end-to-end: the three fields ride the existing entry dicts through list/get/add/update/duplicate; diagnostics ride the existing `diagnostics` key. Validation is computed client-side from the already-loaded entry list — no new service calls, no I/O in the module; a validation-compute failure logs and renders an empty panel (never blocks the editor). Priority Input parse failure → the existing inline form-error pattern ("Priority must be a whole number."). All engine error envelopes stay as P1b left them.

## Testing

- **Legacy-equivalence pins (the compat promise):** a no-priority/all-enabled/case-default dictionary produces byte-identical results to pre-P1c for: matching, group winners, application order, and off-budget-pressure outputs. (Budget-pressure ordering is the one spec'd delta — its own test pins strategy-order survival.)
- **Engine:** priority survival (high-priority survives tiny budget over cheaper-but-lower-priority), priority application order (applied_order follows -priority then strategy), disabled near-miss (`skipped:disabled`, fired elsewhere unaffected), case-sensitive literal match AND replacement (both sites — "BP" vs "bp"), group winner by priority (legacy length tie-break pinned too), wrapper contract still byte-identical for the legacy fixture.
- **Validation:** one unit test per code incl. the degraded-regex probe (`/[unclosed/` → `invalid_regex`) and a clean-entries → empty list.
- **Service:** three-field round-trip through create/get/update; `enabled` in responses reflects stored value.
- **UI:** form round-trips the new fields; disabled row renders dimmed + `off`; validation panel lists findings and jump moves the cursor; duplicate round-trip asserts all nine fields; Try-it renders "skipped: disabled".
- **Fake:** mirrors the real shape AND the semantics the tests depend on — disabled filter (`skipped:disabled`), priority ordering, walk-and-stop `break`. Case matching stays approximate (`pattern in text`); no test may depend on the fake's case behavior.

## Scope / non-goals

- **P1c does NOT** build Attachments/Stats/Versions tabs, import/export, bulk ops, per-entry usage counts, timed-effect persistence, or regex-safety (ReDoS) analysis (P1d / later). No live-as-you-type validation (recompute-on-mutation only). `ChatDictionary.matches()` stays untouched (dead code).
- **Forward:** P1d inherits the validation panel for import previews; P2 Lore inherits priority-aware budgeting wholesale (it was the worldbooks digest's #1 correctness gap).

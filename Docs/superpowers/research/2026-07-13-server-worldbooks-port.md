# tldw_server2 → World Books / Lorebooks (Lore) port research (digest)

Source: research agent over `/Users/macbook-dev/Documents/GitHub/tldw_server2` (canonical) vs tldw_chatbook (`world_book_manager.py`, `world_info_processor.py`, `ChaChaNotes_DB.py` V8→V9).

## Baseline correction (important)
tldw_chatbook is **AHEAD** of the server on world-book *structure*: it already implements `position` (before_char/after_char/at_start/at_end), `insertion_order`, `selective` + `secondary_keys` (AND-logic), and FTS5 tables — the server has NONE of these (flattens all lore into one block). tldw_chatbook is **BEHIND** on: regex matching, priority-aware budget survival, per-match diagnostics, attachment model (character-level), and any management UI.

## Server fields to port (tldw_chatbook lacks)
- **`priority` (0–100)** entry-level, drives injection order AND which entries survive token-budget truncation. tldw_chatbook only has `insertion_order` + walk-and-stop budget → arbitrary loss under pressure. **Real correctness gap.**
- **regex_match** + ReDoS-safe compile (`regex_safety.py` denylist + bounded test + timeout). tldw_chatbook: literal keyword only.
- **Per-match diagnostics**: `activation_reason`, matched `keyword`/`secondary_key`, `token_cost`, `depth_level`, `content_preview` + response-level `budget_exhausted`/`skipped_entries_due_to_budget`. tldw_chatbook's `_find_matching_entries`/`process_messages()` DISCARD all this → **biggest functional gap; blocks diagnostics UI**.
- whole_word toggle (chatbook hardcodes word-boundary); per-entry `recursive_scanning` opt-in seed (chatbook: book-level all-or-nothing, depth hardcoded 3 vs server configurable); `appendable` (no-separator concat); entry `group`/category; bulk ops; statistics; SillyTavern/Kobold IMPORT (chatbook only round-trips its own JSON).
- **NOT a real server feature (don't port as shipped):** "constant/always-on" entries — server has inert cache-hint metadata only, no CRUD field. `sticky` timed-effect also unimplemented.

## UX the server settled on
- **Two-panel List + tabbed Detail** (replaced 8+ modals): List (~35%: checkbox/Name+Desc/Entries/Status/Modified/Actions) + Detail (~65%, **tabs: Entries · Attachments · Stats · Settings**). Only Create / Relationship-Matrix / Test-Matching / Import stay modal. → Textual: DataTable left + TabbedContent right.
- **Progressive disclosure:** two-tier labels + "show technical labels" toggle ("Scan Depth"→"Messages to search", "Token Budget"→"Context size limit", "Recursive Scanning"→"Chain matching"); matching options (case/regex/whole-word) collapsed by default — only Keywords/Content/Priority shown for 90% of entries.
- **Empty state:** 3-step (Create → Add keyword entries → Attach) + concrete worked example ("keyword 'magic system' → user asks about it → AI receives your lore") + template quick-starts (Fantasy/Sci-Fi/Product-KB).
- **Import/export:** JSON round-trip + client-side format detection for SillyTavern V2 (`character_book.entries`) and Kobold; lossy fields surfaced as warnings.
- **Attach-to-character:** character shows attached-book chips; Relationship Matrix (scales poorly >10 — avoid). Audit P1: chat dictionaries NOT character-visible like worldbooks are.

## UNIQUE tldw_chatbook opportunity (server never built this)
tldw_chatbook has Characters + Personas + Dictionaries + Lore as **sibling modes in ONE workbench**. The server scatters them. → Studio can offer a shared **"what's in play"** / attachments view: for a given character or conversation, show the persona + lore books + dictionaries that apply. Also adopt **character-level attachment** (not just conversation) so one lorebook/dictionary serves all a character's chats.

## Prioritized ports
- **HIGH:** (1) **Trigger-diagnostics "Test Match" panel** (the Lore "Try it"). (2) surface match diagnostics from `world_info_processor` (activation_reason/keyword/token_cost/depth + budget_exhausted/skipped IDs) — prerequisite. (3) entry `priority` + priority-aware budget trim (correctness). (4) two-panel List+tabbed-Detail layout. (5) character-level attachment.
- **MED:** expose selective/secondary_keys + position in editor (**UI-only — data model already supports!**); regex + safety; bulk ops; statistics; token-budget bar; onboarding empty state; wire existing FTS5 to search.
- **LOW:** group/category+filter; SillyTavern/Kobold import + lossy warnings; appendable; duplicate; cross-book stats; defer AI-gen; surface runtime constants in help.

## "Try it" (Trigger diagnostics) — TUI design brief
**Stage 1 — ad-hoc Test Match pane** (Lore mode, keybinding `t`): inputs = sample-text TextArea (+ "pull last N turns from active conversation" toggle), a checklist of which books to test (default = attached to active character/conversation), scan_depth/token_budget steppers + recursive toggle (prefilled from book settings). Run → new diagnostics-returning match fn.
Results top→bottom: (a) summary strip — entries matched, books used, tokens used/budget (color bar), skipped-due-to-budget (amber if >0); (b) DataTable of matched entries in injection order — content preview, source book, **activation_reason** ("keyword: dragon" / "secondary+primary: fire+dragon" / "regex: /drag.n/i" / "depth 1 (recursive): ember"), token cost, priority; (c) empty state.
**Beyond the server (answers "why did X NOT fire"):** a **"near misses"** section — entries that matched a primary key but were skipped (disabled / failed required secondary key / dropped by budget). Server only reports *matched* entries; extend the scan to keep skipped IDs+reasons, not just a count. Export diagnostics as text/JSON (one keystroke).
**Stage 2 (later):** live per-turn indicator ("Lore: 3 entries, 210/500 tok") reusing Stage 1's results view with real last-turn diagnostics. Depends on Stage-1 diagnostics plumbing landing first.

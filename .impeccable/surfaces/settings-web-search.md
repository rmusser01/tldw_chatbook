# Settings — Web Search

Mode: Operate. Disposition: ship after the finish-review and TASK-32191 lifecycle fixes; no outstanding findings in the reviewed scope.

## Authority and ownership

This is an ordinary extension of the incumbent Textual Settings surface. The main project’s `PRODUCT.md` and `DESIGN.md` at `/Users/macbook-dev/Documents/GitHub/tldw_chatbook` remain the product and visual authority. This record adds local guidance; it establishes no new visual world, tokens, or shared components. The implementation is the design artifact; no separate comp or concept card is required.

- `tldw_chatbook/Widgets/settings_web_search_panel.py` owns the form and its direction contract.
- `tldw_chatbook/UI/Screens/settings_web_search.py` owns the live session draft, selected backend, save state, and test-result validity across destination recreation. The panel is a disposable view.
- `tldw_chatbook/UI/Screens/settings_screen.py` supplies category placement, the pinned save contract, and native Settings Save/Revert behavior.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` supplies production styling. Backend fields, setup requirements, notices, and guide links come from the shared `Web_Scraping/search_backend_settings.py` catalog.

## Local extension contract

Keep the reading order: **Default search backend → Configure backend → required fields and value sources → local setup status → explicit Test saved settings**. The first viewport prioritizes the default and separate backend editor. Selecting a default opens that backend’s fields; configuring another backend preserves the default shared by basic and deep search.

Use the Settings detail pane as the single scroll owner, with stacked labels, native selects, masked inputs, compact actions, and existing semantic tokens. Keep Save/Revert in the form at compact widths, routed through the screen’s actions and Revert confirmation. Explain disabled save controls with readable “No unsaved changes.” text. Single-letter shortcuts retain Settings’ text-entry focus rules.

Provider, category, and destination navigation must retain drafts, including masked replacement keys and the selected backend. Keep the live session in the existing memory-only screen-state store and persistence/test workers at app lifetime. Capture pending input before async completion checks; a status paint must not overwrite a widget value waiting for its change event. Empty replacement fields keep saved secrets; Clear stages removal of a local value. Source labels explain environment precedence without displaying secret values.

Keep setup, persistence, and network-test states separate. “Required settings present” and a successful save do not establish authentication or quota. Test saved settings requires a clean draft and complete local setup; its copy names the sample query, possible API-quota use, and absence of an AI-generated answer. Edits and navigation invalidate displayed test evidence. Keep provider restrictions in the shared catalog instead of duplicating them in view logic.

Capture pending input before applying Clear or confirmed Revert, so a status refresh cannot restore discarded text. Distinguish a failed write (draft retained) from a committed write with failed reload (saved to disk, restart required). During navigation, a pending test stays busy until it finishes and its obsolete result is discarded; new view mounting must invalidate evidence even when it precedes the old view's teardown.

## Evidence and limits

Reviewed captures under `.impeccable/review/web-search/`: `settings-120x35.png` and `settings-80x24.png` show entry; the corresponding `settings-120x35-test.png` and `settings-80x24-test.png` show scrolled test controls. These came from mounted Textual with production CSS. SVG-to-PNG rendering used Menlo because Fira Code was unavailable; this is layout evidence, not exact terminal-font evidence. No live provider requests were made, so authentication, network reachability, quota, and provider responses remain unverified by this review.

User-facing walkthroughs live in `Docs/User_Guide/settings.md` (first setup and additional backends) and `Docs/User_Guide/console/agent-runs-and-tools.md` (shared default and per-call overrides).

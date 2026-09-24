# Size decomposition candidates (core review 2026-09-17)

**Status: a record, not a work order.** This file exists so that a future
decomposition of the repository's largest modules starts with the
responsibility-cluster maps and line ranges the core-runtime code review
(`qa/core-code-review-2026-09-17/report.md`) already produced, instead of
re-deriving them. **No decomposition is opened by recording them here**
(core-review TASK-32809.3 AC#3). Every split below must follow the repo's own
recipe — `backlog/docs/library-decomposition-recipe.md` — never a rewrite off
the back of a review finding:

- **§1** per-subsystem pull-request series (one subsystem per PR, verbatim moves)
- **§2** the field-ownership script (which `__init__` attribute each cluster owns; the ≥2-subsystem rule)
- **§3** monkeypatch-name routing (several of these files are patched by name in dozens of tests — a move that breaks an import path breaks the suite silently)
- **§6** measure after the final rebase; lower/re-pin the ratchet row in the landing PR
- **§17** file-size governance (add the moved-into file's `_BUDGETS` row in the same PR that first moves code)

Line ranges are the review's, measured at the review commit; re-measure before
acting (the files have grown since — see the ratchet re-pins in
TASK-32809.1/.2). Size ratchets now guard all of these: screens in
`Tests/Architecture/test_screen_size_ratchet.py`, `Library_Modules/*_controller.py`
in `test_library_modules_size_ratchet.py`, and the rest in
`test_module_size_ratchet.py`. `settings_screen.py` (31,781 lines) is **not**
listed here — its split and ratchet row are already tracked by **task-1378 /
task-31202** (linked, not duplicated).

**Correction, 2026-09-21 (tier-2 review, task-32901).** "Size ratchets now
guard all of these" was true of the modules this file listed and false of the
repository: four modules LARGER than three of the seven rows TASK-32809.2 added
had no row anywhere, and four separate tier-2 slices each reported one of them
independently. They now have rows in `test_module_size_ratchet.py`, pinned at
their exact measured size as of `origin/dev` 9e33252708:

| module | lines | class / methods | responsibilities (tier-2 count) |
|---|---:|---|---|
| `UI/Screens/watchlists_collections_screen.py` | 14,324 | `WatchlistsCollectionsScreen`, 392 methods | **12** — tree/scope navigation; watchlist CRUD; source CRUD + OPML; runs; notifications inbox; briefings (generate/keep/export/presets/cadence/schedules); scripts + cast; audio synthesis + playback + feed server; items reader (paging/snapshot/filter/search/status/star); rules + noise selectors; region-layout persistence + responsive relayout; keyboard actions |
| `UI/Wizards/FirstRunSetupWizard.py` | 10,404 | — | obvious first extraction: `ProviderStep` (`:1148-3299`, ~2,150 lines of endpoint resolution / discovery / probe evidence that own no pixels outside `compose_step`) |
| `UI/Screens/llm_screen.py` | 5,180 | `LLMScreen`, 173 methods | **7** |
| `UI/Screens/change_review_screen.py` | 4,967 | 91 methods + 3 modal classes + 17 module functions | — |

As everywhere else in this file, **recording a row opens no decomposition**;
§2's field-ownership script has to run first. `test_library_modules_size_ratchet.py`'s
glob is `Library_Modules/*_controller.py`, which matches none of these, and
`test_screen_size_ratchet.py`'s rows additionally pin a per-class method count
that needs a decomposition plan to be meaningful — hence the module ratchet,
following the `personas_screen.py` precedent already there.

---

## `UI/Screens/chat_screen.py` — `ChatScreen`, ~25k lines
Recipe: §1 / §2 / §17. Candidate first extractions (own no pixels, move
verbatim): the settings-durability / default-intent state machine (2804–3320 +
5569–6100, ≈1,050 lines of pure state over `app_instance` attributes), the
Conversation-settings suspend/return handoff (3318–4190), the roleplay
persistence drain (4192–4780), the Environment focus-restore cluster (8512–9330).

Full cluster map (line ranges, all read): imports 1–696 · module
constants/helpers 697–1757 (turn-undo planner 970–1093, inspector exchanges
loader 1585–1696) · bindings/actions/focus 1758–2160 ·
left-rail/terminal/inspector-section events 2173–2810 · settings-durability +
default-intent state machine 2804–3320 · Conversation-settings modal
open/suspend/return handoff 3318–4190 · roleplay projection persistence drain
4192–4780 · workbench help/F6 panes 4782–5245 · session switcher + model popover
5245–5605 · settings submission durability + default recovery 5606–6100 ·
palette actions 6101–6465 · row/workspace action menus + markdown export
6465–7240 · class attrs + `__init__` 7239–7560 · config memo / provider intents /
vLLM handoff 7560–8340 · context estimate / control state 8338–8500 ·
Environment/Tasks section + focus restore 8512–9330 · settings summary / agent
section sync 9323–9620 · provider-selection derivation + runtime-handle
properties 9626–10000 · store/controller/gateway ensure + view hooks
10000–10500 · ~60 proxy properties to controllers 10528–10860 ·
TTS/voice/impersonate/composer menu 10854–11340 · hands-free/realtime
delegations 11341–11470 · collapse toggles 11466–11620 · control state + cost
chip + timers 11621–12000 · retrieval scope / chips / inspector push
12002–12500 · character avatar render 12502–12800 · rail prefs persistence +
onboarding flags 12792–13320 · rail state/preferences/visibility 13320–14020 ·
workspace context sync 14022–14180 · inspector state build 14187–14400 ·
dictionary/world-book appliers + 4 attach/detach workers 14390–14690 ·
inspector rows / live-work cards 14684–15100 · readiness copy / blockers / setup
modal 15015–15650 · live-work strip/swap/center builder 15649–16070 ·
`compose_content` 16071–16632 · on_mount / attach reconciliation / resume startup
16633–16960 · on_unmount 16954–17010 · state serialize/restore + attachment stash
17011–17430 · handoff consumers 17433–17690 · citation counts 17683–18030 ·
transcript fingerprint + `_sync_native_console_transcript` 18029–18345 · run copy
/ mode bar / sync maintenance 18355–18500 · `_sync_native_console_chat_ui` + tabs
+ 0.2 s poll 18496–18810 · send pipeline 18811–19130 · slash commands
19131–20070 · collapse/stop/attach/paste/clipboard/chatbook 20071–20460 · change
review / turn undo / approval focus 20456–20890 · canvas bridge 20892–21310 ·
summarize range 21327–21500 · control-bar sync / coalescing / config-sync / popup
21497–21960 · composer undo/redo + action state 21956–22270 · resize /
rail-collapse notice / focus frame 22268–22450 · `on_key` 22447–22690 · selection
/ side-chat / review notes / paste / mouse 22690–23300 · compact shell sync
23294–23390 · suspend/resume 23396–23660 · watchlists ops 23656–23860 · task
cards / questions / park approvals / run toasts / agent chat create 23853–24460 ·
sidebar state persistence 24664–24843 · collapsibles 25089–25206 · lazy modal
loader tail.

## `Chat/console_chat_controller.py` — ~29k lines
Recipe: §1 / §2 / §3 (patched by name in dozens of tests) / §17. No single
consolidated cluster map in the review; the actionable anchors it gives:
- Longest bodies: `_submit_draft_body` 8800–10375, `__init__` 3758–4588,
  `_run_agent_reply` 26259–27221, `resume_durable_postcommit` 11206–11883,
  `_apply_conversation_memory_preflight` 23892–24411.
- Candidate first extractions (fewest shared fields): module-level review-hook
  builders (1626–3040, no `self`), project-instruction authority (827–1625, no
  `self`), the settings rebase (12648–12885, pure), the interrupt bridges
  (13382–17874, already mostly delegated to `InterruptRoundHost`).

## `Chat/console_chat_store.py` — ~22k lines
Recipe: §1 / §2 / §17. The review did not produce a consolidated cluster map for
this file; a future split should derive one with the §2 field-ownership script
first. Subsystem anchors the review touched: trajectory writes 14357–14365,
exchanges-only persist 20174–20236, dispatch-branch mutation 21971–21994,
settings drain 9006–9128, stream-buffer materialization / pending-message
persist, context-summary-on-resume 22083–22124.

## `DB/ChaChaNotes_DB.py` — `CharactersRAGDB` :709–23951 (+ module helpers :1–708, `TransactionContextManager` :23952–24180), ~24k lines
Recipe: §1 / §2 / §17. Obvious first PRs: the self-contained study cluster
(13–15, ~2.7k lines) and the migration steps (5, ~4.2k lines). 15 clusters:
(1) module-level validators/authorizations/SQL splitters :1–708; (2) schema +
25 migration SQL literals as class attributes :720–3305; (3) connection
lifecycle, quiescence, local-authority, backup/integrity :3306–3988;
(4) `execute_query`/`execute_many`/`transaction` :3989–4197 + context manager
:23952–24180; (5) migration runner primitives + 69 `_migrate_from_*` steps
:4198–8420; (6) `_initialize_schema` + per-open repair hooks :8421–8720;
(7) character cards + expression images + FTS :8951–10402; (8) conversations
:10403–13165; (9) messages :13166–16493; (10) generic-item CRUD +
keywords/collections :16494–17461; (11) notes + owner proofs + links + Library
note seams :17462–19431 (Library conversation seams interleaved :18466–18874);
(12) link tables + sync_log intents/retention/prune + `backfill_messages_fts`
:19432–21214; (13) flashcards/decks/templates/assets :21242–22256; (14)
quizzes/questions/attempts/grading incl. a Levenshtein implementation
:22257–23478; (15) learning paths/topics/stats + kept briefings/scripts
:23402–23951.

## `app.py` — ~21k lines, 16 classes
Recipe: §1 / §17. Candidate first extractions in order: (1) `LibraryIngestQueueMixin`
2585–7089 → `Library/ingest_queue_mixin.py` (already a mixin; keep a
`from tldw_chatbook.app import LibraryIngestQueueMixin` re-export — 5 tests
import it by that path); (2) the 10 palette providers 1171–2222 →
`UI/command_providers.py` (2 tests import `ThemeProvider`/`TabNavigationProvider`
from `app`); (3) the `_wire_*` composition 9520–11464 → an `app_wiring.py`
function set taking `app`. §2 field-ownership script before each move.

## `UI/Screens/personas_screen.py` — `PersonasScreen`, ~16k lines
Recipe: §1 / §2 / §17. Natural first peels (each already has its own snapshot
dataclasses at 671–826): Character TTS controls (2523–3475, ~950 lines), Actor
Pack export/import/create (4604–5131 + 7512–7920, ~1,000), Persona Visual pack
authoring incl. `_persona_visual_thread` (8920–10205, ~1,300), Character visual
identity pack incl. `_visual_identity_thread` (11004–12353, ~1,200). Other
clusters: module constants + 12 frozen snapshot dataclasses + drains + lifetime
decorators 1–1067; compose/state round-trip/mount/demand-mounted center views
1529–2035; character-conversation deep link 2077–2223; runtime-backend switch +
responsive rails 2225–2397; library paging/sort/search/tag + dictionary/lore row
rendering 2398–2522 + 3480–4386; mode switching + header copy 4387–4603;
selection 5132–5540; dictionaries incl. character attach 5541–6485; lore/world
books 6485–6932; saved conversations + Console handoff + preview delegation
6933–7389; create/edit/duplicate/toggle 7390–8302; Persona shared visual
identity 8303–8919; character edit + visual-identity load + avatar upload
10206–10575; LLM-assisted character generation 10576–10723; avatar/expression
thumbnails 10724–11003; expression slots upload/generate/style 12355–13342;
expression-set import/export 13343–13547; import 13548–14288; export 14289–14638;
delete 14639–15108; character save 15109–15432; policy rules + persona save
15433–15703; cancel 15704–15763; `_show_center` + draft snapshot + navigation
veto 15764–16058; `_run_guarded`, key bindings, focus, footer sync 16060–end.

## `Widgets/Console/console_transcript.py` — `ConsoleTranscript` at 2969–8333 (~5.4k-line class, 193 methods)
Recipe: §17 (governed now by `test_module_size_ratchet.py`) + §1 / §2 for a
split. Clusters inside the one class: windowing/hydration 3491–4197 ·
presentation setters 4198–4738 · pruning 4842–5058 · thinking/activity projection
5085–5182 · keyboard text-selection mode 5412–5623 · mouse drag-selection +
floating menu 5750–6460 · row planning 6623–7030 · row build/reconcile 7074–7742 ·
signatures/caching 7744–7831 · action rows + overflow menu 8124–8333.

## `Widgets/Console/console_settings_modal.py` — `ConsoleSettingsModal` at 1005–7601 (~6.6k-line class, 245 methods)
Recipe: §17 + §1 / §2. Clusters: `compose()` 1589–2593 (~970 lines) ·
focus/scroll/layout 2837–3253 + 3787–3828 + 4023–4139 · draft snapshot/restore
2695–2836 · default-durability recovery 3522–3772 · context-policy controls
4267–4530 + 7238–7394 · provider/model rebase 5047–5560 · model discovery +
connection probes 5560–6486 · generation test 5928–6127 · readiness sync
6488–6612 · provider/base-URL resolution 6858–7208.

## `UI/MCP_Modules/mcp_workbench.py` — ~6.5k lines
Recipe: §1 / §17. The review named responsibilities (no line ranges); a split
should derive ranges with the §2 script. The two clean first extractions: the
prepared Tool-Test admission/nonce/lease state machine (~600 lines) and the
permission-matrix/row derivation (`_tool_policy_inventory`,
`_capture_permission_render_state`, `_build_permission_rows`,
`_build_permission_preview`, `_builtin_permission_matrix_rows`). Other
responsibilities: triad assembly + deferred canvas mounting; readiness snapshot
+ CHECKING overlay; local/server source switch + rail scope; Tools catalog
derivation; profile CRUD + `mcpServers` import w/ path validation;
server-mutation/credential-slot panel wiring; audit log + findings;
recovery-review dialog flow; lifecycle dispatch + in-flight bookkeeping;
view-state save/restore.

---

### Other modules >5,000 lines the review flagged (no cluster map produced here)
See the review's coverage table (`qa/core-code-review-2026-09-17/report.md`) for
the full population. Notably `Widgets/Console/` (a package-wide gap the review
recommends a sibling glob ratchet for — `test_module_size_ratchet.py` now pins
its two biggest), the `LibraryFileNotesWorkspace` 8,846-line widget whose git
half duplicates the git panel beside it (report §W-library), `RAG_Search/`
(6 of 14 files >600 lines), and the `DB/` remainder. Each is governed by the
same recipe; none is opened here.

---

### `UI/Speech/` — a 5,707-line mixin namespace shared with no declared owner (tier-2 review S21, 2026-09-21)

**Record only; explicitly NOT a decomposition candidate.** This is the inverse
shape of everything above: the Speech Playground is already split -- five
mixins totalling 5,707 lines feeding a 3,028-line pane -- and the problem is
that the split has no ownership contract, not that a split is needed.

Re-measured at `origin/dev` 9e33252708 (AST, `self.X = ...` write sites):

| mixin | lines | attributes it assigns |
|---|---:|---:|
| `speech_catalog_mixin.py` | 1,859 | 31 |
| `speech_playback_mixin.py` | 1,291 | 12 |
| `speech_settings_mixin.py` | 1,194 | 6 |
| `speech_profile_mixin.py` | 704 | 18 |
| `speech_synthesis_mixin.py` | 659 | 5 |

**13 attributes are written by two or more co-mounted mixins**, led by
`_generation_operation_id` (3 mixins, 9 write sites), `_profile_save_suppressed`
(3), `_provider_ids` (3), `_profile_effective_availability` (catalog x10 /
profile x3), `_profile_preview_loading` (x5 / x3) and
`_profile_voice_validation_token` (x3 / x4). None of the five declares
ownership of any of them.

**Why nothing was changed for this.** The review's own evidence records **zero
live MRO collisions** among the four co-mounted mixins (the four method-name
collisions all pair `SpeechSettingsMixin` with a mixin it never co-occurs
with), and the one real double-dispatch is deliberate and documented at
`speech_settings_pane.py:1538-1540`. The order dependency the finding raises --
`speech_playground_pane.py:2734` calling `init_profile_state` a second time on
preset re-adopt -- is guarded: that call site retires the generation context
and the test authority immediately before re-initialising, which is exactly
the invalidation that makes the re-zero safe. So the recommended fix (one
`SpeechPlaygroundState` dataclass holding the 13 fields as `self._state`) is a
mechanical rewrite of 40+ write sites across five files with **no observable
behaviour change and no demonstrated defect**. Recorded here so a future
`UI/Speech/` change starts from the ownership map instead of re-deriving it.

# Library Starter Rail and Landing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a genuinely new, empty Library profile a compact Get started rail and landing while preserving the complete Library for existing users, deep links, command routes, and profiles that have already expanded or added usable content.

**Architecture:** Extend the existing Library rail preference owner with one small lifecycle enum and pure transition functions. Each existing local source owner exposes one provenance-aware tri-state evidence method that returns no records or private content; `LibraryScreen` gathers those six facts under one generation, aggregates them, serializes lifecycle writes through the existing config owner, and owns transition timing. `LibraryRail` and `LibraryLandingCanvas` only render that state using existing production Import, New note, and navigation actions. The broad source snapshot and its cache remain presentation/RAG accelerators and never participate in onboarding evidence.

**Tech Stack:** Python 3.11, Textual 8.x, SQLite, pytest

---

## Scope and governing records

- Approved design: `Docs/superpowers/specs/2026-08-20-library-lifecycle-progressive-disclosure-design.md`
- Backlog task: `backlog/tasks/task-19022 - Add-Library-starter-rail-and-lifecycle-aware-landing.md`
- New decision: `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`
- Existing paging/authority decision: `backlog/decisions/067-library-top-level-pagination-contracts.md`
- This plan implements only the first atomic Wave 1 slice: the Starter rail, lifecycle-aware landing, graduation, persistence, and recovery. Source-specific Notes/Media/Conversations/Prompts/Skills/Collections/Search/Import/Export empty canvases remain outside this task's implementation scope.
- Per user direction, do not run repository-wide pytest. Run only the modified/touched components and direct owners named below.

## UX contract

The production hierarchy remains one Library shell. Starter is a filtered rail and landing presentation, never a second router:

```text
NEW PROFILE / EVIDENCE UNRESOLVED
+----------------------+----------------------------------------+
| LIBRARY              | GET STARTED                            |
|                      |                                        |
| > Import content     | Add something useful, then use it in   |
|   New note           | Console or Study.                      |
|                      |                                        |
|   Explore all tools  | [ Import content ]  [ New note ]       |
|                      | Checking existing Library content…     |
+----------------------+----------------------------------------+

NEW PROFILE / AUTHORITATIVELY EMPTY
+----------------------+----------------------------------------+
| LIBRARY              | GET STARTED                            |
|                      |                                        |
| > Import content     | 1 Add  ->  2 Find  ->  3 Use           |
|   New note           |                                        |
|                      | [ Import content ]  [ New note ]       |
|   Explore all tools  |                                        |
+----------------------+----------------------------------------+

EXPANDED OR GRADUATED
+----------------------+----------------------------------------+
| Search Library       | LIBRARY                                |
| Browse               | Current counts, recents, and normal    |
| Create               | production actions.                    |
| Study                |                                        |
| Ingest               |                                        |
| Details              |                                        |
+----------------------+----------------------------------------+
```

Rules:

- `unknown`, `starter`, `expanded`, and `graduated` are profile-local persisted lifecycle values.
- Missing lifecycle on a newly created profile starts `unknown`; missing lifecycle on a legacy profile starts `expanded`; malformed storage also starts `expanded`.
- `unknown` and `starter` show the same three safe production actions. Only the status copy differs: unresolved/partial evidence never says the Library is empty.
- Any eligible positive fact makes `graduated` sticky. Deleting everything never moves it backward.
- `Explore all tools` persists `expanded` independently of section collapse. An empty expanded profile may explicitly return to Get started; a graduated profile cannot.
- Existing deep links, pending navigation context, keyboard routes, and command-palette admissions operate on the full shell state and bypass rail filtering.
- The application-wide first-run wizard remains the sole startup owner. The config loader records one process-session admission fact when it creates the active profile config; the app captures that durable in-session fact before later config reloads can lose the transient `_first_run` key. Library only reads `app_instance.library_new_profile_admission` and never writes the admission fact or wizard state.
- Import and New note reuse `LIBRARY_ROW_INGEST_MEDIA` and `LIBRARY_ROW_CREATE_NOTE` through the existing `.library-rail-row` / `.library-hub-action` dispatch.

## Eligibility and evidence matrix

```text
Source          Source-owned proof                   Exclusions
Notes           active local user-note count          deleted/inaccessible
Media           active non-Trash local media count    Trash/deleted/incomplete
Conversations   saved local conversation count        absent/failed saves
Prompts         active local user-prompt count         deleted/bundled/sample
Skills          available local user-skill count      blocked/quarantined/bundled
Collections     active local user-collection count     soft-deleted
```

- Every evidence method returns only `LibraryContentEvidence`; it may use an existing exact count internally only after its owner tests prove that every counted record is eligible. If the owner cannot prove provenance, it returns `unknown` rather than exposing the count to Library.
- A missing seam, source failure, inexact total, malformed payload, or timeout is `unknown`.
- Blocked/inaccessible Skills do not graduate and do not prevent an otherwise successful Skills owner from reporting empty usable content.
- Deleted/Trash records are excluded inside their source owner. Prompt/Skill tests explicitly prove bundled/sample/system values cannot be converted into positive evidence. Cached sample rows and broad record tuples are never used as evidence.
- Collections are the one necessarily local source: `LocalLibraryCollectionsService` has no bundled seeder, remote adapter, or per-record ACL, and active rows can only be created/restored through that user-owned service. Its owner test proves a fresh DB is empty, `create_collection` is the only positive path, and soft deletion removes the evidence.
- One positive evidence value graduates immediately after the guarded read settles. Starter requires all six values to be exact empty in the same generation.

## Task 1: Commit governance and add pure lifecycle contracts

**Files:**

- Create: `tldw_chatbook/Library/library_content_evidence.py`
- Modify: `tldw_chatbook/Library/library_rail_state.py`
- Create: `Tests/Library/test_library_content_evidence.py`
- Modify: `Tests/Library/test_library_rail_state.py`
- Verify/add: `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`
- Verify/add: `backlog/tasks/task-19022 - Add-Library-starter-rail-and-lifecycle-aware-landing.md`

- [ ] **Step 1: Add RED lifecycle coercion tests.**

  Add focused cases named:

  ```python
  def test_missing_lifecycle_uses_unknown_only_for_new_profile(): ...
  def test_corrupt_lifecycle_fails_safe_to_expanded_without_resetting_sections(): ...
  def test_lifecycle_round_trips_beside_section_preferences(): ...
  ```

  Require exact string values and prove lifecycle coercion is independent from `LibraryRailPreferences` coercion.

- [ ] **Step 2: Add RED evidence/transition tests.**

  Add:

  ```python
  def test_any_usable_content_graduates_and_graduation_is_sticky(): ...
  def test_starter_requires_every_source_to_report_empty(): ...
  def test_unknown_evidence_never_claims_starter(): ...
  def test_explore_expands_separately_and_empty_expanded_can_return_to_starter(): ...
  ```

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Library/test_library_content_evidence.py Tests/Library/test_library_rail_state.py
  ```

  Expected RED: lifecycle/evidence types and transition functions do not exist.

- [ ] **Step 3: Implement the smallest pure model.**

  Add only:

  ```python
  class LibraryLifecycle(str, Enum):
      UNKNOWN = "unknown"
      STARTER = "starter"
      EXPANDED = "expanded"
      GRADUATED = "graduated"

  ```

  Put the shared three-value `LibraryContentEvidence` enum (`unknown`, `empty`, `has_user_content`) and three-value presentation status `LibraryEvidenceStatus` (`loading`, `settled`, `partial_failure`) in the neutral `library_content_evidence.py` module so source services do not depend on rail presentation. Keep lifecycle aggregation in `library_rail_state.py`. Add pure coercion, serialization, and transition helpers. Use `None` as the absent-storage input; any other invalid value is corrupt and maps to `expanded`. Do not add a controller, state machine framework, or generic preference store.

- [ ] **Step 4: Verify GREEN and inverse behavior.**

  Temporarily make missing lifecycle always `unknown`; the legacy-profile test must fail. Restore immediately and rerun the owner file GREEN.

- [ ] **Step 5: Update the Backlog task Implementation Plan.**

  Link this plan and ADR-076 in the task file before implementation continues.

- [ ] **Step 6: Commit.**

  ```bash
  git add tldw_chatbook/Library/library_content_evidence.py tldw_chatbook/Library/library_rail_state.py Tests/Library/test_library_content_evidence.py Tests/Library/test_library_rail_state.py backlog/decisions/076-library-lifecycle-progressive-disclosure.md 'backlog/tasks/task-19022 - Add-Library-starter-rail-and-lifecycle-aware-landing.md' Docs/superpowers/plans/2026-08-20-library-starter-rail-landing.md
  git commit -m "feat(library): define starter lifecycle state"
  ```

## Task 2: Add source-owned usable-content evidence seams

**Files:**

- Modify: `tldw_chatbook/Notes/notes_scope_service.py`
- Modify: `tldw_chatbook/Media/media_reading_scope_service.py`
- Modify: `tldw_chatbook/Chat/chat_conversation_scope_service.py`
- Modify: `tldw_chatbook/Prompt_Management/prompt_scope_service.py`
- Modify: `tldw_chatbook/Skills_Interop/skills_scope_service.py`
- Modify: `tldw_chatbook/Library/library_collections_service.py`
- Modify: `Tests/Notes/test_notes_scope_service_library_canvas.py`
- Modify: `Tests/Media/test_media_reading_scope_service.py`
- Modify: `Tests/Chat/test_chat_conversation_scope_service.py`
- Modify: `Tests/Library/test_library_prompts_seam.py`
- Modify: `Tests/Skills/test_skills_scope_service.py`
- Modify: `Tests/Library/test_library_collections_service.py`

- [ ] **Step 1: Add RED contract tests to every existing source owner.**

  Add one method with the same exact public name on each scope/service owner:

  ```python
  async def get_library_user_content_evidence(...) -> LibraryContentEvidence: ...
  # Collections stays synchronous because its existing owner is synchronous.
  ```

  Owner tests must prove that the method returns only the enum and no records, titles, paths, bodies, or IDs. Cover exact empty, eligible positive, malformed/failure, and the source exclusions that matter:

  ```text
  Notes          active local user record; deleted/inaccessible excluded
  Media          active non-Trash record; deleted, Trash, incomplete excluded
  Conversations  durably saved local conversation; missing/failed save excluded
  Prompts        active local user prompt; deleted and any bundled/sample fixture excluded
  Skills         available local user skill; blocked/quarantined/bundled excluded
  Collections    active local collection; soft-deleted excluded
  ```

  If an owner cannot prove that its existing count contains only eligible content, its method returns `LibraryContentEvidence.UNKNOWN`; LibraryScreen never reinterprets that count.

- [ ] **Step 2: Implement thin owner-specific adapters.**

  Reuse each owner's existing local count/list/context policy and off-loop behavior internally:

  - Notes and Prompts may map an already-proven exact active count for the requested authority; where server lacks count-only support, a bounded one-row list envelope may supply the validated total.
  - Media and Conversations may request a one-row summary inside the owner for the requested authority, validate the exact total, and return the enum; no row escapes.
  - Skills maps only the validated `available_skills` population from the requested authority; `blocked_skills` is explicitly not positive.
  - Collections performs one active `COUNT(*)` and returns the enum; it does not page a record.

  Each scope method accepts the same `mode`/scope authority its existing Library source owner uses. Accessible authoritative local **or server** user content can therefore graduate; unsupported or provenance-ambiguous authority returns `UNKNOWN`, never a guessed Empty. Collections remains local because that feature has only one local authority. Let owner failures raise their existing safe exception (or return `UNKNOWN` only where that is already the owner's normal unavailable contract); the screen converts per-source failures to Unknown. Do not log values.

  Add at least one server-mode fake per scope family proving an exact accessible positive survives, plus an unsupported/ambiguous server response proving `UNKNOWN` rather than false Empty.

- [ ] **Step 3: Pin aggregation independently.**

  In `Tests/Library/test_library_rail_state.py`, prove the lifecycle resolver accepts exactly six enum values; any positive wins, all six exact empty values are required for Starter, and one Unknown prevents negative settlement. There is no records/count parameter, making broad-cache influence impossible by construction.

- [ ] **Step 4: Run only these source owners.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Notes/test_notes_scope_service_library_canvas.py -k 'user_content_evidence or count_notes'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Notes/test_server_notes_workspace_service.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Media/test_media_reading_scope_service.py -k 'user_content_evidence or library_summary'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Media/test_local_media_reading_service.py -k 'library_media_summary'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/DB/test_client_media_debug_logging.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Chat/test_chat_conversation_scope_service.py -k 'user_content_evidence or list_conversations'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Library/test_library_prompts_seam.py Tests/Skills/test_skills_scope_service.py Tests/Library/test_library_collections_service.py \
    -k 'user_content_evidence or count_prompts or get_context or collections'
  ```

- [ ] **Step 5: Run the exclusion inverse.**

  Temporarily count blocked Skills as usable content; the Skills evidence test must fail. Restore immediately and rerun its focused owner GREEN.

- [ ] **Step 6: Commit.**

  ```bash
  git add \
    tldw_chatbook/Notes/notes_scope_service.py \
    tldw_chatbook/Media/media_reading_scope_service.py \
    tldw_chatbook/Chat/chat_conversation_scope_service.py \
    tldw_chatbook/Prompt_Management/prompt_scope_service.py \
    tldw_chatbook/Skills_Interop/local_skills_service.py \
    tldw_chatbook/Skills_Interop/skills_scope_service.py \
    tldw_chatbook/Library/library_collections_service.py \
    Tests/Notes/test_notes_scope_service_library_canvas.py \
    Tests/Media/test_media_reading_scope_service.py \
    Tests/Chat/test_chat_conversation_scope_service.py \
    Tests/Library/test_library_prompts_seam.py \
    Tests/Skills/test_skills_scope_service.py \
    Tests/Library/test_library_collections_service.py \
    tldw_chatbook/Library/library_rail_state.py \
    Tests/Library/test_library_rail_state.py
  git commit -m "feat(library): expose usable content evidence"
  ```

## Task 3: Make LibraryScreen own guarded evidence and persistence

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_shell.py`
- Direct owner check: `Tests/Library/test_library_snapshot_cache.py`

- [ ] **Step 1: Add RED mounted lifecycle authority tests.**

  Add production-shaped fakes and cases named:

  ```python
  async def test_library_onboarding_new_profile_waits_for_all_fresh_empty_evidence(): ...
  async def test_library_onboarding_cached_zero_snapshot_cannot_declare_starter(): ...
  async def test_library_onboarding_partial_failure_keeps_truthful_unknown_actions(): ...
  async def test_library_onboarding_fresh_usable_content_graduates_and_persists(): ...
  async def test_library_onboarding_blocked_skill_and_trash_only_do_not_graduate(): ...
  async def test_library_onboarding_late_generation_and_unmount_cannot_apply(): ...
  async def test_library_onboarding_legacy_and_corrupt_preferences_open_expanded(): ...
  async def test_library_onboarding_restart_after_partial_failure_restores_unknown(): ...
  async def test_library_onboarding_restart_restores_each_settled_lifecycle(): ...
  async def test_library_onboarding_explore_during_read_still_allows_graduation(): ...
  async def test_library_onboarding_new_generation_revokes_back_to_starter(): ...
  async def test_library_onboarding_persistence_failure_keeps_session_and_warns(): ...
  async def test_library_onboarding_positive_wins_while_another_owner_hangs(): ...
  async def test_library_onboarding_hanging_owner_times_out_to_retry(): ...
  ```

  Gate the six source-owned evidence methods independently with Events. The broad snapshot/cache is a separate control only in the cached-zero isolation test and must never be the gated authority in generation, unmount, partial-failure, or positive-settlement tests. Assert no use of `workers.wait_for_complete()` and no timing sleeps.

- [ ] **Step 2: Initialize one screen-owned lifecycle.**

  In `LibraryScreen.__init__`:

  - capture `bool(app_instance.library_new_profile_admission)` once from the app-owned process-session new-profile admission fact;
  - load `library.rail_state.lifecycle` from in-memory config first, then the existing CLI fallback;
  - coerce into `_library_lifecycle` without disturbing section preferences;
  - initialize exactly one monotonic `_library_onboarding_generation` as evidence apply authority;
  - initialize `_library_onboarding_all_empty = False` (negative evidence is unavailable until one fresh all-owner settlement);
  - initialize explicit `LibraryEvidenceStatus.LOADING`, where the minimal status enum is `loading`, `settled`, or `partial_failure`;
  - if a genuinely new profile has no stored lifecycle, queue persistence of `unknown` on mount before evidence can fail, so a restart cannot reclassify it as a legacy Expanded profile.

  Do not inspect or mutate `first_run.setup_completed`.

- [ ] **Step 3: Gather only source-owned evidence under one generation.**

  Keep `_list_local_source_snapshot()` and `LibrarySourceSnapshot` completely outside onboarding. Add `_refresh_library_onboarding_evidence()` as one source-specific screen worker. Each invocation of the existing broad refresh choke point starts one independent evidence generation at entry (initial mount and successful mutation refreshes already converge there), but broad success/failure, records, and counts are never parameters to or authority for the evidence result.

  At request start:

  1. increment/capture `_library_onboarding_generation`;
  2. immediately clear `_library_onboarding_all_empty` so stale negative evidence cannot admit Back to Get started during a mutation/read;
  3. set evidence status to `loading` and sync only lifecycle status copy;
  4. concurrently call the six `get_library_user_content_evidence` owners through the same currently active local/server authority used by each Library source;
  5. consume completions progressively under one overall `LIBRARY_ONBOARDING_EVIDENCE_TIMEOUT_SECONDS` bound (reuse the existing source-snapshot timeout value unless a RED test proves a distinct bound is needed);
  6. apply `graduated` as soon as any accepted completion is `HAS_USER_CONTENT`, then cancel/ignore remaining tasks under the same generation;
  7. when no positive settles, wait only to the overall deadline, convert failures/timeouts/missing results to `UNKNOWN`, and finish as all-empty or partial-failure.

  At apply, require only the same profile/Library-screen admission, screen lifecycle, and onboarding generation. Do **not** require lifecycle equality: Explore may change `starter`/`unknown` to `expanded` while the read is pending, and a later positive must still graduate the current lifecycle. Resolve from the lifecycle value current at apply time:

  - any positive: `graduated`, status `settled`, all-empty false;
  - all six empty: `starter` only if current is `unknown`/`starter`; preserve `expanded`; status `settled`, all-empty true;
  - otherwise: preserve current lifecycle, status `partial_failure`, all-empty false.

  Late/inactive generations are silent. Cached snapshot application never starts or applies evidence.

- [ ] **Step 4: Persist through the existing config owner.**

  Add one serialized write-behind sibling of `_save_library_rail_preferences` that coalesces to the latest lifecycle and never allows concurrent Explore/Graduated writes to finish out of order. It writes:

  ```python
  save_setting_to_cli_config("library.rail_state", "lifecycle", lifecycle.value)
  ```

  Mirror accepted transitions into `app_config["library"]["rail_state"]` before dispatching the worker so recomposes/re-entry see one value. Do not rewrite `sections` during lifecycle persistence. On failure, keep current session behavior and set visible text: `Library view is updated for this session, but the choice may not be remembered.` A later transition/retry may persist again; persistence failure never blocks Import, New note, Explore, or navigation. Test a rapid Explore-then-Graduate sequence and prove the last stored value is `graduated`.

- [ ] **Step 5: Fence unmount before its first await.**

  Increment/revoke onboarding authority at the start of `on_unmount`, beside the existing Conversation/Prompt/Media invalidations and before `await workspace.shutdown()`.

- [ ] **Step 6: Trigger evidence after successful production mutations without adding mutation hooks.**

  Reuse the existing `_refresh_local_source_snapshot()` completion choke points already reached by successful Note creation, Media import, Conversation save, Prompt/Skill changes, and Collection changes to start one new evidence generation. Add only a bounded call at a true omission proven by a RED test. Do not derive evidence there or add per-source lifecycle transition code.

- [ ] **Step 7: Run focused authority tests and inverses.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_shell.py -k 'library and (starter or onboarding or first_run or lifecycle or snapshot)'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Library/test_library_snapshot_cache.py
  ```

  Inverses, one at a time with immediate restoration:

  - let cache application settle Starter: cached-zero test must fail;
  - remove the generation/unmount check: gated late-result test must fail;
  - require lifecycle equality at apply: Explore-during-read graduation test must fail;
  - leave `_library_onboarding_all_empty` true at generation start: Back-admission test must fail;
  - replace progressive bounded settlement with one unbounded gather: hanging-owner positive test must fail.

- [ ] **Step 8: Commit.**

  ```bash
  git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py
  git commit -m "feat(library): own starter lifecycle evidence"
  ```

## Task 4: Render the compact Starter rail and explicit disclosure

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_rail.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/Widgets/Library/test_library_rail.py`
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/UI/test_command_palette_providers.py`

- [ ] **Step 1: Add RED direct widget tests.**

  Add:

  ```python
  async def test_starter_rail_renders_only_import_note_and_explore(widget_pilot): ...
  async def test_unknown_rail_uses_same_safe_actions_without_empty_claim(widget_pilot): ...
  async def test_expanded_and_graduated_rails_render_the_full_shell(widget_pilot): ...
  async def test_starter_rail_tab_order_and_labels_are_text_complete(widget_pilot): ...
  ```

  Assert a constant three-action composition, no per-source section headers, no hidden duplicate Import row, and exact production IDs for Import/New note.

- [ ] **Step 2: Pass lifecycle through every rail construction/sync path.**

  Add `lifecycle: LibraryLifecycle` to `LibraryRail.__init__` and `sync_state`. Update the four real screen paths that construct or sync `#library-rail` (`compose_content`, canvas replacement, snapshot reconcile, and retained-owner repair). Keep an optional default only if a direct unrelated fixture needs compatibility; production paths must always pass the screen value.

- [ ] **Step 3: Compose Starter as a filtered presentation.**

  In `LibraryRail.compose`:

  - retain the Navigation heading and Collapse action;
  - for `unknown`/`starter`, omit search, section headers, Details, and full rows;
  - compose two `LibraryRailRowButton`s using the existing shell rows for `LIBRARY_ROW_INGEST_MEDIA` and `LIBRARY_ROW_CREATE_NOTE`;
  - add one plain `Button("Explore all tools", id="library-rail-explore-all")`;
  - for `expanded` with no usable content, add a quiet `#library-rail-back-to-starter` action below the full rail; omit it after graduation.

  Do not duplicate target routing or create alternate row IDs.

- [ ] **Step 4: Add screen handlers and persistence.**

  `Explore all tools` resolves to `expanded`, persists through the lifecycle owner, recomposes once, and restores focus to `#library-search-input` (or the first full rail row if search is unavailable). `Back to Get started` is admitted only when lifecycle is `expanded` and `_library_onboarding_all_empty` is true from a fresh accepted generation; it persists `starter` and focuses Import. Unknown evidence never exposes this backward transition.

- [ ] **Step 5: Prove deep-link and palette bypass.**

  Add mounted tests:

  ```python
  async def test_starter_deep_link_opens_hidden_collection_or_note_route(): ...
  async def test_starter_pending_nav_context_opens_ingest_without_explore(): ...
  async def test_palette_library_skills_command_opens_hidden_starter_route(): ...
  async def test_library_lifecycle_explore_persists_without_changing_sections(): ...
  ```

  Exercise the existing screen admission API/pending navigation context and `TabNavigationProvider.LIBRARY_SUBROUTE_COMMANDS` production command, not a test-only direct field mutation. The palette test executes the existing Library — Skills command and proves Starter rail filtering cannot block the resulting legacy-route context.

- [ ] **Step 6: Verify focused rail owners and inverse.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Widgets/Library/test_library_rail.py Tests/UI/test_library_shell.py -k 'library and (starter or onboarding or lifecycle or rail or deep_link or nav_context)'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_command_palette_providers.py -k 'library and starter'
  ```

  Temporarily gate deep-link dispatch on visible Starter rows; the hidden-route test must fail. Restore and rerun GREEN.

- [ ] **Step 7: Commit.**

  ```bash
  git add tldw_chatbook/Widgets/Library/library_rail.py tldw_chatbook/UI/Screens/library_screen.py Tests/Widgets/Library/test_library_rail.py Tests/UI/test_library_shell.py Tests/UI/test_command_palette_providers.py
  git commit -m "feat(library): add compact starter rail"
  ```

## Task 5: Make the landing lifecycle-aware and focus-safe

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_entry_canvases.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_entry_compose_once.py`
- Modify: `Tests/UI/test_library_shell.py`
- Conditional CSS only if mounted RED proves necessary: `tldw_chatbook/css/components/_agentic_terminal.tcss`, generated `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Add RED retained-owner state tests.**

  Extend `LibraryLandingCanvasState` with explicit lifecycle presentation fields rather than inferring from count strings. Add tests for:

  ```python
  async def test_library_landing_syncs_unknown_to_starter_without_duplicate_actions(): ...
  async def test_library_landing_syncs_starter_to_expanded_without_stale_recents(): ...
  async def test_library_landing_late_sync_cannot_replace_a_new_route_owner(): ...
  async def test_library_landing_partial_failure_shows_one_retry(): ...
  async def test_library_landing_persistence_warning_keeps_actions_enabled(): ...
  ```

- [ ] **Step 2: Define exact landing modes.**

  Keep one `LibraryLandingCanvas` and one state object:

  - `unknown`: Get started value statement; Import and New note; `Checking existing Library content…` or `Some Library sources are unavailable.`; no counts/recents/empty claim; on partial failure, render a contextual `#library-hub-retry-evidence` recovery button that starts one fresh guarded evidence generation;
  - `starter`: Get started, `1 Add  ·  2 Find  ·  3 Use`, Import and New note; no Search/counts/recents;
  - `expanded`/`graduated`: current purpose, counts, Import/Search/New note, and recents.

  Add `#library-hub-lifecycle-status` for text status. Exactly one Explore control may be composed: while the rail is visible, `#library-rail-explore-all` is the sole owner and the landing has no duplicate; when the compact/collapsed layout removes the rail action from the active focus tree, compose `#library-hub-explore-all` in the landing instead. Both reuse the same screen transition method. Retry is contextual recovery, not a fourth persistent Starter navigation action.

  `sync_state` may patch counts/recents while the mode is unchanged. When lifecycle mode changes and the widget set differs, use the retained owner's guarded recompose callback rather than attempting to query widgets that no longer exist.

  Evidence `loading` shows checking copy and no Retry. Evidence `partial_failure` shows the unavailable copy plus exactly one Retry. Preference persistence warning is an additional readable status line and never disables the production actions.

  Add wide/compact assertions that the active focus tree contains exactly one Explore action, owned by the rail when visible and by the landing only when the rail action is absent.

- [ ] **Step 3: Preserve semantic focus across settled transition.**

  When evidence changes unknown/starter to graduated after a production creation/import:

  1. preserve any live focused descendant or newly created item focus;
  2. announce `Library tools are now available.` in persistent visible text;
  3. recompose the rail/landing only after the authoritative mutation/list refresh settles;
  4. never move focus while the user is traversing an unrelated control.

  Add mounted gated tests that move focus after dispatch and prove the completion does not yank it.

- [ ] **Step 4: Verify compact and wide production hierarchy.**

  Add a parametrized mounted test using `TldwCli.CSS_PATH` at `(100, 30)` and `(170, 48)`. Assert:

  - Import, New note, the single active Explore owner, and lifecycle status have non-zero compositor regions;
  - Import and New note are above the fold at 100x30;
  - Tab reaches Import, New note, and the single Explore owner in visual order;
  - the full rail/landing returns after Explore;
  - no `Media (0)`, zero pager, empty recents container copy, or false source count appears while unknown/starter.

- [ ] **Step 5: Change CSS only if the production-hierarchy test is RED.**

  Prefer existing `.destination-workbench`, `.ds-toolbar`, and Library rail tokens. If CSS must change, edit only `_agentic_terminal.tcss`, regenerate, and verify:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_css_build_integrity.py -k library
  ```

  If the mounted test is already GREEN without CSS changes, do not touch CSS.

- [ ] **Step 6: Run focused landing owners and inverse.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_entry_compose_once.py Tests/UI/test_library_shell.py -k 'library and (starter or onboarding or lifecycle or landing or first_run or focus or geometry)'
  ```

  Temporarily let a late unknown completion focus Import unconditionally; the user-focus-veto test must fail. Restore and rerun GREEN.

- [ ] **Step 7: Commit.**

  ```bash
  git add tldw_chatbook/Widgets/Library/library_entry_canvases.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_entry_compose_once.py Tests/UI/test_library_shell.py
  # Add CSS source + generated bundle only if Task 5 Step 5 changed them.
  git commit -m "feat(library): add lifecycle-aware landing"
  ```

## Task 6: Touched-only verification, mounted UAT, docs, and closeout

**Files:**

- Modify: `Docs/User_Guide/library.md`
- Modify: `backlog/tasks/task-19022 - Add-Library-starter-rail-and-lifecycle-aware-landing.md`
- Modify lessons only if an actual generalizable incident occurs

- [ ] **Step 1: Run the exact pure/source/cache gate.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Library/test_library_content_evidence.py \
    Tests/Library/test_library_rail_state.py \
    Tests/Library/test_library_collections_service.py \
    Tests/Library/test_library_snapshot_cache.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Notes/test_notes_scope_service_library_canvas.py -k 'user_content_evidence or count_notes'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Media/test_media_reading_scope_service.py -k 'user_content_evidence or library_summary'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Chat/test_chat_conversation_scope_service.py -k 'user_content_evidence or list_conversations'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Library/test_library_prompts_seam.py Tests/Skills/test_skills_scope_service.py \
    -k 'user_content_evidence or count_prompts or get_context'
  ```

- [ ] **Step 2: Run the exact mounted/direct-owner gate in bounded partitions.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Widgets/Library/test_library_rail.py -k 'starter or lifecycle or rail'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_entry_compose_once.py -k 'library and (starter or lifecycle or landing)'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_shell.py -k 'library and (starter or onboarding or lifecycle or landing or first_run or deep_link or nav_context or snapshot or focus or geometry)'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_command_palette_providers.py -k 'library and starter'
  ```

  Do not run bare `pytest`, the whole `Tests/UI/test_library_shell.py` without its selector, or repository-wide coverage.

- [ ] **Step 3: Run static checks on the exact final changed Python list.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
    tldw_chatbook/app.py \
    tldw_chatbook/config.py \
    tldw_chatbook/Library/library_content_evidence.py \
    tldw_chatbook/Library/library_rail_state.py \
    tldw_chatbook/Library/library_collections_service.py \
    tldw_chatbook/Notes/notes_scope_service.py \
    tldw_chatbook/Notes/Notes_Library.py \
    tldw_chatbook/Notes/server_notes_workspace_service.py \
    tldw_chatbook/DB/Client_Media_DB_v2.py \
    tldw_chatbook/Media/local_media_reading_service.py \
    tldw_chatbook/Media/media_reading_scope_service.py \
    tldw_chatbook/Chat/chat_conversation_scope_service.py \
    tldw_chatbook/Prompt_Management/prompt_scope_service.py \
    tldw_chatbook/Skills_Interop/local_skills_service.py \
    tldw_chatbook/Skills_Interop/skills_scope_service.py \
    tldw_chatbook/UI/Screens/library_screen.py \
    tldw_chatbook/Widgets/Library/library_rail.py \
    tldw_chatbook/Widgets/Library/library_entry_canvases.py \
    Tests/Library/test_library_content_evidence.py \
    Tests/Library/test_library_rail_state.py \
    Tests/Library/test_library_collections_service.py \
    Tests/Notes/test_notes_scope_service_library_canvas.py \
    Tests/Notes/test_server_notes_workspace_service.py \
    Tests/Media/test_media_reading_scope_service.py \
    Tests/Chat/test_chat_conversation_scope_service.py \
    Tests/Library/test_library_prompts_seam.py \
    Tests/Skills/test_skills_scope_service.py \
    Tests/UI/test_library_shell.py \
    Tests/UI/test_library_entry_compose_once.py \
    Tests/Widgets/Library/test_library_rail.py \
    Tests/UI/test_command_palette_providers.py
  git diff --check 6a2e7fa50
  git diff --check
  ```

  Run exact format checking on every new or already-conforming touched owner:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
    tldw_chatbook/Library/library_content_evidence.py \
    tldw_chatbook/Library/library_rail_state.py \
    tldw_chatbook/Media/media_reading_scope_service.py \
    tldw_chatbook/Prompt_Management/prompt_scope_service.py \
    tldw_chatbook/Widgets/Library/library_rail.py \
    Tests/Library/test_library_content_evidence.py \
    Tests/Library/test_library_rail_state.py \
    Tests/Media/test_media_reading_scope_service.py \
    Tests/Chat/test_chat_conversation_scope_service.py \
    Tests/Library/test_library_prompts_seam.py \
    Tests/Skills/test_skills_scope_service.py
  ```

  Explicit baseline exclusions from whole-file format claims: before this task, Ruff format already reports drift in `library_collections_service.py`, `notes_scope_service.py`, `chat_conversation_scope_service.py`, `skills_scope_service.py`, `library_screen.py`, `library_entry_canvases.py`, `test_library_collections_service.py`, `test_notes_scope_service_library_canvas.py`, `test_library_shell.py`, `test_library_entry_compose_once.py`, `test_library_rail.py`, and `test_command_palette_providers.py`. Do not bulk-format them. Ruff **check** and range/worktree `diff --check` still cover every changed file; self-review must confirm modified hunks introduce no formatting-only churn.

- [ ] **Step 4: Repeat the seven decisive mutation/inverse checks one at a time.**

  Record command, exact failing node, and restored GREEN for:

  1. cached zero incorrectly declaring Starter;
  2. removed generation/unmount guard;
  3. legacy absence incorrectly mapping to Unknown;
  4. hidden route incorrectly blocked by Starter composition;
  5. initial Unknown not persisted across partial-failure restart;
  6. preference-write failure disabling or reverting current-session disclosure;
  7. one hanging owner suppressing an already-settled positive.

- [ ] **Step 5: Complete the atomic task's bounded mounted UAT.**

  Using the production `TldwCli.CSS_PATH` mounted tests at 100x30 and 170x48, verify:

  - new empty profile shows Import, New note, Explore, and truthful checking/Starter copy;
  - Import and New note activate production canvases;
  - Escape/Back and disclosure transitions preserve semantic focus;
  - Explore shows the full Library and survives a fresh screen/app instance;
  - an active Note or imported Media record changes the profile permanently to graduated after settled authoritative refresh;
  - deleting that record does not return to Starter;
  - a deep link and the Library — Skills palette command reach hidden routes before Explore;
  - all three Starter actions are keyboard reachable and visible at both sizes;
  - a failed lifecycle write leaves the session usable and displays the not-remembered warning.

  Per the approved delivery boundary, do **not** run isolated-profile/tmux live UAT in this atomic task. One live UAT at both geometries is owned by Wave 1 closeout after all Wave 1 atomic work is complete.

- [ ] **Step 6: Update user documentation and task evidence.**

  Document:

  - what Get started shows;
  - how Explore and Back to Get started behave;
  - that adding usable content permanently graduates Library;
  - that deep links and command routes remain available;
  - exact touched-only test counts, inverses, and mounted compositor geometry evidence;
  - ADR-076 and ADR-067;
  - explicit deviation: `Per user direction, repository-wide pytest was not run; only modified/touched Library component and direct-owner gates are claimed.`

- [ ] **Step 7: Obtain final independent spec and quality/minimality reviews.**

  Review the full implementation range against the design, ADRs, task AC, focused tests, and mounted UAT evidence. Resolve every Critical/Important finding before closeout.

- [ ] **Step 8: Mark the task Done by direct task-file edit and commit docs.**

  Only after all ACs, Implementation Notes, tests, static checks, docs, ADR hygiene, reviews, and task-scoped mounted UAT evidence are complete:

  TASK-19022 is a five-digit task ID, which the installed Backlog CLI may parse
  incorrectly. Edit this task file's frontmatter `status` directly from
  `In Progress` to `Done`, then commit the docs:

  ```bash
  git add Docs/User_Guide/library.md 'backlog/tasks/task-19022 - Add-Library-starter-rail-and-lifecycle-aware-landing.md'
  git commit -m "docs(library): close starter lifecycle task"
  ```

## Expected final changed files

```text
backlog/decisions/076-library-lifecycle-progressive-disclosure.md
backlog/tasks/task-19022 - Add-Library-starter-rail-and-lifecycle-aware-landing.md
Docs/superpowers/plans/2026-08-20-library-starter-rail-landing.md
Docs/User_Guide/library.md
tldw_chatbook/Library/library_content_evidence.py
tldw_chatbook/Library/library_rail_state.py
tldw_chatbook/Library/library_collections_service.py
tldw_chatbook/app.py
tldw_chatbook/config.py
tldw_chatbook/DB/Client_Media_DB_v2.py
tldw_chatbook/Media/local_media_reading_service.py
tldw_chatbook/Notes/notes_scope_service.py
tldw_chatbook/Notes/Notes_Library.py
tldw_chatbook/Notes/server_notes_workspace_service.py
tldw_chatbook/Media/media_reading_scope_service.py
tldw_chatbook/Chat/chat_conversation_scope_service.py
tldw_chatbook/Prompt_Management/prompt_scope_service.py
tldw_chatbook/Skills_Interop/local_skills_service.py
tldw_chatbook/Skills_Interop/skills_scope_service.py
tldw_chatbook/UI/Screens/library_screen.py
tldw_chatbook/Widgets/Library/library_rail.py
tldw_chatbook/Widgets/Library/library_entry_canvases.py
Tests/Library/test_library_content_evidence.py
Tests/Library/test_library_rail_state.py
Tests/Library/test_library_collections_service.py
Tests/Notes/test_notes_scope_service_library_canvas.py
Tests/Notes/test_server_notes_workspace_service.py
Tests/Media/test_media_reading_scope_service.py
Tests/Chat/test_chat_conversation_scope_service.py
Tests/Library/test_library_prompts_seam.py
Tests/Skills/test_skills_scope_service.py
Tests/UI/test_library_shell.py
Tests/UI/test_library_entry_compose_once.py
Tests/Widgets/Library/test_library_rail.py
Tests/UI/test_command_palette_providers.py
```

Conditional only if mounted geometry proves a gap:

```text
tldw_chatbook/css/components/_agentic_terminal.tcss
tldw_chatbook/css/tldw_cli_modular.tcss
Tests/UI/test_css_build_integrity.py
```

No new dependency, schema migration, generic lifecycle framework, duplicate router, tutorial dataset, analytics store, or full-suite claim is part of this plan.

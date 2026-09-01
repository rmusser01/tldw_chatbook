# TASK-26042 — Console Workspace Files read-only inspector implementation plan

> **Execution:** Use subagent-driven development. Each numbered task gets one fresh implementer and one fresh independent task reviewer. Do not begin a later task until the preceding task is implemented, reviewed, and committed.

**Goal:** Ship a useful read-only Workspace Files inspector that opens from either Console workspace surface without activating or retargeting Console state, presents every local-folder binding explicitly, and supports bounded safe tree/filter/file viewing across the required terminal sizes.

**Architecture:** Add a filesystem-domain service under `Workspaces` that treats registry snapshots and raw path identities as addresses, revalidates every operation, and returns immutable typed outcomes. Add one Console-owned safe modal whose view state and bounded worker lanes are ephemeral. Existing workspace controls emit a typed request; `ChatScreen` and `ConsoleWorkspaceController` own single-visit admission and lifecycle wiring. Slice 1 contains no editing, saving, root leases, Git invocation, persistence, or File Notes ownership.

**Tech stack:** Python 3.11+, Textual 8.x, immutable dataclasses/enums, `asyncio.to_thread` / Textual workers, pytest with real temporary filesystems, production `TldwCli.CSS_PATH` geometry harnesses.

## ADR check

- **ADR required:** yes
- **ADR path:** `backlog/decisions/079-workspace-file-inspector-direct-user-authority-and-save-publication.md`
- **Reason:** This slice directly implements ADR-079's non-activating Console modal, direct-user read authority, revalidation, privacy, and bounded lifecycle boundaries. No new ADR is needed because the approved record already governs the long-lived application structure and authority model.

## Global constraints

- Opening, navigating, filtering, paging, dismissing, or quitting the inspector must not mutate the active workspace, active task/session/conversation, composer text/attachments, approvals, staged context, or workspace activation state.
- A mounted visit is pinned to one stable workspace ID. Repeated same-workspace requests focus it; a different-workspace request is blocked. No request retargets or stacks visits.
- Every operation re-fetches workspace and binding state and verifies unarchived ownership, local-folder kind, binding fingerprint, canonical containment, no-follow/link policy, expected target kind, and size limits. A rendered row is never authority.
- Raw path identity is never reconstructed from display text. C0/C1, ESC, newline, tab, bidi controls, markup tokens, and undecodable surrogate bytes have visible one-way escaped labels with markup disabled.
- Directory pages contain at most 200 entries and scan at most 10,000 immediate entries. Directories sort before files by Unicode casefold then exact name. The modal performs no recursive scan on open.
- Filter is a case-insensitive literal substring over selected-binding root-relative paths only, debounced 150 ms, visits at most 50,000 entries, returns at most 500 results, does not traverse VCS internals/generated caches/symlinked directories, and exposes idle/searching/partial/complete/truncated/cancelled/failed states.
- Files above 8 MiB are metadata-only. Safe UTF-8 files above 200,000 decoded characters through 8 MiB use revision-pinned pages of at most 100,000 decoded characters, never split a code point, keep only current plus adjacent pages, and refuse to combine changed revisions.
- List/read/filter lanes allow one active request and at most one coalesced latest request per lane. Generation checks reject stale results; graceful teardown cancels/joins owned work and leaves no modal resources.
- The modal subclasses `SafeModalDismissMixin` first, declares exact content selectors, restores its opener, and routes Back to Console, Escape, and safe backdrop through one dismissal path. Below 100 columns the modal is full-screen, so backdrop dismissal is unavailable by geometry while Back/Escape remain.
- The UI remains read-only. Do not add hidden Edit/Save hooks, Git subprocesses/decorations, database/sync ownership, persistent drafts/selection/filter state, agent-context injection, or file/path/content logging.
- Tests use real temporary filesystems for path/byte behavior and production-shaped Textual hosts for visible geometry. Assertions cover prohibited side effects, worker cleanup, compositor output, and stable Console fingerprints, not only widget existence.

---

## Task 1: Implement the revalidating read-only filesystem service

**Files:**
- Create: `tldw_chatbook/Workspaces/file_inspector.py`
- Modify: `tldw_chatbook/Workspaces/__init__.py`
- Create: `Tests/Workspaces/test_workspace_file_inspector.py`

### TDD sequence

1. Add failing service tests for immutable workspace/binding snapshots and fingerprints, including archived/deleted/retargeted/foreign/default/non-local binding outcomes. Assert every list/read/filter call observes current registry state rather than trusting the opening snapshot.
2. Add failing real-filesystem tests for component-aware canonical containment, `..`, absolute-path, symlinked file/directory, special-file, disappeared/replaced target, and version-control-internal rejection. Assert no operation follows a rendered label or escapes the selected canonical root.
3. Add failing tests for `safe_filesystem_text`: ordinary Unicode stays readable; markup characters remain literal; C0/C1, ESC, newline/tab, bidi controls, and surrogateescaped bytes render as visible escapes while raw components remain separate and round-trip for authority checks.
4. Add failing directory-page tests: directory-first deterministic ordering, 200-entry pages, 10,000-entry scan cap, opaque continuation identity tied to one binding/directory revision, explicit empty/partial/truncated/failed outcomes, dotfile visibility, and generated-cache/VCS exclusion flags.
5. Add failing filter tests: one selected binding only, 150 ms policy surfaced to the caller, literal case-insensitive path matching, 50,000 visited/500 result limits, partial progress, cancellation, truncation copy, only-excluded versus no-match outcomes, reveal-cache generation changes, and no symlink-directory traversal.
6. Add failing read tests for safe UTF-8 preview, UTF-8 BOM, invalid/binary/control-text classification, metadata-only above 8 MiB, paging above 200,000 decoded characters, at-most-100,000-character pages, UTF-8 boundary safety, sparse offsets/current-plus-adjacent cache contract, and revision mismatch on external replacement between pages.
7. Run the focused service test file and capture the expected RED failures before production implementation.
8. Implement typed enums/dataclasses, a narrow registry protocol, sanitized error codes, opening scope capture, per-operation scope revalidation, no-follow filesystem access helpers, safe display formatting, directory/filter bounds, stable revision identity, and incremental revision-pinned decoding. Keep synchronous filesystem work pure and suitable for off-loop execution; do not add UI or persistence.
9. Re-run `Tests/Workspaces/test_workspace_file_inspector.py`, Ruff on touched Python files, and `git diff --check`.
10. Commit this task only after self-review confirms no writes, no raw sensitive logging, and no File Notes/Git coupling.

## Task 2: Build the safe read-only modal and bounded operation lanes

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_workspace_files_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_console_workspace_files_modal.py`
- Modify: `Tests/UI/test_console_modal_dismissal.py`

### TDD sequence

1. Add the modal to the exact Console safe-modal contract/inventory with no exemption. Add failing interaction tests for Back, Escape, backdrop, inside-content click, exact one-shot dismissal, opener restoration, and resize without recreation.
2. Add failing state/render tests for explicit inspected/active identities, pinned `Inspector only · Console remains …`, viewing/access contract row, all captured bindings with access/availability, automatic one-binding selection, multi-binding selector focus, and unavailable selection without fallback.
3. Add failing tree tests for loading/partial/empty/cancelled/failed/truncated states, directory expansion/collapse and paging, filter debounce/Enter/Cancel/Clear behavior, pre-filter tree restoration, reveal-cache confirmation, and raw-identity selection that never parses labels.
4. Add failing viewer tests for metadata-only, safe escaped preview, paged character ranges with Previous/Next, revision-change Refresh recovery, adjacent-page cache bounds, and stale read results never flashing after rapid selection.
5. Add failing operation-lane tests using deterministic barriers: one active plus one latest list/read/filter request, coalescing, cancellation, generation invalidation on binding/filter/teardown, no late DOM publication, and no owned tasks after graceful dismissal.
6. Add failing responsive/focus tests for 80x24 full-screen compact+short, 100x30 near-full compact, 120x40 two-pane minimums, and 160x50 bounded tree/editor growth. Assert pinned rows/actions, fold indicator, logical focus remapping, path truncation/full identity availability, and no clipped controls via compositor/region checks.
7. Run the focused modal test file plus safe-modal contract nodes and record RED before production implementation.
8. Implement the modal with immutable presentation state, injected service/callbacks, separate compact/short layout classes, one selected binding/file, a staged compact flow, markup-disabled file-derived text, explicit status announcements, and bounded generation-tagged worker lanes. Do not query registry or perform blocking filesystem work on the event loop.
9. Register exact modal contract factories/launch metadata and update both canonical CSS bundles consistently.
10. Re-run focused modal/dismissal tests, touched-file Ruff, CSS mirror checks used by existing tests, and `git diff --check`; commit after self-review.

## Task 3: Wire both non-activating Console entry points and single-visit lifecycle

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py`
- Modify: `tldw_chatbook/Chat/console_workspace_actions.py`
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_action_menu.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py` only if a new late-bound dependency is required
- Modify: `Tests/UI/test_console_workspace_context_rail.py`
- Modify: `Tests/UI/test_console_workspace_action_row_geometry.py`
- Modify: `Tests/UI/test_console_button_routing.py`
- Create: `Tests/UI/test_console_workspace_files_integration.py`

### TDD sequence

1. Add failing rail tests for a dedicated **Show Files** row immediately after RAG Scope, visible/focusable blocked guidance for Default/no-local-folder states, unchanged Switch/New geometry, exact focus order, and a typed `WorkspaceFilesRequested(workspace_id)` message that never parses display text.
2. Add failing Workspaces-tree action-menu tests for a permanent **Show files** command, exact keyboard/click routing by stable workspace ID, and unchanged workspace activation/switcher controls.
3. Add failing production-shaped integration tests that fingerprint active workspace, task/session/conversation, composer text/attachments, approval state, staged context, and conversation selection before opening the active card and a non-active tree-menu entry; exercise navigation/dismissal and assert every fingerprint remains unchanged.
4. Add failing admission tests: one mounted visit; same-workspace request focuses it; another workspace is blocked with exact copy; stale request after binding disappearance opens the pinned empty recovery state; below-minimum geometry refuses open with exact copy; no activation or worker duplication occurs.
5. Add failing attention tests for generation-checked, privacy-minimized pending-approval count and generic blocked/failed/new-activity flags plus Back to Console. Assert no approval body/path/tool args/error details leak and no underlying action resolves in the modal.
6. Add failing graceful-quit and screen lifecycle tests for a clean read-only visit: dismissal/quit tears down lanes once, does not leave timers/workers/resources, and returns focus to opener or composer fallback after recomposition.
7. Run focused rail/geometry/routing/integration tests and capture RED before implementation.
8. Add the two controls and typed message. Extend display state only with stable identity/availability data needed for rendering; keep services and DOM out of display builders.
9. Add the controller-owned modal admission gate and service construction using the existing registry service. Route attention through a typed privacy-minimized snapshot and ensure same/different-workspace requests cannot retarget the mounted modal.
10. Re-run focused tests, relevant existing Console workspace/safe-modal suites, Ruff, CSS geometry checks, and `git diff --check`; commit after self-review.

## Task 4: Production-shaped and live scratch evidence, documentation, and task closure

**Files:**
- Modify tests only where evidence gaps are found
- Modify: `Docs/superpowers/specs/2026-08-31-workspace-files-inspector-design.md` only for implementation clarifications, never silent scope reduction
- Modify: `backlog/tasks/task-26042 - Console-Workspace-Files-read-only-inspector.md`
- Modify relevant `backlog/docs/lessons-*.md` only if a real generalizable incident occurred

### Verification sequence

1. Run all focused service, modal, entry, geometry, routing, integration, and existing Console safe-dismiss/workspace-context tests. Do not run the full repository suite without the user's explicit approval.
2. Run relevant Ruff/static checks and `git diff --check`.
3. Run production-shaped Textual evidence using the exact `TldwCli.CSS_PATH` stack at 80x24, 100x30, 120x40, and 160x50. Capture compositor/geometry assertions for both entry controls, pinned modal rows/actions, tree/viewer transitions, filter states, paging, and focus restoration.
4. Launch the real TUI with `TLDW_CONFIG_PATH` and all writable state redirected to a verified scratch profile/root. Through actual input, open the active-card entry and a non-active workspace's tree-menu entry; browse hostile names, filter, page a large file, resize through all four target geometries, dismiss by supported paths, trigger generic attention, and gracefully quit.
5. Compare before/after fingerprints for active Console state, approvals, profile files, database/sync state, logs, worker/process inventory, and temporary roots. Confirm there were no file writes, binding mutations, agent approvals, context injection, persistence of file/filter/path data, or leaked raw filesystem errors.
6. If live evidence exposes a defect, return to the owning task's TDD/fix/review loop rather than editing around the review gate.
7. Update the Backlog task: check ACs only with evidence, add concise Implementation Notes with commits/files/trade-offs/results, record ADR-079, complete DoD, and set Done through Backlog CLI.
8. Run a fresh whole-slice independent review against the task base and final head. Resolve every Critical/Important finding and document any accepted Minor finding before presenting integration options.

## Review and completion gates

- Every implementation task must have a committed report and independent spec+quality review before the next task begins.
- Reviewer findings are fixed by the original task implementer, then re-reviewed; the controller does not patch implementation findings directly.
- The final whole-slice reviewer checks cross-task contracts, security/privacy, accessibility, bounded lifecycle, and absence of hidden Slice 2/3 behavior.
- TASK-26042 is not Done until all eight ACs and all six DoD items have direct evidence. A green subset is not permission to check unsupported criteria.

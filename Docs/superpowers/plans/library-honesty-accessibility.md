# Plan: Library honesty + accessibility batch (tasks 4011, 4024, 4023)

Branch `fix/settings-appearance-crash` — **rename intent**: the branch was cut for task-4010, which turned out already fixed (closed on evidence in the first commit). It now carries this batch. Cut from dev `642567627`.

Source: the 2026-08-09 re-critique (22/40) and its follow-ups. Three plan tasks. Each owns its backlog file end-to-end (In Progress + Implementation Plan before code; ACs checked + Implementation Notes + Done after).

**Standing rule, earned three times this programme (LIB-03, half of LIB-09, task-4020's self-refutation, and now task-4010): re-verify every finding at HEAD before implementing it.** A finding whose premise has dissolved gets closed on evidence, not "fixed".

## Global Constraints

- **Python/tests**: `.venv/bin/python` from the repo root; pytest is the ONLY entry point that may import app modules. Targeted tests + `--collect-only -q` sanity. "no tests ran" = failed gate. **Foreground Bash only — NEVER end your turn waiting on background work** (four agents have died parked on background suites this programme; if a suite exceeds the tool timeout, split it per-file or per `-k` selector).
- **TDD**: failing test first, RED reproducing the OBSERVED EFFECT (not merely calling the function). For contrast/geometry claims use ANSI (`capture-pane -e`) or rendered-region assertions — **a colorless capture is not evidence about a colour mechanism** (task-4020's whole story).
- **CSS**: never edit the generated bundle; source tcss → `build_css.py` → `check_bundle_sync.py`. Widget `DEFAULT_CSS` needs no bundle step. Remember the tier rule: `App.CSS_PATH` rules outrank widget `DEFAULT_CSS` regardless of specificity — a fix proven only at the DEFAULT_CSS tier is not proven.
- **Recompose discipline**: any conditional a compose branch owns, the in-place updater must own too.
- **Bindings/footer**: multiple same-key bindings resolve via `check_action` in declaration order (broadest last); footer/F1 advertise only keys that work on the current surface; LibraryScreen's F1 and footer share `_library_footer_shortcuts_for_current_state()` — extend that seam, never fork it.
- **Live verification** per task: unique socket `haT<N>lib$RANDOM`, scratch `/tmp/haT<N>`, `users_name = "sdd_hat<N>"`, `[first_run] setup_started/completed = true`. Cold start ~12s; palette nav; clicks by CHARACTER index (python `str.find`), never byte offsets; `capture-pane -p -e` for styling proof. Cleanup: `C-q`, `kill-server`, `rm -rf ~/.local/share/tldw_cli/sdd_hat<N>`.
- **Git**: commit per task, message ends `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; `git add` SPECIFIC paths; never stash; never push. Always operate with the worktree path explicit.
- **Docs**: UI-visible changes update the matching `Docs/User_Guide/` page or its stamp (`Verified against dev @ 642567627`).
- **Known ambient**: `Tests/UI/test_library_shell.py` carries ~45 failures on dev (note-editor 60x20 geometry family) — A/B before blaming your change; do NOT fix them here.

## Task 1: Study-from-Home honesty + Settings nav-scroll (tasks 4011, 4024)

Read both backlog files (`ls backlog/tasks/ | grep -E "4011|4024"`); their ACs bind.

- **4011** — the Study screen hardcodes a `Library ▸ Study` breadcrumb (`UI/Screens/study_screen.py:156`) and `action_study_back_to_library` (`:1270`) unconditionally posts a Library nav-context, regardless of where the user came from. Reached from Home, both lie. Confirmed still reproducing at dev.
  Fix: make the breadcrumb and the back target reflect the ACTUAL origin. Look for an existing origin/nav-context seam before adding one (task-2854 built the Library→Study path on `LIBRARY_NAV_CONTEXT_MODE`; Home's entry is `open_home_flashcards_review()` → `open_study_screen(initial_section=...)`). If Home's path carries no origin today, thread the minimum — do not invent a general navigation-history system. Escape from a Home-origin Study must return to Home, and the breadcrumb must say so.
- **4024** — Settings-screen nav-scroll settle race filed by task-4020's implementer. Re-verify it reproduces before fixing; if it does not, close on evidence.

## Task 2: Accessibility + honesty half of task-4023

Read `backlog/tasks/task-4023 - *.md`; you own AC#1-#4 (leave #5-#7 to Task 3, and say so in the notes so the task isn't flipped Done early).

1. **Disabled state is colour-only, measured 1.08:1 / 1.45:1 / ~1.4-1.51:1 / 2.30:1** across Select-mode bulk buttons (invisible exactly when the user enters Select mode looking for them), Media `Select` when empty (click does nothing, says nothing), Export, and Collections' three buttons (2.30:1 *even when enabled*). This violates two stated product principles at once ("colour never sole carrier of meaning"; "disabled controls say why"). Fix: floor disabled contrast at 3:1, add a non-colour marker, and attach the reason to the control. **The product's non-colour vocabulary already exists** (`☐/☑`, `▸`, `┃…┃`, `(selected)`, `✓/○`) — extend it rather than inventing a new one. Prove ratios with ANSI measurement, before and after.
2. **The Notes canvas says "switch to Files" but the `Database | Files` strip does not render on first paint** — the fast rail-click path (`_replace_library_browse_canvas`) swaps only the inner canvas and never composes the strip (which lives in `compose_content`, gated on `canvas_kind == "notes"`); only a full recompose renders it. Named `task-3317` in an unmerged sibling branch's test docstrings — **check that branch for prior art and reconcile explicitly** (adopt/adapt/supersede), as task-4021 did with task-3315.
3. **Details DB sizes are computed once and never refreshed** — UI showed `Prompts 148.0KB / Media 476.0KB` while disk incl. sidecars was 180.0KB/508.0KB; a recompose with no disk change corrected both. The WAL-inclusive helper is correct and stale.
4. **F1 contradicts itself**: lists Escape 2-3× with conflicting labels (`- esc: focus rail` / `- escape: Back` / `- escape: Focus rail`), omits F6 on Search/RAG though the footer advertises it, says nothing about Collections on the Collections panel, and does not close on a second F1.

## Task 3: Interaction grammar, search, layout, copy (task-4023 AC#5-#7)

Read the same task file; you own AC#5-#7 and the close-out (verify ALL seven ACs against reality, consolidated notes, flip Done).

- **Grammar (the re-critique's stated score ceiling)**: four footer dialects across seven canvases (different separators AND different names for the same key — `F6 panes` vs `F6 next pane`, `/ Find` vs `/ focus search`); three active-state markers (`▸` prefix, `┃…┃` bars, `(selected)` text); three toolbar layouts (Media vertical, Notes 3×2 grid, Prompts/Skills single row); `▸` overloaded as both disclosure and silent value-cycler whose option set is undiscoverable. Converge on ONE vocabulary per concept. This is the batch's highest-value work and its largest diff — if it needs splitting, split it and say so rather than half-doing it.
- **Search**: results land ~30 rows below the fold behind the configuration panel, so `Run` leaves the visible half of the canvas pixel-identical; Enter in the rail search navigates and pre-fills but does not run; two search inputs hold different values and navigation silently overwrites one with the other; never-executed strings enter `Recent searches`.
- **Layout/copy**: Media list in a ~30-char column on a 170-col terminal with titles truncated at 17 chars; viewer gives a 33-line document a 7-row viewport while spending 2 lines on a `file://` temp path; landing canvas vanishes at ≤100 cols while the rail still says "pick a section on the left"; "opens staging canvas" ×3 in the nav; Collections' four stacked empty-state sentences; "Everything" excluding Prompts/Skills/Collections; `Type: plaintext` for a rendered `.md` with extensions stripped from titles.
- Escape still inert on Export, Collections, and the Study staging canvas; `Export…` from within Media navigates away with no return.

# Prompt resize focus — TASK-32602

Verified on `feat/component-pattern-library`, based on `e950dba427`, on
2026-09-15. This record covers the local worktree; integration into `dev`
remains pending.

## Result

Basic and Advanced text fields keep their widget identity, content and visible
focus through 170×48 → 80×24 → 170×48, including height-only changes to 170×24.
A newer focus move inside or outside the editor wins over pending resize work.
Resize neither reloads the Prompt source nor writes reader preferences.

The original Advanced regression failed with its field at y=37 below the
24-row terminal; widening then focused the Prompts rail row. Notes focus capture
does not recognize Prompt work fields and falls back to its rail identity.
Prompt routes now skip that Notes focus restoration after updating shared
presentation. The Prompt work pane reveals the current focused descendant
after its geometry settles. It does not restore an old focus target, and its
scroll runs immediately within that settled callback, avoiding a second deferred
scroll that could outlive newer focus.

Existing ADR-086 governs destination ownership and transient adaptive layout;
ADR-150 governs the unchanged token styling. No new ADR or visual token is needed.

## Automated evidence

[Verification results](verification.txt):

- **58 passed:** complete Prompt resize, reader, save-continuity and Library
  resize-gate files. Real SQLite, production CSS, both editor modes and themes,
  retained widgets, compositor-painted text, newer focus, source-read and
  preference-write spies.
- **6 passed:** neighboring Notes breakpoint round trips, newer-focus guard,
  unsafe editing, delete confirmation and load failure.
- **31 passed:** design-token, component-pattern and generated-CSS governance.

These are disjoint selections, totaling **95 passing tests**. No full repository
sweep was run. Pytest reported existing temporary-directory cleanup warnings;
governance also reported existing unknown-marker warnings.

The first combined run exposed a Prompt bulk-preview test querying a TextArea
before it mounted. Its readiness predicate now includes that read-only field.
An older Notes test failed both with this change and with HEAD's original resize
transition method: it called `focus()` on an already programmatically focused
preview and expected a new user-intent generation. It now activates Preview
with keyboard Enter. The original stale-focus and scroll assertions remain.

[Static checks](static.json) show no new Ruff diagnostics. The small production
file and new regression file are lint-clean and formatted; changed methods in
legacy files pass formatting. Existing full-file diagnostic counts are retained.

## Native evidence

[The native record](result.json) contains **16 passing real tmux resize cases**
through `TldwCli` and `LinuxDriver`, covering both modes, both themes, compact
crossings and height-only changes. Resizes came from tmux window changes, not
synthetic Pilot resize events. Each case checked live focus, painted text,
unchanged browse request token and unchanged config-file hash.

The run used a new disposable profile with all ten database paths,
`database.USER_DB_BASE_DIR`, `paths.data_dir`, and both XDG directories isolated.
It acquired the exclusive profile lock. [Persisted state](persistence.json)
confirms the saved Prompt body and version 1. No provider interaction was tested.

Selected screenshots were rendered and visually inspected:

| Mode and theme | Terminal | Evidence |
| --- | --- | --- |
| Basic, dark | 80×24 | [Focused message](basic-textual-dark-80x24.svg) |
| Basic, light | 80×24 | [Focused message](basic-textual-light-80x24.svg) |
| Advanced, light | 80×24 | [Focused block](advanced-textual-light-80x24.svg) |
| Advanced, dark | 170×48 | [Focused block after expansion](advanced-textual-dark-170x48.svg) |

The application returned from `app.run` after the normal Ctrl+Q key at 80×24.
The fresh shell receipt was **exit 0**, and the owned pane was back in `zsh`
before its session was closed. This establishes compact shutdown for this run;
it does not retrospectively qualify TASK-32603's earlier unverified process.

[Log review](native-log-review.json) retains the existing ChatScreen sidebar
initialization error, optional audio and terminal-capability warnings, and
unhandled worker-status warnings. All Prompt assertions and normal quit passed;
this record does not qualify a warning-free application startup.

`native_check.py` is the executed inspection script, formatted for storage. It
accepts an already isolated profile directory and an owned tmux socket/session.
The private run fixture remains in the ignored
`.superpowers/sdd/2026-09-15-prompt-resize/run-001` directory. Fresh exit and terminal
receipts were `/tmp/task-32602-native-run-001.{exit,ansi}`.

## Commands

```sh
.venv/bin/python -m pytest Tests/UI/test_library_prompt_resize_focus.py Tests/UI/test_library_prompts_reader.py Tests/UI/test_library_prompt_save_continuity.py Tests/UI/test_library_resize_focus_gates_t23025.py -q --tb=short
.venv/bin/python -m pytest Tests/UI/test_library_shell.py -k 'breakpoint_round_trip or user_focus_vetoes_stale_deferred_restore or unsafe_session_outranks_rail_focus or delete_confirmation_outranks_rail_focus or load_failure_outranks_rail_focus' -q --tb=short
.venv/bin/python -m pytest Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py Tests/UI/test_component_pattern_governance.py -q --tb=short
```

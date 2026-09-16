# bd6 UI-ready module census causal review

Finding: the sole ratchet failure in run34908706826/job104191168600 is1024 UI-ready modules against the approved1022 cap. The retained log explicitly identifies Notes.note_import_parsers and Widgets.select_values as the two additions. Import-time census passes669/686. This is a module-count regression, not evidence of a new latency timeout or the SQLite scope fix loading more backup modules. Source reviewed at bd6ba3b126; no source changes, tests, app boots or network requests were made.

## Origin and exact eager edges

Both edges originate upstream and were first brought onto this branch by d4e1d0dff8 (the earlier00240 Notes merge), rather than being newly authored by the latest SQLite change or the later2f97 merge. Both commits are ancestors of incoming2f97a42c9aa9cc737cf304f927e6d861824e1b3a:

- a28c88b834ffe9bdacfe07e1a3cff4ee96885cc9, `feat(notes-sync): synced notes keep the vault's folder tree and its frontmatter`, adds module-level imports of _split_frontmatter/_frontmatter_title/_frontmatter_keywords in Notes/notes_sync_runtime.py:28. All three are used only by _lifted_note_metadata:366. app.on_mount:15484 accesses notes_sync_runtime_owner, whose constructor imports the runtime module; the pinned UI-ready snapshot already includes notes_sync_runtime. The runtime import now also loads the parser module even for a fresh profile with no Obsidian source to parse. Parser uses elsewhere are deferred Library import flows; note_import_planner and note_import_receipts are absent from the pinned UI-ready snapshot.
- 1faf59d9a6ce21e81b924b358e3575c94f22947a, `fix(select): one guard for every Select value this app does not pick itself (task-32533)`, adds the shared Select helpers. chat_screen.py:669 imports Console.console_model_popover, already present in the pinned UI-ready snapshot. The popover's module-level helper import:47 therefore loads Widgets.select_values even before the popover is opened. Uses are solely compose:464 and _sync_controls_from_draft:885. Settings has another eager import:88, but Settings itself is absent from the pinned ready snapshot; that separate edge is not established as the fresh-Console census cause.

These upstream features are legitimate: preserve frontmatter title/tag handling and invalid Select-value safeguards. Their eager placement is avoidable; removing either feature or copying its implementation elsewhere would be unnecessary scope expansion.

## Smallest reviewable proposal (not implemented)

Two product files, imports only:

1. Remove the three parser imports from notes_sync_runtime module scope. Import those exact helpers inside _lifted_note_metadata, AFTER its existing `not obsidian or type(text) is not str` early return and immediately before _split_frontmatter. Non-Obsidian/fresh startup never needs them; actual Obsidian parsing still uses the exact shared functions. No parser body, bounds, producer lifetime, offload, watcher or binding behavior changes. The imported module declares parser constants/functions/classes and an RLock; it has no required startup registration side effect. The normal first-use import must still precede any parsing.
2. Remove select_values imports from console_model_popover module scope. Import select_value_or_blank in compose before first use; import assign_select_value in _sync_controls_from_draft after its existing `not self.is_mounted` return and before the guarded assignment. Keep both actual calls and all Select options/prevent/flag/fallback semantics unchanged. The shared module defines helper functions over Textual.Select; it has no startup registration requirement. Leave the Settings import untouched in this narrow correction because it is not the evidenced fresh-ready edge.

No cache, module unloading, sys.modules trick, budget increase, new feature, source authority change or broad deferral of Notes runtime is proposed. This is an integration-local correction for newly merged eager edges, not backup admission optimization. Because the edges are upstream feature code rather than backup product work, review/authorize this exact two-file proposal before implementation instead of silently broadening UAT remediation.

## Necessary verification if approved

Use the existing UI-ready census and import ratchet unchanged (1022/686), and preserve all absent-family/expected-module assertions. Expected source-level effect is removing these two modules from a fresh Console ready census; only an actual fresh census can establish the resulting count. Do not preemptively update its snapshot or caps.

Exercise existing behavior at deferred first use: Notes test_setup_review_under_obsidian_mode_skips_vault_folders_and_lifts_frontmatter plus executor test_create_note_lifts_frontmatter_title_and_tags_into_the_note; Console invalid-provider compose/update regressions in the existing Console settings tests, with Tests/UI/test_console_model_popover_registry_options.py as a small registry-parity check. Keep known fixed-profile fixture setup requirements when selecting merged Notes tests; do not weaken admission for legacy retargeted fixtures. Settings out-of-options tests remain relevant if touching Settings later, but that is not part of the minimal proposal.

Evidence: /private/tmp/uat-bd6-ui-latency-failure.log; current callers and git -S / first-parent -m history above; Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt. No claim that upstream dev's current census passes or fails was made, since no dev execution/log was supplied for this task.

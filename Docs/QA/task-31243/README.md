# TASK-31243 compositor evidence

These six SVGs were regenerated from the Task3 delivery worktree on the merged
Task2 base `2b4973971e5dcf101c5a6ddcc55aa082ff22f814`. They are Textual Pilot
compositor output, **not native-terminal or human walkthrough evidence**. The
final implementation commit contains the exact source, tests and CSS used.

| Capture | Cells | Exercised surface |
| --- | --- | --- |
| roleplay-preview-52x20.svg | 52×20 | One-pane saved preview, full action labels, painted transcript, pointer Back and list-focus restoration |
| roleplay-preview-120x50.svg | 120×50 | Wide saved preview and Back |
| roleplay-draft-52x20.svg | 52×20 | Aggregate draft dialog with Save and continue / Discard and continue / Stay |
| roleplay-draft-120x50.svg | 120×50 | Same dialog, wide |
| library-repair-stale-52x20.svg | 52×20 | Explicit repair confirmation followed by stale-CAS Refresh recovery |
| library-repair-stale-120x50.svg | 120×50 | Same recovery, wide |

The fixtures use synthetic conversation/card identities and scratch databases;
no provider request, user profile or user terminal was used. Draft dialog
captures prove presentation and choice dispatch; separate production-owner
tests cover aggregate save/discard/failure handling. Reuse tests cover exact
Console activation/rollback and Library repair/return on cached screens.

The initial compact preview capture exposed clipped Back copy, a collapsed-rail
fragment and a hidden transcript. The bounded correction gives narrow preview
navigation full-width rows, suppresses the independent test-chat toggle and
collapsed rails during saved preview, and removes its redundant inner border.
The corrected test asserts painted Back, Send, Library and transcript text,
one visible pane, the Back hit target, and actual pointer-driven return focus.
Wide layout and card-authoring surfaces retain their incumbent presentation.
Final controller inspection confirms full labels and visible transcript, but a
right-edge `In` fragment still paints beside Send at 52×20 (the inspector-rail
area, `#personas-inspector-rail-handle`; exact paint ownership unconfirmed).
That residual compact chrome is a known review concern, not a visual pass.

Reproduce the six captures and adjacent compact checks:

```sh
TASK_31243_QA_DIR=Docs/QA/task-31243 ../../.venv/bin/python -m pytest \
  Tests/UI/test_personas_workbench.py \
  Tests/UI/test_roleplay_character_conversation_browse.py \
  Tests/UI/test_library_character_repair.py \
  -k 'task_31243 or real_textual_pilot or conversation_actions_fit_production_css or conversation_action_hierarchy or 52_by_20 or draft_dialog' -q
```

Result: 9 passed, 421 deselected. Native/platform verification remains pending:
there was no dedicated empty terminal readiness response, so no real-terminal
walkthrough was attempted. The task remains In Progress with unsupported
completion claims open. The controller's local delivery report contains raw
logs, resource attribution, inherited static failures and commit provenance.

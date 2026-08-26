# TASK-22033 live verification evidence

The production-CSS Textual harness completed successfully against the real
Library screen and isolated Prompt databases on 2026-08-25. The driver imported
the package from the `library-prompts-reader` worktree and used the scratch
profile rooted at `/private/tmp/task22033-live-final.nakRS9`.

## Verified journeys

- `160x50`: Library, Prompts, and Work panes are open; Work receives the largest
  share (`78` columns).
- `120x35` and `100x30`: Library collapses and Prompts expands (`56` and `42`
  columns respectively) while Work remains usable.
- `80x24`: Library and Prompts collapse, Work expands to `70` columns, and both
  five-column restore grips remain focusable and correctly named.
- Basic is the default editor mode at every geometry.
- A Basic-mode edit preserved hidden structured Prompt block metadata, saved a
  new version, and appeared in Info/history with Local provenance.
- Blank-name validation retained the dirty draft and focused the owning Name
  input with actionable copy.
- Bulk mode kept the loaded Prompt visible as a labelled, disabled read-only
  preview.
- Opening Import from a clean editor replaced only Work content, hid the editor,
  and retained the permanent Items and Work widget identities.
- A browse failure retained recovery copy and Retry; Retry issued the same scope
  again and rendered the recovered row.
- A detail failure kept the previously loaded Prompt visible and read-only while
  truthfully labelling the newly selected Prompt; Retry atomically adopted the
  selection, unlocked the editor, and removed the recovery action.

`summary.json` is the machine-readable ledger. Each journey also has a `.json`
fact file, a plain rendered `.txt` capture, and a production-CSS `.svg`
screenshot. `task22033_live_matrix.py` is the reproducible isolated driver.

## Reproduction

Run the driver from the worktree with a fresh `TASK22033_SCRATCH_ROOT`, XDG
directories, `TLDW_CONFIG_PATH`, and `TASK22033_DATA_DIR`, plus
`TLDW_TEST_MODE=1`, `TLDW_DISABLE_MODEL_CATALOG_NETWORK=1`, and the worktree on
`PYTHONPATH`. The driver creates a private scratch config whose `paths.data_dir`
also stays under that root; it refuses to run when isolation variables are
missing or when config/data escape the scratch root. It does not replace
`HOME`.

The harness deliberately stubs unrelated Library source services. Their
unavailable study-backend debug traces do not affect the Prompt journeys and are
not treated as product evidence.

# Permission-summary Settings review — TASK-32762

This review covers the existing immediate-save mode/provider/model group in
Console Behavior. It does not qualify external model calls or the full approval
workflow. The broader feature review remains open; PR #2704 stays draft pending
the user's visual review and explicit merge approval.

ADR required: no. The repair implements existing ADR-090 (permission summaries),
ADR-150 (design language) and ADR-161 (component patterns).

## Changes

- One app-owned writer serializes and coalesces form snapshots across category
  changes and actual Settings removal/recreation. Returning to Off while an
  earlier enable write runs can no longer leave summaries enabled on disk.
- Failed edits remain visible with an inline result and keyboard Retry. Retry
  uses the same immediate-save field guidance; successful retry returns focus
  to Summary mode. Unchanged viewing does not write, and sparse updates retain
  unrelated section keys.
- Post-replacement cache failures report that distinction. A later successful
  config publication retires the cached form and warning. Its generation is
  captured inside the write transaction, so a concurrent later writer cannot
  be mistaken for the failed write's baseline.
- `config.py` now projects the saved `permission_summary` section into the
  runtime view already consumed by Settings and the approval resolver. The
  previous reader silently saw defaults despite a successful TOML write.
- The existing external-content disclosure, default Off, and consent semantics
  are preserved. Existing token-backed component classes supply the new result
  and Retry; no stylesheet or token change was needed.

## Automated evidence

`ui-final.txt`: **15 private-profile mounted cases pass**, covering all three
fields under a held write, category/screen departure, compact/wide retry, no-op
viewing, runtime consumer configuration, and cache recovery before/after receipt.
Original stale-write, missing-runtime-projection, cache recovery, publication
interleaving and Retry guidance failures are retained in the `red-*.txt` logs.

`related-final.txt`: **52 pass, six fail** across the existing summary service,
approval wiring, token/component governance, bundle sync and boot-byte checks.
The same six failures reproduce with both changed production owners loaded from
saved commit `5ac787ce6a` (`wiring-baseline.txt`, `baseline-overlay.json`): the
trigger stub does not start and bare controller fixtures lack newer decision
fields. These failures are not claimed fixed or counted as passes. In total,
**67 distinct targeted cases pass; six pre-existing wiring failures remain**.
No full suite was run.

All seven derived-artifact guards pass in `preflight.txt`; the diagnostic
inventory remains unchanged. New test/runner Ruff check and format pass. The
large legacy owners retain 69 and 110 unique inherited lint findings respectively,
with none introduced (`static-large.json`). Independent bounded review is clear
after the cache-publication interleaving repair. The final ID scan covers all
316 fetched refs and 29 worktrees; TASK-32762 belongs only to this worktree.

## Native evidence

Final run `/private/tmp/tldw-32762-native-003`, PID 31992, passes dark/light ×
190×55/80×24. `native-result.json` pins the exact final runner and source hashes.
The runner chooses private HOME/config/data before imports, validates the profile,
and runs real `TldwCli.app.run` through LinuxDriver with all rendering streams
attached to the owned tmux terminal.

Each cell makes the private config directory read-only, attempts an opt-in,
checks that the file/runtime remain Off, leaves and returns with the failed edit
preserved, restores directory permissions, and retries by keyboard. It then edits
provider/model, verifies persistence after category navigation, and restores Off.
The actual safety boundary refuses these writes with `PrivatePathError`: one
lock-phase and three pre-replacement errors. Those four expected errors are
recorded explicitly in `lifecycle.json`; no other error or faulthandler output
occurred. Native verification does not simulate a post-replacement cache fault;
the mounted tests cover that case through the real mutation callback.

Lifecycle evidence confirms app.run returned, exit 0, PID absent, primary instance
lock reacquired, owned terminal closed, normal stop logged, 11 healthy private
databases, zero conversations/messages and unchanged default-profile fingerprints.
Seven active database object identities remain unchanged during the journeys.
No approval or external-model request was issued. Twelve SVGs and matching native
terminal captures were visually inspected; manifests pin original and stored bytes.

| Theme / size | Disclosure | Failed save / Retry | Saved model |
| --- | --- | --- | --- |
| Dark 190×55 | [View](textual-dark-190x55-disclosure.svg) | [View](textual-dark-190x55-failed.svg) | [View](textual-dark-190x55-saved.svg) |
| Dark 80×24 | [View](textual-dark-80x24-disclosure.svg) | [View](textual-dark-80x24-failed.svg) | [View](textual-dark-80x24-saved.svg) |
| Light 190×55 | [View](textual-light-190x55-disclosure.svg) | [View](textual-light-190x55-failed.svg) | [View](textual-light-190x55-saved.svg) |
| Light 80×24 | [View](textual-light-80x24-disclosure.svg) | [View](textual-light-80x24-failed.svg) | [View](textual-light-80x24-saved.svg) |

The generic category banner describes staged Console controls; the bordered
Permission summaries group and its field guide identify immediate saving.
Broader model-thinking departure, exchange capture, budgets, backgrounds and
context/memory workflows remain in the completion ledger.

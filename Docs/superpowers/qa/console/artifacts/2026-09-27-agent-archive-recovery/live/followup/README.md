# Approved native follow-up qualification

Source commit: `cd068fadb3e0c37f9573ebeca81a2bf47604aa66`.
Scratch: `/private/tmp/archive-followup-live-0tuqm_8b`; native tmux160×48.
The parent [live report](../../../../2026-09-27-agent-archive-recovery-live.md#approved-follow-up-implementation-and-native-qualification) explains acceptance, controls and limits.

These are native terminal captures (`.txt` plus color-preserving `.ansi`), synthetic fixture/storage assertions, structural wire metadata and reproducible disposable launcher/driver files. No raw wire request/response payloads, normal-profile config contents, database files or embedding assets are included. `wire-metadata.jsonl` carries scalar schema/HTTP/finish/tool-name and retained page/row counts only. `storage-final.json` records synthetic source messages and run identities from copies of the fully stopped scratch stores; those copies included any available WAL/SHM. They were used because directly opening the stopped workspace DB read-only failed.

Sequence: `search-exact` → `search-order` → `followup-exact` → `restore-confirm` → `restore-settled` → `source-before-open` → `original-opened` → both `*-draft-after-open` → `original-before-quit`. Idle native Ctrl+Q exited immediately, so there is no fabricated quit-modal capture. The private tmux server was absent afterward and the model slot idle.

`tests-*.txt` are raw targeted phase logs, including intermediate failures. They do not claim a whole-file green rerun. The independent actual-source review is [retained separately](../followup-context-implementation-review.md).

The original RED log and three intermediate failed-phase logs retain pytest/loguru-generated trailing spaces byte-for-byte. Whole staged `git diff --check` reports those four raw artifacts; source and authored documentation whitespace checks pass with only those exact logs excluded. This is not a whole-branch whitespace-clean claim.

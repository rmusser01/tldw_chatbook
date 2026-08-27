---
id: TASK-23088
title: Fork Console chat from a selected message
status: Done
assignee:
  - '@codex'
created_date: '2026-08-26 21:00'
updated_date: '2026-08-27 14:54'
labels:
  - console
  - chat
  - ui
  - persistence
references:
  - Docs/superpowers/specs/2026-08-26-console-chat-fork-design.md
  - Docs/superpowers/plans/2026-08-26-console-chat-fork.md
  - backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user create and immediately open an independently owned Console chat copied through one selected stable message while leaving the source chat and all of its live and durable state unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Eligible selected USER and ASSISTANT messages expose Fork immediately before Regenerate plus the `f` action, and a compact naming dialog clearly identifies the boundary, saved or temporary destination, exclusions, validation, progress, cancellation, and degraded-success recovery.
- [x] #2 Confirming copies exactly the canonical active lineage through the selected boundary and its fenced visible text or generated-image choices with fresh mutable ownership, while off-path, later, display-only, unsettled, unsaved-durable, and unsupported state is rejected or excluded as designed and the source remains byte-for-byte and live-state unchanged.
- [x] #3 Durable forks commit conversation ancestry, messages, supported sidecars, active leaf, policy, governed citation owner links, and sanitized project context atomically before publication; temporary forks remain detached and sanitized, and a non-ephemeral source without durable IDs produces a saved independent-root fork without persisting the source.
- [x] #4 Forks preserve declarative Workspace, model, role, Library, RAG, and project-instruction selections without copying scratch, approvals, permissions, resolved instruction bodies, continuations, recovery, derived context, usage, tool activity, or ephemeral video authority; citation and media degradation remains truthful.
- [x] #5 One preallocated conversation or session identity makes retries idempotent, precommit failure creates nothing, and postcommit publication or activation failure identifies and reopens the already-created fork without duplication.
- [x] #6 The USER and ASSISTANT action row uses the approved stable direct actions and labelled `More…` menu with captured message targeting, safe teardown, and deterministic focus fallback at 80x24 and wider production-shaped layouts.
- [x] #7 Targeted domain, real-SQLite persistence, authority, media, cancellation-race, action/menu, modal, reload, and live local TUI verification pass, and Console user documentation describes the boundary, temporary behavior, shortcut, exclusions, and video/citation caveats.
<!-- AC:END -->

## Implementation Plan

ADR required: yes

ADR path: `backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md`

Reason: ADR-092 already governs the durable copy, identity, authority, and publication boundaries; this task implements that accepted contract without a schema migration or new ADR.

1. Define the pure allowlisted fork projection, title rules, eligibility, and sanitized project-instruction contract.
2. Fence and revalidate the canonical active-lineage prefix plus selected generated-image state, then stage fresh independent ownership without source mutation.
3. Add one idempotent real-SQLite bundle for ancestry, messages, supported sidecars, policy, governed citations, project context, and active leaf.
4. Add the direct Fork action, captured-target More menu, media-card controls, compact six-state modal, and cancellable controller orchestration.
5. Verify atomic failure, cancellation races, reload, layout/focus, temporary promotion, source immutability, and the provider-free live journey; update user docs and task evidence.

Detailed TDD steps and commands: `Docs/superpowers/plans/2026-08-26-console-chat-fork.md`.

## Implementation Notes

Implemented the ADR-092 fork projection and durable bundle, Console orchestration,
stable direct action row and captured-target `More…` menu, six-state naming modal,
temporary-fork promotion, citation/media handling, and fresh authority ownership. The
copy contract is an explicit allowlist: supported declarative settings and sidecars are
copied into fresh owners while live execution and authority state remain source-only.
No schema migration or new dependency was introduced.

### Production-shaped integration evidence

`Tests/integration/test_console_chat_fork_flow.py` uses a pytest-owned HOME/XDG profile,
file SQLite, a named Workspace and bound project, the production Console screen/app
hierarchy and stylesheet, the real citation repository, the real unified MCP permission
store/effective resolver (including the local-tool hub), controller approval/run owners,
durable and ephemeral recovery, and real scratch snapshots/leases. It seeds an active
branch plus an off-path sibling, citations and textual `[S1]`, attachments, two generated
image variants and selects the non-default green variant, video state, Library/RAG and
project-instruction policy. The selected default name is replaced by keyboard input in
the mounted modal; no `Input.value` assignment is used.

The test forks middle USER and ASSISTANT boundaries, a temporary citation/video source,
and a non-ephemeral source without durable IDs. It proves exact prefixes and ancestry,
selected-image inclusion and unselected-image exclusion, atomic sidecars/policies,
source durable and expanded live-state equality, temporary-source equality before and
after promotion, independent-root promotion with sanitized project controls, and
canonical reload of the source, both durable forks, and the promoted temporary fork.
Snapshots, notices, and captured logs are checked for the absence of approval arguments,
permission grants, recovery/run state, resolved instruction bodies, scratch paths,
attachment bytes, video paths, and provider secrets. No provider is called.

Focused correction and integration commands:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_conversation_hydration.py \
  Tests/Chat/test_console_chat_fork.py -q --tb=short
# 255 passed, 2 warnings in 29.99s

PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/integration/test_console_chat_fork_flow.py -q --tb=short
# 1 passed, 2 warnings in 14.58s
```

The exact Task 7 regression command was:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/integration/test_console_chat_fork_flow.py \
  Tests/Chat/test_console_chat_fork.py \
  Tests/Chat/test_console_chat_fork_persistence.py \
  Tests/Chat/test_console_message_actions.py \
  Tests/Chat/test_console_edit_message_modal.py \
  Tests/Chat/test_console_regenerate_branching.py \
  Tests/UI/test_console_fork_chat_modal.py \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_message_controller.py \
  Tests/UI/test_console_session_controller.py -q
```

Result: `663 passed, 2 warnings in 199.41s`. The two reported warnings are the existing
requests dependency-version warning and Python's `audioop` deprecation through pydub;
pytest also emitted existing macOS temporary-cleanup warnings after the passing summary.
No full suite was run.

### Genuine isolated PTY evidence

The live fixture root was `/tmp/console-fork-live.HLlDNa`. Its `config/config.toml`
disabled splash and model-catalog refresh, selected Console and an offline fixture model,
and pointed every application database at `data/` under that root. Fresh `home/`,
`config/`, `data/`, `cache/`, and `state/` directories isolated the run from developer
data. `seed.py` used production `get_chachanotes_db_lazy`, `ChatPersistenceService`, and
`ConsoleChatStore` to create source conversation
`7a9d771e-253c-4ad5-9f77-88060a7c86ce` with persisted messages
`3c970922…`, `96acdb6c…`, `b8c4eb27…`, `0d5c1e93…`, and `ae95ed54…`.

The exact real-app launch shape was (first `-x 120 -y 35`, then after restart
`-x 80 -y 24`):

```bash
env TMUX_TMPDIR=/tmp/console-fork-live.HLlDNa/tmux \
  tmux -L fork23088-live new-session -d -x 120 -y 35 -s app \
  'env HOME=/tmp/console-fork-live.HLlDNa/home \
  XDG_CONFIG_HOME=/tmp/console-fork-live.HLlDNa/config \
  XDG_DATA_HOME=/tmp/console-fork-live.HLlDNa/data \
  XDG_CACHE_HOME=/tmp/console-fork-live.HLlDNa/cache \
  XDG_STATE_HOME=/tmp/console-fork-live.HLlDNa/state \
  TLDW_CONFIG_PATH=/tmp/console-fork-live.HLlDNa/config/config.toml \
  TLDW_TEST_MODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
  ALL_PROXY=http://127.0.0.1:9 NO_PROXY="" \
  OPENAI_API_KEY=OFFLINE-NONSECRET-FIXTURE \
  PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/console-chat-fork \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  -m tldw_chatbook.app'
```

All interaction was sent to the real tmux PTY and evidence was read with
`tmux capture-pane -p -t app:0.0`. At `120x35`, `Ctrl+K`, `Tab`, `Down`, `Enter`
opened the source; `Esc`, `F6`, `F6`, `k`, `k`, `k` selected the middle USER row;
`f` opened a painted `Through User 3` dialog; typing
`Live renamed TASK-23088 fork` replaced the selected default; and Enter opened a
three-message fork. Switching back showed `Live later excluded` still present in the
source. The window was resized with `tmux resize-window -x 80 -y 24`; `Esc`, `F6`,
`F6`, `k`, `k`, `f`, `Enter` exercised the quick-accept path at `Through Assistant 4`
and opened the four-message default-titled fork. After `Ctrl+Q`, a fresh 80x24 process
reopened the default fork, renamed fork, and original source through the real session
switcher; captures showed the exact boundary prefixes and intact later source message.
No composer send occurred. Offline catalog settings plus closed-loopback proxy variables
prevented external access, and no provider was contacted.

`assert_db.py` read the fixture SQLite directly after restart. It proved the source
kept all five original message IDs and contents; the renamed fork had a fresh three-row
chain, `parent_conversation_id`/`root_id` equal to the source and
`forked_from_message_id=b8c4eb27…`; the quick fork had a fresh four-row chain and
`forked_from_message_id=0d5c1e93…`. The source row retained null parent/fork fields.

### Live-discovered correction, static checks, and review

The first genuine app run exposed a production resume defect: the ordinary persisted
identity `assistant_kind='generic', assistant_id='console'` was normalized by
conversation metadata to kind `None` while retaining `assistant_id='console'`, so strict
fork validation correctly disabled the action as an inconsistent unscoped identity.
The new hydration regression first failed on that retained ID, then passed after
`hydrate_console_session` canonically cleared assistant ID, authority, and persona memory
when the kind is unscoped. Persona identity/memory round-trip assertions confirm scoped
identity remains intact. Fork validation was not weakened.

Changed-file Ruff check and format check plus `git diff --check` passed. Ruff formatting
of `tldw_chatbook/UI/Console_Modules/session.py` remained a three-line mechanical layout
change in the Task 6 fork region; the integration formatter was also mechanical.
Self-review against ADR-092 found no source mutation, shared mutable owner, authority
transfer, non-atomic durable publication, duplicate retry, or narrow-layout regression.
User docs now describe Fork, the `More…` two-step Delete path, and Edit & resend as a
response branch in the same chat.

ADR required: yes

ADR path: `backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md`

Reason: ADR-092 is the governing copy, identity, authority, atomicity, and publication
contract implemented by this task; no new decision was introduced. No lessons entry
was added: the live-found mixed-identity incident is already covered by the repository's
existing lesson that a safety-boundary normalizer must be proven canonical rather than
merely plausible, so duplicating that lesson would add no new guidance.

Immediately before closeout, the uniqueness sweep checked 900 local/remote refs and
238 worktrees. It found one committed-ref occurrence and two worktree occurrences of
`id: TASK-23088` or the exact task title: the branch copy, this worktree, and the
task-scoped baseline at `/private/tmp/tldw-chat-fork-task3-baseline`, all at the same
canonical task path. There were zero competing ID claimants and zero same-title tasks at
another path. The five-digit task was marked Done by direct file edit because the
Backlog CLI cannot reliably address it.

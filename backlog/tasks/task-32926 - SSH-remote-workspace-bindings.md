---
id: TASK-32926
title: SSH remote workspace bindings
status: Done
assignee:
  - '@robert'
created_date: '2026-09-24 22:35'
updated_date: '2026-09-25 09:55'
labels:
  - console
  - workspaces
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Full fs parity on ssh-filesystem bindings
- [x] #2 Availability guarantee both directions
- [x] #3 Nothing left behind on server
- [x] #4 No credentials stored
<!-- AC:END -->

## Implementation Notes

Branch `feat/ssh-remote-workspace-bindings` (36+ commits off `origin/dev`);
ADR-181 (`backlog/decisions/181-ssh-remote-workspace-bindings.md`) records the
decisions; spec revision 5 is the binding design.

- **Approach**: `ssh-filesystem` runtime binding kind executing `fs_*` ops
  server-side via a transient stdlib-only worker bundle piped over SSH per
  call (Phase 0 split the wire decode so the worker ships without pydantic;
  parent-side acceptance contract unchanged, pinned by a conformance corpus).
  Availability is an in-memory status cache bucketed by the protocol's
  `admitted` marker, with debounced automatic recovery probes; run
  composition reads cache only (hot-path rule).
- **Six phases** (21 tasks, each with its own review + fix rounds):
  wire-decoder split → binding foundation (locator/argv discipline,
  `ssh -G` canonicalization, registry) → transport (explicit ControlMaster
  lifecycle, admitted-marker taxonomy, two-tier watchdog Timer+`signal.alarm`
  exit 75, version-gated bootstrap exit 76) → server-side authority
  relocation (`LocalRoot|RemoteRoot` type boundary; CAS stamps from worker
  responses; worker-side exclusions + remote-home denylist) → run
  composition + remote AGENTS.md (incl. end-to-end lazy nested activation) →
  Settings/Console UI, docs, availability check, CI 3.10 floor.
- **Live UAT (2026-09-25): 23/23 passed** against a real user-level sshd —
  both availability directions, recreated-root STALE→re-capture, server-down
  BLOCKED→probe recovery, dead-mux restart, version gate on real python 3.9.6,
  verified worker-reported stamps, no stray files. TUI smoke: boots and
  renders under scratch isolation; no crashes.
- **UAT found and fixed two real defects** (both re-reviewed clean):
  ControlPath sun_path overflow on long data dirs (short-dir fallback,
  `742ccb1d81`) and missing shell-quoting of the remote bootstrap (ssh
  flattens remote commands through the login shell; fakes now emulate the
  flattening, `120d6ee339`).
- **Known deferrals** (ledgered, triaged by the final whole-branch review):
  denylist symlink-aliasing (pre-existing lexical class), `.root`-consumer
  lint gate for the supertype, sync `ssh -G` on the Settings UI thread,
  rc-noise live check (byte-level strip covered by fake-ssh), git_*/change
  review on remote bindings are v1 non-goals.

---
id: TASK-583
title: OS-level sandbox for skill script execution
status: To Do
assignee: []
created_date: '2026-07-25 15:05'
labels:
  - skills
  - security
  - sandbox
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Skill script execution currently runs under a best-effort sandbox: no shell, a scrubbed environment (no API keys), a fresh scratch working directory, RLIMITs applied via a Python trampoline, a wall-clock deadline with process-group kill, and bounded output. That is deliberately not a jail, and the feature documentation says so.

Three residuals follow directly from the absence of real OS-level containment:

- **network egress is not blocked** — a script may open sockets;
- **reads and writes outside the scratch directory are possible** — only the working directory is scratch, and nothing stops an absolute path;
- **memory is uncapped on macOS/BSD** — `RLIMIT_AS` cannot be lowered there, so peak memory is bounded only by CPU and wall-clock limits.

The compensating control today is the human confirm card plus per-run trust re-verification. Real containment would let those residuals be closed rather than documented, and would make a standing "always allow" grant a much safer thing to give.

This is a design project, not a patch: the mechanism differs per platform (macOS `sandbox-exec` — Apple-deprecated but widely used; Linux seccomp/namespaces or bubblewrap; containers add setup and latency cost that may be wrong for fast local helpers). Script execution is already POSIX-only, so Windows is out of scope here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A containment mechanism is chosen per supported platform, with the trade-offs (deprecation risk, setup cost, latency, failure modes) written down
- [ ] #2 A script cannot write outside its scratch directory, verified by a test that attempts an absolute-path write
- [ ] #3 A decision is recorded on network egress — blocked by default, or opt-in per skill — and enforced if blocked
- [ ] #4 Memory is genuinely capped on macOS, or the platform's limitation is re-documented as irreducible with evidence
- [ ] #5 A script that the sandbox refuses to launch fails closed with a clear user-facing reason, never a partial spawn
- [ ] #6 The residual-risk section of Docs/Features/Skills-Script-Execution.md is rewritten to match what is actually enforced
- [ ] #7 Existing skill scripts that only read their own bundle and write to the scratch directory continue to work unchanged
<!-- AC:END -->

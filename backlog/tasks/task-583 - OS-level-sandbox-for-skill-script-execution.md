---
id: TASK-583
title: OS-level sandbox for skill script execution
status: To Do
assignee: []
created_date: '2026-07-25 15:05'
updated_date: '2026-07-25 23:25'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PAUSED by user 2026-07-25, before any implementation. Recording the feasibility prototype so it is not re-derived.

FEASIBILITY: PROVEN on macOS 15.6. sandbox-exec exists at /usr/bin/sandbox-exec and a deny-default profile genuinely blocks both network egress and writes outside an allowed subpath. Full-stack prototype composed cleanly with the EXISTING runner: RLIMITs from our Python trampoline still applied (verified CPU 7 / NOFILE 64 inside the sandbox), network blocked, scratch write allowed, write outside blocked, and start_new_session still gave the child its own process group (so killpg teardown is unaffected). Exit 0.

TWO CONSTRAINTS LEARNED THE HARD WAY:
1. subpath rules MUST use the RESOLVED path. mktemp -d yields /var/folders/... but the sandbox matches /private/var/folders/...; the unresolved form silently denies writes to the very directory you meant to allow.
2. A malformed profile makes sandbox-exec fail LOUDLY (it errored on a deliberate typo) rather than silently running unsandboxed — the failure direction you want from a containment layer.

USER DECISIONS TAKEN DURING THE BRAINSTORM (carry these into the eventual spec):
- Network: blocked by default with a per-skill opt-in.
- The opt-in is a DECLARE/GRANT PAIR (user answered '1+2'): the skill declares  in SKILL.md frontmatter (so the request rides in reviewed, fingerprinted content — adding it later changes the digest, re-triggering review and dropping any standing script grant) AND the user grants it via a per-skill toggle in the Library trust panel. Network is permitted only when BOTH hold, so a skill cannot self-grant and a user cannot accidentally grant to a skill that never asked.
- Fail mode: refuse to run when the sandbox cannot be applied, with a config knob permitting degraded best-effort execution for someone who knowingly wants it.

RISKS TO ACCEPT KNOWINGLY: sandbox-exec is Apple-deprecated, so a future macOS could remove it and the fail-closed default would then turn script execution off until a replacement lands. Linux needs a different mechanism entirely (seccomp/namespaces/bubblewrap), so this ships macOS-first with Linux refusing-unless-opted-in.

SCOPE NOTE: this is a multi-task layer (frontmatter key, second grant dimension, UI toggle, profile generation, fail-mode config), not a single PR.
<!-- SECTION:NOTES:END -->

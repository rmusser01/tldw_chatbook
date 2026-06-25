# ADR-009: Local Skill Trust Boundary

Status: Accepted
Date: 2026-06-25
Related Task: [backlog/tasks/task-137 - Add-local-skill-trust-integrity-controls.md](../tasks/task-137%20-%20Add-local-skill-trust-integrity-controls.md)
Supersedes: N/A

## Decision

Chatbook-managed local skills will use a passphrase-rooted, authenticated local trust boundary with encrypted trusted snapshots, a secure generation marker, logical quarantine, and reviewed re-trust before trust-blocked skills can be staged or executed.

## Context

Local skills are prompt assets that can steer model behavior and future tool use. They currently have metadata validation and Chatbook-owned storage, but no durable way to prove that an on-disk `SKILL.md` or supporting file matches the version a user previously trusted.

The user concern is offline tampering: Chatbook or a server process is stopped, local skill files are modified, and the next launch unknowingly stages or executes a changed skill. The target scope is Chatbook-managed local skills only. Codex runtime skill folders, server-managed skills, and skill sync remain outside this decision.

The trust root must not live only beside the skill files it protects. A passphrase-derived key provides the main trust root. Secure keyring convenience may cache a protected or wrapped trust root, but it must never store the user's passphrase and must be labeled lower assurance than passphrase-per-start mode.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Hash-only manifest stored next to skills | An attacker who can edit skills can also replace the hash manifest, so this only detects accidental drift. |
| Warn-and-allow changed skills | This preserves convenience but leaves a straightforward prompt-backdoor path when users click through warnings. |
| Physical quarantine by moving files | Moving directories can destroy forensic context, surprise external editors, and create recovery edge cases. Logical quarantine keeps evidence in place. |
| Keyring-only auto-unlock | Keyring convenience is useful, but relying on it as the only trust root weakens the offline-tamper threat model when the user account or unlocked keyring is compromised. |
| Trust-on-first-use without explicit bootstrap | Silent bootstrap could bless already-compromised skill files. First enablement must be explicit. |
| Include Codex runtime skills and server skills in v1 | Those are separate ownership and runtime boundaries. Mixing them into the first local Chatbook trust boundary would obscure the service contract and recovery rules. |

## Consequences

Local skills gain explicit trust states separate from metadata validation. A skill can be metadata-valid but trust-blocked.

The local Skills service must verify trust at use time. Listing/detail paths may expose trust state for review, but context staging and execution must block untrusted, locked, quarantined, or globally unverifiable skills.

Trust records cover all files in a local skill directory. Trusted snapshots must be encrypted and authenticated so the review flow can show trusted-vs-current diffs without storing plaintext history.

Rollback protection requires a secure generation marker store outside the local skills directory, with secure OS keyring as the v1 target. If that marker store is unavailable, high-security mode fails closed. Reduced rollback protection requires explicit opt-in and visible posture reporting.

Lost-passphrase recovery is intentionally high-friction: users can re-bootstrap current files, but previous encrypted snapshots and authenticated audit history become unrecoverable.

This decision protects against offline file tampering when the attacker lacks the passphrase or authenticated trust root. It does not protect against malicious active app code, passphrase interception, active process-memory compromise, or a user approving a malicious diff.

## Links

- [TASK-137](../tasks/task-137%20-%20Add-local-skill-trust-integrity-controls.md)
- [Local skill trust integrity design](../../Docs/superpowers/specs/2026-06-25-local-skill-trust-integrity-design.md)
- [Local skill trust integrity implementation plan](../../Docs/superpowers/plans/2026-06-25-local-skill-trust-integrity.md)

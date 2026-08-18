# ADR-069: Project-Local `.SKILLS/` Folder Convention

Status: Accepted
Date: 2026-08-18
Related Task: [backlog/tasks/task-17651 - Project-skills-SKILLS-folder-discovery-and-prompt-driven-import.md](../tasks/task-17651%20-%20Project-skills-SKILLS-folder-discovery-and-prompt-driven-import.md)
Supersedes: N/A

## Decision

A project may declare a `.SKILLS/` (or `.skills/`) directory at its root. Chatbook
discovers it at two moments — app startup from inside the project, and workspace
creation with a bound folder — and offers a **prompt-driven, never-silent** import into
the existing local Skills store. Import always **copies** discovered skill content
through the existing importer with `trust_approved=False`, landing the skills inside
ADR-009's local trust boundary in the quarantined state; the folder itself is never
executed from, watched, or treated as a live skill source. A per-project prompt ledger
avoids re-nagging until the discovered skill set actually changes, and a config
kill-switch (`[skills] project_skills_prompt_enabled`) disables the feature outright.

## Context

Users bring existing projects into Chatbook — a repo they already work in, with its own
conventions and, increasingly, its own curated prompt/skill assets. Today the only way
to get those into the app is a manual, one-directory-at-a-time import through Library ▸
Skills. That is fine for a skill a user builds inside Chatbook, but it is friction for a
user who already has a folder of skills sitting in their project and just wants the app
to notice.

The obvious per-skill layout choice follows the emerging "agent skill" convention
already used elsewhere in this ecosystem: a per-skill subdirectory containing a
top-level `SKILL.md`, imported via the existing `import_skill_directory`. A project may
also have simpler loose `*.md` files not organized into subdirectories; those import via
`import_skill_file`, one skill per file. Anything else in the folder (a subdirectory
without `SKILL.md`, a non-markdown file) is reported as skipped with a reason rather
than silently dropped, so a partially-matching folder doesn't read as broken or empty.

The two discovery moments are the two points a user is most likely to be thinking about
"this project's stuff": launching the app while sitting inside the project (or a
subdirectory of it — the discovery walk goes upward looking for `.SKILLS/`, stopping at
the first ancestor containing `.git`, at `$HOME`, or at the filesystem root, so a launch
from a nested subdirectory still finds the project-root folder), and creating a
Console/Settings/Library workspace explicitly bound to a folder that turns out to
contain one. Both triggers write to the same ledger and read from the same discovery and
gating logic, so declining (or importing) at one silences the other for that folder.

## Import-copy, not live-load — relation to ADR-009

ADR-009 established a passphrase-rooted, authenticated local trust boundary for
Chatbook-managed skills: trust records, encrypted trusted snapshots, and a secure
generation marker all live *inside* that boundary, and execution requires the file on
disk to match what was trusted. A `.SKILLS/` folder living inside an arbitrary,
externally-editable project directory is explicitly **outside** that boundary — it has
no trust record, no generation marker, and can be modified by anything with filesystem
access to the project (another process, a `git checkout`, a compromised dependency's
postinstall script) without Chatbook ever seeing it.

Two designs were on the table for getting a project's skills into a running session:

1. **Live-load**: treat `.SKILLS/` as an additional skill source read at use time,
   directly from the project folder.
2. **Import-copy**: run discovered entries through the existing importer, which copies
   their content into the Chatbook-managed local skills store.

Live-load was rejected. It would mean skill content used to steer model behavior comes
from a location ADR-009's trust and tamper-detection machinery does not cover at all —
not "trust-blocked pending review" (a state ADR-009 already models and blocks execution
for) but structurally unprotected, since the trust boundary has no jurisdiction over
files it was never told to track. That reintroduces exactly the offline-tampering
concern ADR-009 exists to close, for the one class of skill content most likely to come
from a project the user did not author entirely themselves (cloned repos, downloaded
templates, a colleague's shared project).

Import-copy keeps this feature entirely inside ADR-009's existing model: an imported
skill becomes an ordinary Chatbook-managed local skill, subject to the same trust
records, the same encrypted snapshots, and — critically — the same **quarantine**.
Imports from `.SKILLS/` are never auto-trusted; every import runs with
`trust_approved=False`, identical to a manual Library import. A freshly imported skill
is excluded from the Console skill picker and any `$mention` of it is refused with a
pointer back to Library ▸ Skills, until a human reviews and approves it there. The
import modal states this "one-time trust review" expectation up front, because without
that framing the feature would read as broken (the skill appears to exist but silently
refuses to run).

The cost of import-copy is that the Chatbook-managed copy can drift from the project
folder's copy once either changes — there is no live sync. That tradeoff is deliberate:
a skill that could silently follow live edits to a project file is a skill an attacker
(or an errant `git pull`) can silently rewrite underneath an already-trusted session.
Requiring a fresh import (and fresh trust review) to pick up a changed version is the
same friction ADR-009 already accepts for its "reviewed re-trust before a changed skill
can run" rule — this decision extends it to the point of entry rather than carving out
an exception.

## Fingerprint ledger

Nagging on every launch from inside a project that declined the offer once would be
worse than not offering at all. A prompt ledger at `<user_data_dir>/skills/
project_prompts.json` (same atomic-replace-on-write discipline as the skills index)
records, per resolved project directory: the last decision (`imported` / `declined` /
`never`) and a fingerprint — a stable hash over the sorted (name, size, mtime) of the
recognized skill files at discovery time. The gating rule is: offer only if the feature
is enabled, `.SKILLS/` is present, and either there is no ledger entry, or the recorded
decision was not `never` **and** the current fingerprint differs from the recorded one.

This makes "Not now" mean "ask me again if this project's skill set actually changes"
rather than "ask me again next launch" — a project that hasn't been touched doesn't keep
producing the same prompt, but one that grew two new skills usefully re-surfaces as
exactly that. "Never for this folder" is permanent regardless of future fingerprint
changes. Both triggers (startup, workspace creation) share this one ledger and one
gating function, so a decision made from either surface is honored by the other.

## Kill-switch

`[skills] project_skills_prompt_enabled` (default `true`) disables the entire feature —
no discovery-driven prompting at either trigger — for a user who does not want Chatbook
scanning project directories for skill folders at all, without needing to hunt down and
decline every project individually. It does not affect manual import through Library ▸
Skills, which is unconditionally available regardless of this setting.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Live-load `.SKILLS/` as a skill source at use time | Puts trust-sensitive prompt content outside ADR-009's tamper-detection boundary entirely; see above. |
| Silent auto-import on first launch | Contradicts ADR-009's "first enablement must be explicit" posture, extended here to "first *ingestion* must be explicit" — a folder full of skills silently entering the picker (even quarantined) is a surprise, not a convenience. |
| Auto-trust imports from `.SKILLS/` because the user "already trusted the project" | Trusting a project directory (e.g. running its code, opening it in an editor) is not the same act as trusting arbitrary prompt content to steer the model; conflating them would let a compromised project skip the one review step ADR-009 relies on. |
| One-time-only offer per project (no re-offer on change) | Silently stops covering new skills added to a project after the first decline, which is worse than a single well-gated re-prompt on genuine change. |
| Merge multiple discovered `.SKILLS/` folders into a single combined offer | Rejected in favor of sequential, per-discovery modals — keeps each decision (and each ledger entry) tied to exactly one folder, avoiding a combined accept/decline that can't be attributed cleanly. |

## Consequences

Two new startup/workspace-creation code paths gain filesystem-scanning behavior
(bounded: symlink refusal, 50-entry cap, 64 KiB per-file frontmatter read cap, top-level
scan only) that must run off the main thread at startup so a slow or huge project
directory cannot stall app launch.

Every import from this path lands in the same quarantined state as a manual Library
import — no new execution surface, no new trust-bypass path. The feature adds discovery
and offer plumbing only; `execute_skill` and the Console `$mention` resolution path
remain the sole authority on whether an imported skill may run, unchanged by this
decision.

The prompt ledger is advisory, not authoritative: a lost or corrupted ledger entry only
costs the user one extra prompt, never a security regression, so ledger writes are
best-effort and must never crash the app or block startup on failure.

## Links

- [ADR-009: Local Skill Trust Boundary](009-local-skill-trust-boundary.md)
- [TASK-17651](../tasks/task-17651%20-%20Project-skills-SKILLS-folder-discovery-and-prompt-driven-import.md)
- [Workspace create modal + project skills design spec](../../Docs/superpowers/specs/2026-08-17-workspace-create-modal-and-project-skills-design.md) §5
- [Project skills import implementation plan](../../Docs/superpowers/plans/2026-08-17-project-skills-import.md)

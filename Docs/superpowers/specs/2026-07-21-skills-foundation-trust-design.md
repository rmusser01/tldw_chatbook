# Skills Foundation — Trust Isolation, Recovery & Discoverability (Spec 1)

**Status:** Approved design (brainstorming complete)
**Date:** 2026-07-21
**Author:** Robert (with Claude)
**Scope layer:** Layer 0 (Foundation), Spec 1 of 2

## Context

A UAT of a new user installing the `obra/superpowers` skillset (2026-07-21, against
`dev` with the merged Skills UX fixes) surfaced several foundation problems that make
the Skills feature hard to adopt. This spec addresses the trust and discoverability
findings. A sibling spec (Spec 2) covers path-aware nested supporting-file import
(finding #3) separately, because that change is security-sensitive and touches the
trust scanner.

This is Layer 0 of a larger program whose north star is "a user asks an agent to
install a skill/pack from a GitHub link." Later layers (multi-skill pack import,
remote/GitHub fetch, agent-callable install tool) are **out of scope** here and will
be brainstormed on their own. Layer 0 must be solid first: remote install is
pointless if landed skills are broken or trust dead-ends.

## Problem statement

Three UAT findings, all rooted in the local skill **trust** subsystem and its
surfacing:

1. **Cross-profile trust dead-end (finding #1, severe).** The rollback-protection
   generation **marker** is stored in the OS keychain under a *fixed global* service
   name (`tldw_chatbook.skill_trust`) and account (`local-skills:generation-marker:v1`),
   while the trust **manifest** is a per-profile file (`<store_dir>/trust/…`). A fresh
   profile therefore inherits another profile's (or a prior install's) marker; with no
   matching manifest file it reports `manifest cannot be verified`
   (`quarantined_manifest_error`) — a state whose UI offers **no working action**
   (Unlock/Review/Approve all disabled, no "Set up skill trust" button). The only
   documented recovery is manual on-disk/keychain deletion. Confirmed reproducible;
   it also poisoned the UAT itself via an orphaned marker from earlier testing.

2. **Trust setup is unguided and order-dependent (finding #4).** Nothing tells a new
   user that trust must be set up before imported skills are usable. The natural
   order (import first) lands the skill blocked, and the "Set up skill trust" action
   only appears inside a skill's editor once the trust state is already clean.

3. **Trust controls are unreachable (finding #5).** The trust panel sits at the bottom
   of a long editor form; the Body `TextArea` captures the scroll wheel, so the one
   control a user *must* act on is hard to reach.

Plus a minor finding:

4. **Minor (finding #6).** The command palette shows two near-identical "Library"
   entries ("Switch to Library" vs "Library — Skills"); a first-time user can pick the
   wrong one.

## Goals

- No trust state is ever an unrecoverable dead-end from the UI.
- Trust cannot be contaminated across profiles/users on the same machine.
- A new user can discover and complete trust setup without opening a skill or scrolling.
- Existing users upgrade cleanly (a one-time re-setup is acceptable; see decisions).

## Non-goals (explicitly deferred)

- Path-aware nested supporting-file import (finding #3) — Spec 2.
- Multi-skill / pack import (finding #2) — Layer 1.
- Remote / GitHub fetch — Layer 2.
- Agent-callable install tool — Layer 3.
- Migrating existing trusted baselines across the key-scoping change (decision: one-time
  re-setup, no migration path).

## Approved decisions

- **Keyring fix approach:** profile-scope the keyring key **and** add an in-UI reset,
  rather than dropping the keyring for a plain file (which would forfeit tamper-resistant
  rollback protection).
- **Migration:** one-time re-setup. Old global keyring entries are never read again; no
  migration code. Prior trust users re-run "Set up skill trust" once.
- **Trust surfacing:** a persistent, adaptive trust status line at the top of the Skills
  **list** (not only in the per-skill editor).

## Design

### Component 1 — Trust isolation

**Profile-scope both keyring entries.** The generation-marker keyring account and the
derived-key-cache keyring account gain a per-profile suffix derived from the resolved
absolute skills store directory, so no two profiles can read each other's entries.

- Identifier: a short, stable hash (e.g. first 16 hex of SHA-256) of the **resolved
  absolute** `store_dir`. Appended to the existing account string
  (`local-skills:generation-marker:v1:<hash>` and the key-cache equivalent). Service
  name may remain constant; the account is what scopes.
- The **file-fallback** marker store (`generation_marker.json` under `store_dir`) is
  already inherently per-profile and is unchanged.
- Any pre-existing global keyring entry (e.g. the orphaned marker currently poisoning
  fresh profiles on the developer's machine) becomes **inert** once profiles read only
  the scoped account — no manual keychain deletion is required to unblock it.
- **Both** the marker store and the key-cache store are scoped. Scoping only the marker
  would let a stale cached key mismatch a fresh marker.
- Consequence for existing users: a legit prior manifest file remains, but the marker
  under the new scoped key is absent → the trust service must treat "manifest present,
  scoped marker absent/unreadable" as a **recoverable first-run**, not a dead-end (see
  Component 2's recovery contract). This is the same class of state as the pre-existing
  pollution, and is resolved the same way.

**Interfaces touched:** `skill_trust_store.py` (marker store + key-cache store account
derivation), `app.py` (passes the store_dir-derived scope when constructing the stores;
it already builds them with the store dir in hand).

### Component 2 — Trust recovery (`reset_trust`) and dead-end elimination

**New `reset_trust()` on the trust service.** Clears this profile's manifest, generation
marker, and cached key material, returning the profile to `trust_uninitialized`
(first-run). Best-effort and non-crashing if the keyring is unavailable/locked (clears
what it can; reports partial failure via return value, never raises to the UI).

**Recovery contract — no dead-end state.** After this spec:

- `manifest cannot be verified` (`quarantined_manifest_error`) and the "manifest
  present but scoped marker absent" migration case both present a working recovery in
  the UI: **Reset local skill trust**, then Set up.
- **Set up skill trust** (bootstrap) must succeed even when a **stale/invalid manifest
  file already exists** (it overwrites, or internally resets-then-bootstraps). Today a
  bootstrap that assumes no existing manifest would leave upgraded users stuck; this
  contract closes that.
- The reset action is **destructive** (drops the trusted baseline; every skill returns
  to "needs review"), so it requires an inline confirmation making clear that **skills
  are not deleted — only trust is reset**.

**Surfacing of recovery.** The reset action is reachable from:
- the Skills list trust header (Component 3), and
- the blocked/`manifest_error` **editor** trust panel — replacing the current
  task-421 remediation copy that instructs manual on-disk deletion with the real
  in-UI Reset button (keep a short explanatory line).

**Interfaces touched:** `skill_trust_service.py` (`reset_trust`, bootstrap-over-stale-
manifest tolerance), `library_screen.py` (reset handler + confirmation), the trust panel
in `library_skills_canvas.py` (remediation copy → Reset action).

### Component 3 — Persistent trust header on the Skills list

A single adaptive status line rendered at the top of the Skills **list** canvas,
driven by the trust service's existing global **`trust_posture`** plus a count of
blocked skills:

| Global posture / condition | Header copy | Inline action |
|---|---|---|
| Trust service unavailable (allow-untrusted mode) | *(hidden — no trust concept)* | — |
| Zero skills installed | *(hidden — nothing to nag about)* | — |
| `uninitialized` (incl. migration/`manifest_error`) | "Skill trust isn't set up — set it up to review and use skills." | **Set up skill trust** (+ **Reset** when a stale manifest is present) |
| `locked` | "Skill trust is locked for this session." | **Unlock**, and **Reset** as a secondary "forgot your passphrase?" recovery |
| Any skills blocked / needing review | "N skill(s) need review before use." | **Review** (opens the first blocked skill's trust panel) |
| All clean | "Skill trust: ready." (quiet, low-emphasis) | — |

**Forgot-passphrase recovery:** a locked profile whose passphrase is lost is otherwise
permanently unusable. Reset (with confirmation) is therefore the deliberate escape hatch
from `locked`, not only from `uninitialized`/`manifest_error`. Reset is intentionally
*not* surfaced in the plain "needs review" state, where Review/Approve is the correct
action.

- **Do not nag.** Quiet/low-emphasis when ready; hidden when there is no trust concept
  or nothing to act on.
- Actions reuse the existing modal flows (bootstrap modal, passphrase modal) already
  used by the editor — no new dialog widgets, just list-level entry points.
- "Review" for Layer 0 (no pack) jumps to the first blocked skill's editor at its trust
  panel. Bulk review is a later-layer concern.
- The per-skill editor trust panel is **unchanged** in behavior; it is simply no longer
  the only path to trust.

**Interfaces touched:** `library_skills_canvas.py` (list-view header render + state
helper), `library_screen.py` (header action handlers, reusing existing trust flows).

### Component 4 — Minor polish (finding #6)

- Disambiguate the two palette "Library" entries with a small copy tweak so they read
  distinctly at a glance (e.g. the generic entry stays "Library", the deep link stays
  "Library — Skills" with help text that makes the difference obvious). Cheap, optional;
  no behavior change. Import Enter-to-submit remains the primary path (already reliable);
  no button rework.

## Data flow

1. **App start** → `app.py` builds the marker store + key-cache store, now passing the
   store_dir-derived scope → keyring accounts are per-profile.
2. **Enter Skills list** → canvas asks the screen for the aggregate trust state
   (`trust_posture` + blocked count) → renders the adaptive header.
3. **Set up / Unlock / Reset / Review** (from header or editor) → existing modal flow →
   `trust_service` bootstrap/unlock/reset → re-fetch posture → header + list re-render.
4. **Reset** → confirmation → `reset_trust()` → posture becomes `uninitialized` →
   header shows "Set up skill trust".

## Error handling

- Keyring unavailable/locked during scope read, bootstrap, or reset → best-effort,
  never crashes the UI; surfaces a clear warning and, where relevant, the file-fallback
  path (already exists) still functions.
- `reset_trust()` partial failure (e.g. keyring entry not deletable) → reports what was
  cleared; the manifest file removal (the part that matters for unblocking) still
  proceeds.
- Bootstrap over a stale manifest → overwrite/reset-then-bootstrap; never leaves the
  user in `manifest_error`.

## Testing

Following existing patterns (`Tests/Skills/`, `Tests/UI/test_library_skills_canvas.py`),
TDD:

- **Trust isolation:** two profiles (distinct store_dirs) with the same fake keyring
  backend do not read each other's markers; a marker written under one scope is invisible
  to the other. Key-cache scoped likewise.
- **Recovery:** a profile with a marker but no manifest (the poisoned state) →
  `reset_trust()` → `uninitialized`; then bootstrap → trusted. Bootstrap over a stale
  manifest succeeds. `reset_trust` is idempotent and non-crashing when the keyring is
  unavailable.
- **No dead-end:** `manifest_error` and "manifest present, scoped marker absent" both
  expose a working Reset/Set-up action (unit on the panel state + a harness test).
- **Header states:** each row of the Component-3 table renders the right copy/action;
  hidden when trust-unavailable or zero skills; quiet when ready; no nag.
- **Reset confirmation:** destructive reset requires confirm; confirm resets, cancel
  does not; copy states skills are not deleted.
- **Minor:** palette entries are distinguishable (assertion on the two command labels /
  help texts).

## Risks & mitigations

- **Upgrade churn:** existing trusted skills show "needs review" once after the scope
  change (accepted — one-time re-setup). Mitigated by the discoverable header + working
  recovery so the re-setup is a one-click bootstrap, not a dead-end.
- **Destructive reset:** guarded by confirmation and explicit "skills are not deleted"
  copy.
- **Keyring variance across OSes:** the scope is applied to the account string
  regardless of backend; the file fallback is already per-profile. Tests use a fake
  backend, not the real OS keyring.

## Out-of-scope reminder

Nested supporting files (#3), pack import (#2), remote fetch, and the agent tool are
**not** in this spec. Spec 2 is the immediate next step after this ships.

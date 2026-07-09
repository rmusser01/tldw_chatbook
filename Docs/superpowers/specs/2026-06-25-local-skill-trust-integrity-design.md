# Local Skill Trust Integrity Design

Date: 2026-06-25

## Purpose

Protect Chatbook-managed local skills from offline file tampering between app launches. A local skill that changes outside a trusted path must be visible but blocked until the user reviews the change and explicitly trusts the new version.

The primary concern is a user running Chatbook, shutting it down, having local skill files modified while the app is offline, then launching Chatbook and unknowingly staging or executing a backdoored skill prompt.

## Scope

This design covers Chatbook-managed local skills stored through `LocalSkillsService`.

It does not cover:

- Codex runtime skill folders such as `~/.codex/skills`.
- Server-managed skills.
- Local/server skill sync.
- Physical file quarantine or moving skill directories.
- Runtime protection against malicious active app code or a fully compromised user session.

## Current State

Local skills are stored in a Chatbook-owned library with:

- `tldw_chatbook_skills.json` metadata.
- `skills/<skill-name>/SKILL.md` plus flat supporting files.
- Atomic content/index writes.
- Metadata validation for Agent Skills front matter.
- Prompt-render-only execution with no hidden model invocation.

Current validation proves shape and metadata readiness, but it does not authenticate that the skill files are the same files the user previously trusted.

## Threat Model

The v1 trust boundary is designed for offline tampering where an attacker can modify files in the Chatbook local skills directory while Chatbook is not running, but cannot obtain the user trust passphrase or authenticated trust keys.

This design does not defend against:

- An attacker who can modify Chatbook application code.
- Malware that intercepts the passphrase.
- An attacker with access to the unlocked keyring or active process memory.
- A user intentionally approving a malicious diff.

Keyring convenience mode is lower assurance than passphrase-per-start mode and must be labeled that way.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/009-local-skill-trust-boundary.md`

Reason: This changes a security boundary, trust root, local storage semantics, runtime policy IDs, and the `LocalSkillsService` contract. The ADR must be created before implementation planning begins. If another ADR consumes `009` first, use the next available number and update implementation references.

## Architecture

Add a local skill integrity layer beside `LocalSkillsService`, not inside UI code. It owns trust verification, quarantine state, review snapshots, and trust decisions for Chatbook-managed local skills only.

Core components:

- `SkillTrustService`: scans skill directories, computes canonical file fingerprints, verifies the signed manifest, classifies drift, logically quarantines unsafe skills, and accepts reviewed baselines.
- `SkillTrustManifest`: authenticated manifest stored with the local skills store, accepted only when its HMAC verifies with a key derived from the user's startup passphrase.
- `SkillTrustKeyProvider`: passphrase/session key provider plus optional secure keyring cache. Plaintext fallback is forbidden. Keyring convenience mode stores a secure-keyring-protected trust root or wrapped trust root, never the user's passphrase.
- `SkillTrustGenerationMarkerStore`: stores the latest accepted manifest generation and manifest digest outside the local skills directory. The v1 target is the secure OS keyring. If no secure marker store is available, high-security mode fails closed; a user may explicitly choose reduced rollback protection, which must be visible in Settings and audit output.
- `SkillTrustSnapshotStore`: encrypted and authenticated trusted copies of `SKILL.md` and supporting text files for trusted-vs-current diffs.
- `LocalSkillsService` integration: `list_skills`, `get_skill`, `get_context`, and `execute_skill` include trust state and perform use-time trust checks.

Trust records cover every file in a skill directory, not just `SKILL.md`. The canonical fingerprint input includes normalized relative path, file type, byte length, and SHA-256 content hash.

Verify at use, not only at launch. Launch scan can populate visible status, but staging and execution must re-check the selected skill or a fresh scan generation.

Quarantine is logical in v1. Files stay in place for evidence and editor compatibility; the service marks the skill blocked.

Manifest failure is a global safety event. Missing-after-initialization, malformed, downgraded, replayed, or MAC-invalid manifests block all local skills until recovery or explicit re-bootstrap.

## Data Flow

On first enablement, Chatbook runs an explicit bootstrap flow. If no manifest and no generation marker exist, skills enter `trust_uninitialized`: visible but not stageable/executable until the user bootstraps trust. The bootstrap flow shows the number of discovered local skills and asks the user to confirm that the current files should become the initial trusted baseline. This avoids silent trust-on-first-use.

After confirmation, Chatbook asks for a trust passphrase, derives a root key, derives separate subkeys for manifest MAC and snapshot encryption/authentication, scans every local skill directory, writes encrypted trusted snapshots, writes the authenticated manifest, stores the latest accepted manifest generation marker, and records `trust_bootstrap`.

On startup or first skill access, Chatbook unlocks trust using the passphrase, or the secure-keyring-protected trust root when convenience mode is enabled. If unlock fails or is cancelled, skills remain visible but blocked as `trust_locked`. If unlock succeeds, Chatbook verifies the manifest MAC, schema version, generation, and rollback marker before trusting any skill status.

If a manifest exists but its generation marker is missing, lower than the manifest generation, or has a mismatched manifest digest, Chatbook treats it as `quarantined_manifest_error`. If the secure generation-marker store is unavailable at runtime, high-security mode blocks local skill use until the marker store is available. Reduced rollback protection mode may continue only if the user explicitly selected it during bootstrap or recovery; the UI must label that old full-store replay cannot be detected in that mode.

Scan states:

- `trusted`: current fingerprints match the trusted manifest.
- `trust_uninitialized`: trust has not been bootstrapped; skill use is blocked until explicit bootstrap.
- `trust_locked`: trust key unavailable, so use is blocked without claiming tampering.
- `quarantined_modified`: trusted skill content changed.
- `quarantined_added`: new file or skill appears without a trusted baseline.
- `quarantined_deleted`: expected trusted file is missing.
- `quarantined_manifest_error`: manifest cannot be authenticated or safely interpreted.
- `quarantined_unsupported_path`: symlink, nested path, unsafe filename, crash temp file, or unsupported file type appears.

On list/detail, skills remain visible with metadata and trust state. On context/stage/execute, the selected skill is verified again. If trusted, current behavior proceeds. If blocked, service behavior is deterministic:

- `list_skills()` and `get_skill()` include `trust_status`, `trust_reason_code`, `trust_blocked`, `trust_changed_files`, `trust_manifest_generation`, and `trust_last_verified_at` fields.
- `get_context()` excludes trust-blocked skills from `available_skills` and may include a `blocked_skills` extra field for UI diagnostics.
- `execute_skill()` raises a typed `SkillTrustBlockedError` with `skill_name`, `reason_code`, `trust_status`, and changed relative file names. It does not return a rendered prompt for a blocked skill.

On review, Chatbook captures a current snapshot and diffs it against the encrypted trusted snapshot. For a new skill with no trusted snapshot, the review compares current files against an empty baseline and shows every file as an addition. If the user approves, Chatbook re-scans the live files and verifies they still match the reviewed snapshot before signing a new baseline.

Manifest/snapshot updates are atomic: write new encrypted snapshots, write a new manifest temp file, fsync/replace where practical, then update the generation marker. Partial writes resolve to blocked/quarantine states, not partial trust.

## UX And Recovery

The Skills screen adds a trust/readiness layer without hiding metadata validation. A skill can be metadata-valid but trust-blocked.

Visible state is split:

- Metadata: `valid` / `invalid`
- Trust: `trusted` / `trust_uninitialized` / `trust_locked` / `quarantined_*` / `quarantined_manifest_error`
- Action: stage/execute enabled only when metadata and trust are both valid

For a quarantined skill, the detail/inspector pane shows plain-language reason, machine-readable reason code, changed files, last trusted timestamp, reviewed snapshot status, and recent trust audit events.

Actions:

- `Review Diff`: captures the current file snapshot and opens trusted-vs-current diff for supported text files.
- `Trust Reviewed Version`: disabled until the diff has been opened and the reviewed snapshot still matches live files.
- `Keep Quarantined`: leaves the skill blocked and records the decision.
- `Trust New Skill`: uses the same review gate, scoped to one new out-of-band skill.
- `Export Evidence`: optional follow-up that exports manifest entry, fingerprints, and audit events.

Unsupported or binary files show fingerprint and metadata only. Because current local skills are text-file based, unsupported files remain blocked unless a future policy explicitly allows them.

For global manifest failure, the UI shows global recovery instead of per-skill trust buttons:

- unlock again
- restore manifest/snapshots from backup
- re-bootstrap all local skills

Global re-bootstrap is high-friction: passphrase required plus explicit typed confirmation or a two-step prompt, because it trusts the current on-disk state for every local skill. Single-skill trust is available only when the manifest is valid and the issue is isolated to that skill.

`trust_locked` shows an unlock action and keeps skills visible but unusable. Settings > Privacy & Security reports skill trust posture and whether keyring convenience auto-unlock is enabled, but primary recovery lives in Skills.

User copy avoids "infected" or "hacked." Use precise states such as "changed since trusted baseline," "new untrusted file," "trusted file missing," and "trust manifest cannot be verified."

## Error Handling And Policy

Trust failures are deterministic and non-destructive:

- `trust_locked`: passphrase/key unavailable; block use without implying tampering.
- `trust_uninitialized`: no manifest and no generation marker exist; block use until explicit bootstrap.
- `manifest_invalid`: MAC/schema/generation/rollback check failed; global quarantine.
- `rollback_marker_unavailable`: secure generation marker store is unavailable in high-security mode; block use until the marker store is available or the user explicitly accepts reduced rollback protection.
- `skill_modified`: trusted file content changed.
- `skill_added`: untrusted skill directory or file appeared.
- `skill_deleted`: trusted file missing.
- `unsupported_path`: symlink, nested path, unsafe filename, crash temp file, or unsupported file type.
- `snapshot_mismatch`: reviewed snapshot no longer matches live files at approval time.
- `trust_store_write_failed`: baseline update failed; keep previous trust state if still valid, otherwise quarantine.
- `trust_history_unrecoverable`: passphrase lost; old snapshots/audit history cannot be decrypted or verified.

Trust operations get explicit local policy IDs, separate from normal skill CRUD:

- `skills.trust.unlock.local`
- `skills.trust.review.local`
- `skills.trust.approve.local`
- `skills.trust.reject.local`
- `skills.trust.rebootstrap.local`
- `skills.trust.rotate_key.local`
- `skills.trust.audit.local`

Policy defaults:

- No plaintext key storage.
- No silent downgrade from passphrase to insecure keyring backend.
- Keyring convenience mode stores a protected trust root or wrapped trust root, never the user's passphrase.
- A secure generation marker store is required for full rollback protection. Reduced rollback protection requires explicit user opt-in and visible posture reporting.
- Approval, re-bootstrap, and key rotation require an unlocked passphrase-derived key.
- Global re-bootstrap requires fresh passphrase confirmation.
- Lost-passphrase recovery is high-friction re-bootstrap of current files, with prior trust history marked unrecoverable.
- Key rotation verifies/decrypts the current manifest, snapshots, and authenticated audit records, then re-encrypts/re-MACs them with newly derived keys.
- Audit events are authenticated as part of the trust store, even if v1 does not implement a full append-only hash chain.
- UI/audit output uses skill names and relative file names by default; absolute paths appear only in explicit evidence export.
- Trust-blocked skills cannot be staged or executed, even if metadata-valid.
- Chatbook editor/import may update the trust baseline atomically only after explicit user approval for that mutation. The user's Save/Import confirmation can serve as the approval when the UI clearly states that the trusted baseline will be updated. Background or automatic edits do not silently re-trust.
- Physical quarantine/move is out of scope for v1.
- Server skills, Codex runtime skills, and sync semantics are out of scope.

Audit events include bootstrap, unlock failure, manifest failure, quarantine, review opened, trust approved, trust rejected, re-bootstrap, key rotation, lost-passphrase recovery, and write failure. Events include skill name, reason code, manifest generation, and changed relative file names.

## Testing Strategy

Add focused tests around the trust layer before UI tests:

- Manifest MAC verification rejects tampered manifest content.
- Missing/malformed/downgraded/replayed manifest causes global quarantine.
- No manifest and no generation marker produces `trust_uninitialized`, not per-skill quarantine.
- Missing or mismatched generation marker after initialization causes global quarantine.
- Secure generation marker store unavailability blocks high-security mode and produces visible reduced-protection posture only after explicit opt-in.
- Passphrase unlock derives stable keys and separates manifest/snapshot purposes.
- Secure keyring convenience mode refuses insecure/plaintext keyring backends and never stores the user passphrase.
- Trusted skill remains executable/stageable after unchanged scan.
- Modified `SKILL.md` quarantines and blocks `get_context`/`execute_skill`.
- Modified supporting file also quarantines and blocks use.
- New out-of-band skill/file quarantines until reviewed and approved, with review showing all files as additions against an empty baseline.
- Deleted trusted file quarantines.
- Symlink, nested path, unsafe filename, crash temp file, and unsupported file type classify as `unsupported_path`.
- Review approval fails with `snapshot_mismatch` if files change after diff capture.
- Approving reviewed snapshot updates encrypted snapshots, manifest generation, audit event, and restores use.
- Lost-passphrase recovery requires re-bootstrap and marks old history unrecoverable.
- Key rotation preserves trust status and makes old key fail verification.
- Trust-store partial write/crash residue resolves to blocked/quarantine state.

Service integration tests:

- `LocalSkillsService.list_skills()` includes trust state alongside metadata validation.
- `get_skill()` exposes blocked reason but still returns safe detail for review.
- `get_context()` excludes trust-blocked skills from `available_skills` and may include `blocked_skills` diagnostics without making blocked skills available to stage.
- `execute_skill()` raises `SkillTrustBlockedError` and never returns a rendered prompt for trust-blocked skills.
- Chatbook-approved create/update/import paths update the baseline only as part of explicit approved mutation, and Save/Import confirmation text states that the trusted baseline will be updated.

UI tests:

- Skills screen shows metadata-valid but trust-blocked as blocked.
- Attach button enables only for metadata-valid and trusted skill.
- Review Diff enables Trust Reviewed Version only after a captured diff.
- Global manifest failure shows global recovery, not per-skill approval.
- Settings Privacy & Security shows trust posture and keyring convenience state without leaking paths or secrets.

Verification should run focused Skills tests plus affected Settings/UI tests. Because this introduces a security boundary, add the ADR before implementation planning.

## Implementation Notes For Future Planning

Start implementation with the ADR and a small trust-service substrate before touching Skills UI. The first implementation slice should prove manifest verification, locked/quarantined states, and `LocalSkillsService` blocking behavior without building the full diff UI.

Keep the first PR narrow enough that a reviewer can audit every trust transition. Avoid combining this with server skills, Codex runtime skills, or sync.

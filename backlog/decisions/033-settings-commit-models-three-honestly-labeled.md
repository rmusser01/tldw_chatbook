# ADR-033: Settings Commit Models — Three Honestly-Labeled Models, Not One

- Status: Accepted
- Date: 2026-08-05
- Context: The Settings screen re-critique (`.impeccable/critique/2026-08-05T16-56-50Z__tldw-chatbook-ui-screens-settings-screen-py.md`, Consistency score 3/4) posed the open question: three commit models coexist — (1) **staged draft** (edits accumulate dirty keys → `*` markers → banner → `s` save / `r` confirmed revert), (2) **labeled instant-apply** for low-risk toggles (model-catalog auto-refresh checkboxes, stale-hours, splash default card, theme Apply), (3) **guarded raw TOML** in Advanced Config (validation-gated save, atomic write + `.bak`, backup preview) — is "honest about complexity" the goal, or should Settings converge on one model with instant-apply as the justified exception?
- Decision:
  1. **Keep three commit models; do not converge to one.** The models exist because the settings they serve differ in risk, not because of drift.
  2. **Staged draft is the default.** Any setting with validation, cross-field dependencies (e.g. provider switch rewriting the Endpoint field), or non-trivial reversal cost commits through the draft state machine. New fields default to staged unless they qualify for rule 3.
  3. **Instant-apply is the labeled exception, restricted to low-risk, independently-meaningful, trivially-reversible preferences** (operational toggles, cosmetic choices). Every instant-apply control MUST carry the shared `INSTANT_APPLY_BEHAVIOR_COPY` label ("applies immediately - no Save needed") inline or in the focused-field inspector, and MUST NOT participate in the staged banner/dirty-marker machinery — mixing the two on one control is the bug class this ADR exists to prevent.
  4. **Guarded raw TOML stays expert-only** behind its validation gate and atomic-write-with-backup path; it is not a third user-facing model so much as a file editor with a seatbelt, and it MUST NOT grow unguarded save paths.
  5. **Complexity is managed by labeling, not elimination.** The consistency requirement (ADR-031's honesty posture extended) is that a user can always tell which model a control follows from what is on screen — the commitment is to *honest* pluralism, not to a single model.
- Alternatives considered:
  - Converge everything to staged draft: rejected — forces `s`+banner friction onto harmless toggles (splash card choice, auto-refresh flags), training users to expect friction everywhere and inviting them to save-bash; the draft machinery also adds failure modes (stale drafts, lost reverts) to settings that never needed them.
  - Converge everything to instant-apply: rejected — unsafe for validated, cross-field, or expensive-to-reverse settings; removes the reviewable draft that makes provider configuration recoverable for non-experts.
  - Converge to two models by removing guarded TOML: rejected — Advanced Config serves recovery and power-user scenarios the form UI cannot express; its validation gate + `.bak` already mitigate the risk.
- Consequences: the three models are the documented contract; code comments at `settings_screen.py` (`STAGED_SAVE_BEHAVIOR_COPY` / `INSTANT_APPLY_BEHAVIOR_COPY`, task-1341) are the normative labeling mechanism; reviews of new Settings fields must state which model the field follows and why; future critique passes should score consistency on *label truthfulness per control*, not on model count.
- Links: re-critique snapshot above ("Questions to Consider" #1); ADR-031 (footer/copy honesty posture this extends); task-1341 (staged-vs-instant labeling implementation); task-1372 (this decision).

## Guarded raw draft recovery (TASK-32190, 2026-09-09)

Advanced Config owns an in-memory text draft and the exact file/profile snapshot
from which it was loaded. Category and destination navigation retain that draft,
including empty or invalid text; no raw body is stored in durable UI metadata.
The existing memory-only screen-state store retains the live raw-editor session
so destination recreation observes in-flight results. Its workers belong to the
app, and mounted views attach/detach callbacks with ownership checks. Initial
file reads run off the UI thread. Before applying asynchronous results, the
controller captures editor input whose change event has not yet been delivered.
Its dirty marker and banner name the raw editor's own Validate/Save/Revert path.
Validation remains tied to the current text revision. Revert and Load Backup
confirm before replacing unsaved work; a delayed worker cannot replace newer
edits or update another category's shared status.

Raw saves compare the original serialized file and effective profile identity
under the existing config owner's write lock. If either changed, saving is
blocked with recovery guidance and the draft is retained. The owner returns the
post-write snapshot from that same lock, preserving encryption, revision-owned
sections and atomic backup/replacement. This extends the existing owner contract,
not a new persistence path. A successful write clears only its submitted draft;
later edits remain unsaved. Revert reloads the current file after confirmation.

An unconditional overwrite after navigation is rejected because preserving a
draft makes intervening guided edits common. Automatic merging of arbitrary
raw TOML is also rejected: the editor cannot infer the user's intended changes
across comments, section deletion, encryption and revision-owned tables. Users
retain their draft to copy or compare and explicitly reload before reapplying.

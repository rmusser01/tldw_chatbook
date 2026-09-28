# Console hook configuration and review

Date: 2026-09-27
Status: Written spec reviewed positively by the user; requested audit addressed
Task: [TASK-33151](../../../backlog/tasks/task-33151%20-%20Design-Console-hook-settings-and-persistent-review.md)
ADR: [ADR-197](../../../backlog/decisions/197-console-hook-configuration-review.md)

## Purpose and agreed behavior

Make user-configured Console hooks discoverable, editable, and subject to explicit
consent before their external commands run. Use native Textual controls in the
actual Console and canonical F9 Settings screen.

- Keep a Hooks icon beside Settings in the Console toolbar. Its pending count
  indicates enabled hooks needing review; it opens current permissions at any time.
- Review existing hooks once on upgrade. Review new hooks and execution changes
  again. Persist decisions across application restarts.
- Detect changes without interrupting the user. Open automatic review on the next
  actual Send, before accepting or queueing the message.
- Preserve the draft when review is cancelled. Configuration editing lives in
  Settings; the modal links directly to its Hooks category.

This covers standalone user-config hooks and the six implemented lifecycle events.
Plugin activation, expanded v2 hooks, project hooks, shell-string commands, and a
browser settings surface are outside this change.

## Existing owners

The Console composes a visible `ConsoleControlBar`, context rail, transcript,
Inspector, collapsible status strip, and bounded composer. Its ModeStrip and
CommandStrip are hidden. The Approvals chip reviews tool requests in the transcript;
it must keep that meaning. The hook control belongs in the visible top action row,
where composer/status collapse cannot hide it.

`SettingsScreen` owns category/detail/impact panes, staged edits, and category
deep links. Add a Hooks category in its Expert group and include it in category
search. Do not add configuration to either deprecated Settings parallel.

`Agents/run_hooks.py` parses and executes argv lists. `ConsoleRuntime` owns one
live-config engine, shared by manual, queued, durable, and background paths.
Console UI glue belongs in `UI/Console_Modules/` and its existing wiring module;
the screen delegates to it rather than acquiring another large implementation.

## Native layout

The ASCII `H` stands for the Hooks glyph with a supported ASCII fallback. The
number is a count, not a new toolbar or composer row.

```text
Console
New tab  Settings  [H:3]  Context rail  Search Library  Help
+-------------------+---------------------------------+-------------------+
| Context rail      | Transcript                      | Inspector         |
|                   |                                 |                   |
|                   |                                 |                   |
+-------------------+---------------------------------+-------------------+
Status v [Library] [Provider] [Model] [Assistant] [Tools] [Approvals]
Composer v  Menu  | <unsent draft>                     | Send | Dictate
```

Activate the icon with mouse, Enter, or Space using the existing toolbar action
route. With no pending hooks it remains available without a count. With hooks
globally disabled it communicates Disabled and still opens inspection. A tooltip
and accessible label name Hooks and the count; color is supplementary. The pending
count includes enabled invalid rows needing repair, with their separate error count
explained in the label/modal. A malformed master/table/list shows an error indicator
and repair guidance instead of claiming there are no hooks. Keep the action within
the visible toolbar bounds at 80 and 120 columns; compact secondary action labels
if needed and retain their full tooltips and keyboard access.

```text
+-----------------------------------------------------------------------+
| Review hooks                                                          |
| 3 hooks need review. Your message is waiting.                          |
|                                                                       |
| [Needs review: 3] [All hooks: 5]                                       |
|                                                                       |
| [x] PostToolUse - check-ui.py                 [Existing]             ^ |
|     Source       User config                                          |
|     Event        PostToolUse                                          |
|     Command argv ["python3", "/path/to/check-ui.py"]                   |
|     Matcher      fs_*                                                 |
|     Timeout      5s                                                   |
|     Permission   Needs review                         [Keep disabled] |
|                                                                       |
| [ ] PreToolUse - guard.py                     [New]                  v |
| [ ] Stop - notify.py                         [Modified]             v |
|                                                                       |
| Existing hooks need a one-time review before they can run.            |
| Manage in Settings                                                    |
|                                                                       |
|                       [Not now] [Allow all] [Allow selected]           |
+-----------------------------------------------------------------------+
```

Rows summarize event plus executable/script; no separate display-name schema is
needed. Pending rows begin unchecked. Expand a row to inspect the exact argv list,
source, event, optional glob matcher, timeout, and permission. Wrap long arguments
in the scroll body without truncating the reviewable command. Render argv and
matcher details as JSON strings/arrays, escaping control characters and directional
formatting characters; sanitize only derived summary labels. Never interpret user
values as Rich markup or terminal escape sequences. Event labels are the actual
runtime names. Build detailed content lazily when a row expands, and run config
reads, parsing, hashing, and writes off the UI thread when they can exceed 100ms.

The body scrolls independently; title and actions remain reachable. Use the native
modal dismissal/focus-restoration mixin, token-backed modal patterns, and ordinary
Tab/Shift-Tab navigation. Escape and Not now close review and restore focus without
sending or clearing text. Opening from the icon has no pending Send and omits
"Your message is waiting."

All hooks includes Approved, Disabled, Needs review, Needs recovery, and Invalid
entries. Needs review includes enabled invalid rows, visibly marked Invalid with a disabled
approval checkbox, plus recovery-sealed rows with retry guidance, so a blocked Send
cannot hide its cause in the other tab. A persisted grant cannot override a live
recovery seal; only successful recovery or an explicit current decision clears it.
Approved rows expose Revoke approval. Invalid table entries expose their validation
reason and Settings/disable actions; non-table entries and malformed containers
require repair/removal or an explicit master disable. They cannot be approved.
New/Modified/Existing badges
describe discovery history, while permission is a separate explicit state.

Allow all approves the current valid, enabled pending entries. Allow selected
approves only checked entries; unselected entries remain pending. If any enabled
entry still needs review, keep the modal open and explain the remaining count.
Keep disabled immediately persists that hook's disabled state through the config
owner and removes it from the Send requirement; label that action as immediate.
Revocation is also immediate. A failed approval never grants execution. Revocation
and disable seal the affected hooks in the current runtime before waiting for disk
I/O, and failed persistence keeps that seal with retry/recovery guidance. Do not
claim durable or cross-process revocation until the write succeeds. Distinguish a
write failure from a successful replacement followed by failed runtime refresh.

When every enabled entry is valid and approved, a Send-opened modal may resume
exactly that captured Send once. A toolbar-opened modal only closes after applying
the decisions. Manage in Settings cancels automatic continuation and deep-links
to Hooks, retaining the composer draft for the user's next Send.

```text
Settings
+------------------+----------------------------------+------------------+
| Categories       | Hooks                            | Impact           |
| Filter...        | Hooks enabled [x]                | Source           |
| ...              |                                  | User config      |
| Expert           | [Add hook] [Review permissions:3]|                  |
| > Hooks          |                                  | Scope            |
|   Advanced config| PostToolUse - check-ui.py         | All Console      |
|                  | PreToolUse - guard.py            | chats using      |
|                  | Stop - notify.py                 | this config      |
|                  |                                  |                  |
|                  | Event        [PostToolUse v]     | Approval         |
|                  | Command argv                     | Needs review     |
|                  | ["python3", "/path/check-ui.py"] |                  |
|                  | Matcher      fs_*                | Saving execution |
|                  | Timeout      5                   | changes requires |
|                  |                                  | review again     |
|                  | [Disable hook] [Remove hook]     |                  |
|                  | [Save changes] [Revert]          |                  |
+------------------+----------------------------------+------------------+
```

The command editor is a TextArea containing a JSON array of strings. This preserves
argument boundaries without shell quoting or a new row editor. Reuse runtime
validation: a nonempty argv of NUL-free strings with a nonempty executable, a
supported event, a positive finite numeric timeout (not a boolean), and an optional
nonempty case-sensitive glob only for PreToolUse/PostToolUse. New hooks
default to disabled until explicitly enabled and reviewed. Invalid originals remain
visible for repair; do not rebuild the editor solely from the parser's valid subset.

Hook configuration follows Settings' staged Save/Revert model, with its shared
behavior copy and dirty markers. Revert restores the current authoritative config
under existing confirmation rules. Saved rows do not acquire permission. Review
permissions operates only on saved definitions; explain unsaved changes before
opening it. Disabling/removing a hook in Settings is staged, distinct from the
modal's explicitly labeled immediate disable. The per-row action reads Enable hook
for a disabled row and Disable hook for an enabled row. An empty category offers
Add hook and explains the six events without a permission dialog.

## Configuration, identity, and consent

Keep `[hooks]` and `[[hooks.hook]]` in the existing user config, adding optional
per-hook `id` and `enabled` fields. Enable switches must be booleans, and explicit
IDs must be nonempty strings. Legacy missing `enabled` means true. Guided
saves assign opaque stable IDs and preserve them across edits/reordering. IDs are
identity only: a supplied ID, `approved` field, project instruction, or plugin
manifest cannot grant consent. Duplicate explicit IDs are invalid.

Approve an exact normalized execution definition: event, ordered argv strings,
matcher or null, and numeric timeout. Fingerprinting uses fixed-field, deterministic
JSON with a format version; normalize timeout as the validated runtime float, make
omitted matcher equivalent to null, and preserve argv strings/order exactly. Store
its SHA-256 fingerprint with the hook identity and effective config-file identity.
The config scope is its canonical effective path, not a transient config generation
or file inode that changes on atomic replacement. Master/per-hook enable flags
are switches, not grants. Disabling and re-enabling an unchanged approved hook
retains its approval; revocation clears it. An observed execution change or removal
retires the prior grant, including when an old definition is later restored.

Legacy rows without IDs use the definition fingerprint plus occurrence among
identical definitions, scoped to the config file. Array reordering cannot move
approval to a different command. If the number of identical ID-less rows changes,
invalidate that definition's entire legacy grant group: an occurrence number cannot
prove which approved or unapproved duplicate was removed. Adding a duplicate also
requires review of the group. Stable IDs remove this ambiguity after a guided save.
An edited ID-less row is safely classified New rather than guessing lineage by
array position. Guided ID assignment may preserve a grant only for the unchanged,
exact legacy row against the current config snapshot. Otherwise require review.

Use one small application-owned JSON consent file under the canonical user data
directory (`hook_permissions.json`), with schema version, observed-definition
metadata, and current grants. This is local to the user/device and config file,
independent of tool permission profiles; no database migration or dependency is
needed. Persist IDs, fingerprints, and decision/observation metadata, not argv,
prompt/tool payloads, or hook output; command details come from current config.
First adoption records the current baseline as Existing and grants none.
Newly observed identities are New; changed stable IDs are Modified. Observations
carry no execution authority.

One runtime-owned consent owner serializes reconciliation and explicit decisions.
Use the existing private-path readers, atomic private writer, and interprocess lock
pattern for persisted state; include its live path and companions in sensitive-path
exclusions so local filesystem/context tools cannot read or rewrite grants. Moving
to a different effective config file cannot inherit approval by array position.
Missing state requires review; corrupt/unreadable state, unsupported schema versions,
or failed writes never permit execution or report successful persistent consent.
Recovery allows explicit reset of invalid state or disabling hooks, with drafts
retained. A valid master disable or an inventory with no enabled rows allows Send
without requiring a usable grant store. Switching to another user data root fences
cached grants until the new store has been read under its own identity.

Consent is enforced by this upgraded runtime, not by operating-system containment
or older Chatbook versions that ignore these new fields. Follow the existing local
permission-store threat boundary; do not imply cryptographic protection from other
arbitrary processes already running as the same user.

This is consent to a command definition, not executable-content signing. Changing
a script/binary at the same path, its inherited environment, or normal cwd does not
trigger configuration review. The modal states that commands run with the user's
privileges. Existing timeout, cwd, capture, and payload boundaries remain in force.

Read hooks through a lossless inventory of the authoritative saved section, including
invalid rows and malformed containers, independently of the valid-execution subset.
`ensure_run_hooks() is None` cannot serve as consent clearance: today's parser drops
invalid definitions. Refresh through the config owner at Console mount/activation,
review open, Send, and launch, plus after guided/raw saves. Manual TOML changes are
recognized at those checkpoints, without a filesystem polling service. The existing
engine's `app.app_config` reference alone is not an authoritative disk snapshot.

Settings saves and modal disables use the existing single config writer. Compare
the original hooks section and effective file identity under its lock, replace the
whole hook list, and preserve unrelated sections and unknown fields. A generation
or section revision alone does not detect a manual TOML edit with an unchanged
revision. Stale saves retain the draft and require reload/explicit recovery; do not
merge arrays or overwrite a concurrent raw-config edit. Config and grant writes
are separate atomic operations: interruption may require fresh review, never a
broader grant or a falsely successful action. Consume the owner's structured
file-replaced/runtime-refreshed result. If replacement succeeded but publication
failed, retain the actual saved outcome and a runtime fence, offer refresh/recovery,
and do not retry an assumed failed write over subsequent user edits.

When both config and consent locks are required, acquire config first and consent
second, and never call back into config under a consent-only lock. Do not wait for
these locks while holding the engine's pool-admission lock or on the Textual thread.
Re-read state under the consent file's interprocess lock; memory caches cannot
overwrite another running instance's decisions or authorize from stale grants.

## Send admission and shared runtime enforcement

1. Parse Console commands as today. A command that does not submit a user turn
   does not open hook review.
2. Before normal Send or queue admission, capture the originating session, draft
   stash/revision, config identity, and hook snapshot. If the master switch is off
   or all enabled hooks are valid and approved with no live recovery seal, continue
   the existing dispatcher. Only a valid explicit master disable bypasses malformed
   hook entries; a parser-induced disabled result is not that user decision.
3. Otherwise open one review modal without committing the composer, admitting a
   queue entry, echoing a user turn, or launching a hook/provider request.
4. Persist explicit choices only for the still-current reviewed definitions and
   consent revision. Repeated Send/Allow activation joins or ignores the current
   operation rather than creating another continuation. A delayed approval must
   not undo a newer revocation from another view or application instance.
   Configuration edits refresh the rows and require a new selection; a stale
   response cannot approve a changed command.
5. Before continuation, recheck current hooks, session ownership, and draft stash.
   If the user changed chats or the captured draft is no longer current, retain
   the text and require a new Send. Do not replay into another chat or send twice.
   Bind continuation to a single consumable operation generation. Escape, dismissal,
   or Settings navigation cancels that generation immediately; merely returning
   to the same session/draft cannot revive it. Late workers may finish an explicitly
   submitted persistent decision, but cannot update a dismissed modal or auto-send.
   Check screen-stack ownership, not `is_mounted` alone, before publishing UI.

The preflight is presentation convenience, not the authority boundary. Shared
runtime admission checks the same consent owner for direct, queued, recovered,
durable, and wake-origin work. Background/viewless paths never pop a modal: refuse
or retain unsent work through their existing refusal/recovery owner and publish
metadata explaining review is required. A queued prompt is checked again when it
is dispatched; if consent changed, retain it under the queue's existing
`DISPATCH_REFUSED` pause/recovery flow. Preserve durable acceptance/recovery records;
do not pretend an already accepted turn became an unsent composer draft.

Serialize the final consent/config check and actual subprocess creation against
consent changes. Release that launch critical section after creation, before waiting
for output or completion; a separate check followed by an unlocked spawn has a race.
This is the linearization point: revocation may wait for a launch already inside
the section, but no launch can obtain permission after the revoke seal wins.

Capture notification target identities/definitions at event admission. Queued work
rechecks them at launch; skip removed, disabled, changed, or revoked targets and do
not apply newly added handlers to an earlier event's captured payload. Keep the
existing bounded queue/payload limits. An enabled pending
PreToolUse guard denies affected calls; an unreviewed UserPromptSubmit definition
refuses admission rather than inheriting its execution-error fail-open rule.
Unreviewed observers do not launch and report a bounded metadata-only omission.
Do not silently filter an unapproved guard out and thereby relax restrictions.
Represent consent refusal or unavailability explicitly; it must never fall through
the engine's generic UserPromptSubmit exception-to-fail-open handler. A malformed
enabled definition whose event/matcher cannot be established blocks tool dispatch
conservatively until repaired/disabled rather than vanishing from the guard set.

Revocation fences new launches and queued notifications immediately. A process
already launched under valid consent retains the existing bounded timeout/shutdown
ownership; its completed side effects cannot be undone. Do not add per-hook process
termination in this change. Keep the existing deny-only tool guard, approval
exemptions, manual-only UserPromptSubmit firing, payload protocol, event ordering,
and execution-error fail directions. Permission to run a hook never grants a tool.

## Implementation boundaries and verification

Reuse the existing toolbar action state, native modal mixin, settings category and
navigation registries, config mutation owner, queue refusal/recovery mechanisms,
and runtime engine. Add only the consent owner, review presentation, and guided
hook editor needed by these flows. No generic authorization framework, plugin
trust duplication, polling service, extra composer button, or unrelated refactor.

Apply [the design language](../../../backlog/docs/design-language.md): token-backed
status/focus/spacing, literal safe labels, bounded scroll regions, and responsive
pane laws. Edit source CSS modules and rebuild generated CSS if implementation
changes styles. Respect screen-size/view-hook-slot ratchets and the repository's
keybinding conventions; introduce no terminal-convention shortcuts.

Targeted implementation verification must cover:

- Legacy one-time review and persisted restart approval; exact-field changes,
  reordering, duplicate-count changes and deletion of an approved duplicate,
  disable/re-enable, removal/re-add, and revocation.
- Safe failure for invalid config, unreadable/corrupt consent, failed writes,
  unsupported store versions, post-replacement publication failure, concurrent
  settings/raw edits, multiple app instances, and stale modal callbacks.
- Actual mouse/keyboard toolbar activation, both modal tabs, details, focus
  restoration, Settings deep-link, staged save/revert, and narrow terminal layouts.
- Send cancellation without draft loss; exactly-once resume; session/draft changes;
  repeated Send/Allow, dismissal during a write, return to the same chat/draft,
  queue admission/dispatch, recovered/durable submissions, and viewless wakes.
- A real approved command that runs as a positive control, an unapproved command
  that never starts, a controlled revocation-versus-spawn race, frozen notification
  targets, and unchanged deny-only/fail-direction hook behavior.
- Sensitive-path exclusions, private file ownership, design-token governance,
  generated CSS freshness, and existing Console architecture checks.

Use targeted tests and isolated temporary user config/data for live verification.
Do not execute the user's real hooks or run a full test sweep without authorization.
Update the user hook guide only with implemented and verified behavior.

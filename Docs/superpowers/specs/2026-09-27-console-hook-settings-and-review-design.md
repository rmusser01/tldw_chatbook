# Console hook configuration and review

Date: 2026-09-27
Status: Native layout approved; written spec awaiting user review
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
and accessible label name Hooks and the count; color is supplementary. Use existing
toolbar containment and focus rules at narrow widths.

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
in the scroll body without truncating the reviewable command. Render user values
as literal text, never Rich markup. Event labels are the actual runtime names.

The body scrolls independently; title and actions remain reachable. Use the native
modal dismissal/focus-restoration mixin, token-backed modal patterns, and ordinary
Tab/Shift-Tab navigation. Escape and Not now close review and restore focus without
sending or clearing text. Opening from the icon has no pending Send and omits
"Your message is waiting."

All hooks includes Approved, Disabled, Needs review, and Invalid entries. Approved
rows expose Revoke approval. Invalid entries expose their validation reason and
Settings/disable actions; they cannot be approved. New/Modified/Existing badges
describe discovery history, while permission is a separate explicit state.

Allow all approves the current valid, enabled pending entries. Allow selected
approves only checked entries; unselected entries remain pending. If any enabled
entry still needs review, keep the modal open and explain the remaining count.
Keep disabled immediately persists that hook's disabled state through the config
owner and removes it from the Send requirement; label that action as immediate.
Revocation is also immediate. Failures retain the prior decision and show recovery
guidance rather than pretending an approval or disable succeeded.

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
modal's explicitly labeled immediate disable. An empty category offers Add hook
and explains the six events without a permission dialog.

## Configuration, identity, and consent

Keep `[hooks]` and `[[hooks.hook]]` in the existing user config, adding optional
per-hook `id` and `enabled` fields. Enable switches must be booleans, and explicit
IDs must be nonempty strings. Legacy missing `enabled` means true. Guided
saves assign opaque stable IDs and preserve them across edits/reordering. IDs are
identity only: a supplied ID, `approved` field, project instruction, or plugin
manifest cannot grant consent. Duplicate explicit IDs are invalid.

Approve an exact normalized execution definition: event, ordered argv strings,
matcher or null, and numeric timeout. Store its versioned SHA-256 fingerprint with
the hook identity and effective config-file identity. Master/per-hook enable flags
are switches, not grants. Disabling and re-enabling an unchanged approved hook
retains its approval; revocation clears it. An observed execution change or removal
retires the prior grant, including when an old definition is later restored.

Legacy rows without IDs use the definition fingerprint plus occurrence among
identical definitions, scoped to the config file. Array reordering cannot move
approval to a different command, and adding a duplicate still needs review.
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
Missing state requires review; corrupt/unreadable state or failed writes never
permit execution or report successful persistent consent. Recovery allows explicit
reset of invalid state or disabling hooks, with drafts retained.

This is consent to a command definition, not executable-content signing. Changing
a script/binary at the same path, its inherited environment, or normal cwd does not
trigger configuration review. The modal states that commands run with the user's
privileges. Existing timeout, cwd, capture, and payload boundaries remain in force.

Settings saves and modal disables use the existing single config writer. Compare
the original hooks section and effective file identity under its lock, replace the
whole hook list, and preserve unrelated sections and unknown fields. A generation
or section revision alone does not detect a manual TOML edit with an unchanged
revision. Stale saves retain the draft and require reload/explicit recovery; do not
merge arrays or overwrite a concurrent raw-config edit. Config and grant writes
are separate atomic operations: interruption may require fresh review, never a
broader grant or a falsely successful action.

## Send admission and shared runtime enforcement

1. Parse Console commands as today. A command that does not submit a user turn
   does not open hook review.
2. Before normal Send or queue admission, capture the originating session, draft
   stash/revision, config identity, and hook snapshot. If the master switch is off
   or all enabled hooks are valid and approved, continue the existing dispatcher.
3. Otherwise open one review modal without committing the composer, admitting a
   queue entry, echoing a user turn, or launching a hook/provider request.
4. Persist explicit choices only for the still-current reviewed definitions.
   Configuration edits refresh the rows and require a new selection; a stale
   response cannot approve a changed command.
5. Before continuation, recheck current hooks, session ownership, and draft stash.
   If the user changed chats or the captured draft is no longer current, retain
   the text and require a new Send. Do not replay into another chat or send twice.

The preflight is presentation convenience, not the authority boundary. Shared
runtime admission checks the same consent owner for direct, queued, recovered,
durable, and wake-origin work. Background/viewless paths never pop a modal: refuse
or retain unsent work through their existing refusal/recovery owner and publish
metadata explaining review is required. A queued prompt is checked again when it
is dispatched; if consent changed, hold it for explicit recovery rather than
losing it or releasing hooks with stale authority.

Recheck matching hook consent immediately before each subprocess launch, including
notifications queued before a config change or revocation. An enabled pending
PreToolUse guard denies affected calls; an unreviewed UserPromptSubmit definition
refuses admission rather than inheriting its execution-error fail-open rule.
Unreviewed observers do not launch and report a bounded metadata-only omission.
Do not silently filter an unapproved guard out and thereby relax restrictions.

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
  reordering, duplicates, disable/re-enable, removal/re-add, and revocation.
- Safe failure for invalid config, unreadable/corrupt consent, failed writes,
  concurrent settings/raw edits, and stale modal callbacks.
- Actual mouse/keyboard toolbar activation, both modal tabs, details, focus
  restoration, Settings deep-link, staged save/revert, and narrow terminal layouts.
- Send cancellation without draft loss; exactly-once resume; session/draft changes;
  queue admission/dispatch, recovered/durable submissions, and viewless wakes.
- A real approved command that runs as a positive control, an unapproved command
  that never starts, and unchanged deny-only/fail-direction hook behavior.
- Sensitive-path exclusions, private file ownership, design-token governance,
  generated CSS freshness, and existing Console architecture checks.

Use targeted tests and isolated temporary user config/data for live verification.
Do not execute the user's real hooks or run a full test sweep without authorization.
Update the user hook guide only with implemented and verified behavior.

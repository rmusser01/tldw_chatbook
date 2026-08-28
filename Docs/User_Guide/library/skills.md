# Library Skills — create, import, review, and trust reusable skills

## What this screen is for

A skill is a reusable instruction pack — a `SKILL.md` body plus optional
supporting files — that you run in Console by starting a message with
`$name` (see [Agent runs & tools](../console/agent-runs-and-tools.md)).
This Library panel is where skills live: create and edit them, import them
from files, folders, or URLs, and — the part nothing else does — review and
approve them in the trust panel. A skill that isn't trusted refuses to run
in Console, and this is the screen that fixes that.

## Getting there

Press **Ctrl+3** (or **Ctrl+P** → "Tab Navigation: Switch to Library"),
then in the left rail's
**Browse** section click **Skills** (the row shows its count). To start a
new skill, use the rail's **Create ▸ New skill** instead — it opens the
editor in create mode.

## Layout tour

```text
Library rail                 Skills list / editor
┌────────────────────┐       ┌──────────────────────────────────────────┐
│ Browse             │       │ Skills (N)                               │
│   Skills           │  ───▶ │ Filter skills…                           │
│ Create             │       │ sort: Name        Import…                │
│   New skill        │       │ ✓ code-review                            │
└────────────────────┘       │ ⚠ summarize                              │
                             └──────────────────────────────────────────┘
```

The list canvas, top to bottom:

- **Heading** — "Skills (N)".
- **Trust header** — one line stating the current trust posture, with an
  action button when there is something to do (both hidden while the list
  is empty). The lines and their buttons are listed under Features below.
- **Filter** — "Filter skills… (Enter)".
- **Toolbar** — "sort: Name" / "sort: Status" (press to open a one-row
  strip of Name / Status with ✓ on the active one and pick directly;
  Status puts needs-review skills first) and "Import…".
- **Rows** — one per skill: **⚠ name** (blocked — needs review before use)
  or **✓ name** (usable), with a dimmer description line underneath.
- **Empty state** — "No skills yet — use Create ▸ New skill in the rail,
  or Import… above." (a filter with no matches shows "No skills match your
  filter." instead).

Clicking a row opens the **editor**, with the **Trust** panel below it —
the canvas scrolls, so the trust panel may sit below the fold.

## Features & controls

### Trust header (list)

| Line | Action button |
|---|---|
| "Skill trust isn't set up — set it up to review and use skills." | **Set up skill trust** |
| "Skill trust needs to be set up again after an update." | **Set up skill trust** |
| "Skill trust is temporarily unavailable — try again." | **Retry** |
| "Skill trust is locked for this session." | **Unlock** |
| "Skill trust can't be verified — set it up again." | **Set up skill trust** |
| "N skill needs review before use." / "N skills need review before use." | **Review** (opens the first blocked skill's editor) |
| "Skill trust: ready." | — |

A standalone **Reset skill trust…** button appears next to the header for
the locked and set-up-again states. It two-step confirms with "Reset skill
trust? Every skill will need re-approval. Your skills are not deleted."
(**Reset** / **Cancel**).

### Importing

**Import…** opens an inline row: an input with placeholder "SKILL.md file
or skill folder path… or GitHub/zip URL", plus **Browse…** (pick a
SKILL.md file), **Browse folder…** (pick a skill folder), **Import**, and
**Cancel**. A `http(s)://` value fetches the skill from that URL.

Only one skill import runs at a time. While Chatbook shows
`Inspecting/importing…`, the path, Browse, Browse folder, Import, and Cancel
controls are disabled. Library navigation remains available: leaving the
Skills list does not cancel filesystem or network work, and returning shows
the accepted import's current state or actual result. A forced repeat submit
is refused with `An import is already in progress.` The result stays available
until you choose **Cancel**, open **Review…**, or begin a new import draft.

- Success: `Imported "name" · re-review it in the trust panel`, with a
  follow-up button `Review "name"…` that jumps straight to its trust
  panel. **Every import lands trust-pending** — it cannot run until you
  review and approve it.
- Failures you may see: "Could not find that file or folder.", "No
  SKILL.md found in that folder.", "Could not read that file.",
  "Unsupported file type.", "Skill import is unavailable.", and
  `Skipped — a skill named "name" already exists.`

### Project skills (`.SKILLS/`)

A project with a `.SKILLS/` (or `.skills/`) folder at its root can offer its
skills for import automatically instead of one-at-a-time manual imports. The
convention: each skill is either a subdirectory containing a top-level
`SKILL.md` (skill name = the directory name) or a loose `*.md` file (name
derived from the filename). Anything else in the folder — a subdirectory
without `SKILL.md`, a non-markdown file — is listed as skipped with a
reason, never silently ignored. A symlinked `.SKILLS/` directory, or a
symlinked entry inside it, is refused.

Chatbook offers to import from `.SKILLS/` at two moments, never silently:

- **App startup**, when launched from inside such a project (or a
  subdirectory of one — the search walks upward looking for `.SKILLS/`,
  stopping at the first ancestor containing `.git`, at your home folder, or
  at the filesystem root).
- **After creating a workspace** bound to a folder that contains `.SKILLS/`
  (the folder row in the "New Workspace" dialog shows "— contains N project
  skill(s)" while you're adding it).

The prompt lists each discovered skill with a checkbox — **new** entries
checked by default, entries that match an **already installed** skill name
left unchecked (an existing skill is never silently overwritten), and
**invalid** entries (a bad name, or a file that can't be read) shown
unselectable with a reason. Unparseable frontmatter is not one of those
reasons — a skill file whose frontmatter can't be parsed still imports
fine, just with an empty description (matching how a manual import treats
the same case). **Import selected** runs the same importer as a manual
import; **Not now** declines for this launch only — you're asked again only
if the project's skill set actually changes (a new or removed skill file
changes its fingerprint); **Never for this folder** declines permanently for
that project. Declining or importing from either trigger (startup or
workspace creation) is remembered for both.

**Every import still lands trust-pending, exactly like a manual import** —
the prompt states this up front ("Imported skills require a one-time trust
review in Library ▸ Skills before they can run") because a project-imported
skill is otherwise indistinguishable from any other skill in the list. The
result view's **Review in Library ▸ Skills** button brings you straight
here to review and approve them (see [Trust panel](#trust-panel) above).

The whole feature can be turned off — no scanning, no prompts, at either
trigger — with `[skills] project_skills_prompt_enabled = false` in
`config.toml`; it defaults to on. This does not affect the manual
**Import…** row above, which is always available regardless of this
setting.

### Editor

The editor opens in **Basic** by default. **Show advanced** reveals the
technical controls; **Show basic** returns to the concise view. The choice is
remembered for this profile. Switching views does not rebuild the draft, so
text, undo history, focus, and scroll position stay intact.

```text
Basic                                  Advanced
┌──────────────────────────────┐       ┌──────────────────────────────┐
│ Name                         │       │ Basic fields remain mounted  │
│ Description                  │       │ Run context                  │
│ Instructions                 │       │ Restrict tools   [Filter…]   │
│ You can invoke       [on/off]│       │ ┌ SelectionList (bounded) ┐  │
│ Agent can invoke     [on/off]│       │ │ [x] calculator          │  │
│ Argument hint (when useful)  │       │ │ [x] old-tool unavailable│  │
│ Trust summary / safe action  │       │ └──────────────────────────┘  │
│              Show advanced ▶ │       │ Supporting files / metadata  │
└──────────────────────────────┘       │ ◀ Show basic                  │
                                       └──────────────────────────────┘
```

Basic fields:

- **Name** — editable while creating. Existing Skills cannot be renamed;
  create a new Skill instead.
- **Description** — optional list summary.
- **Instructions** — the Skill's `SKILL.md` body.
- **You can invoke** and **Agent can invoke** — independent choices. Turning
  both off makes the Skill reference-only; the editor says so explicitly.
- **Argument hint** — shown when user invocation is enabled.
- **Trust** — a healthy Skill is one compact line plus **View details**.
  Changed, blocked, quarantined, script-enabled, or otherwise actionable
  safety states expand automatically in Basic and Advanced.

Advanced adds:

- **Run context** — inline in this conversation or forked to a sub-agent.
- **Restrict tools** — a bounded searchable checklist of eligible builtin and
  enabled local tools. This narrows what the Skill may use; it never grants a
  permission. Imported unavailable names remain visible and selected. Merely
  opening, filtering, or switching views does not rewrite the stored ordered
  allowlist, including duplicates and unknown names.
- **Supporting files** and technical warnings.
- **Imported model metadata** — read-only and shown only when present; it is
  preserved when saving but is not a runtime model selector.

Warnings appear above the actions when they apply: `Name shadows a
built-in command/tool ("name") — it will not be invocable as /name or as
an agent tool.` and `Saving marks this skill "needs review" — re-approve
it in the trust panel after saving.`

Actions follow the draft lifecycle instead of showing one permanent toolbar:

| State | Available actions |
|---|---|
| New | **Save skill**, **Cancel** |
| Saved, unchanged | **Back to list**, **More actions** |
| Saved, changed | **Save changes**, **Discard changes** |
| Changed elsewhere | **Reload** |
| Delete confirmation | **Delete**, **Cancel** |
| Saving/deleting | Progress plus a readable unavailable reason |

**More actions** contains **Delete** only for a clean saved Skill. Esc closes
the disclosure before leaving the editor. Delete confirms inline with
`Delete "name"? This removes the skill's directory and cannot be undone.`
(naming the supporting files too, when it has any). Save status lines:
`Saved. Review trust before using this Skill with the agent.`, "A skill with
this name already exists.", "Skill name must use
lowercase letters, numbers, and hyphens.", "This skill is blocked by trust
review — approve it in the trust panel before saving.", or "Couldn't save
this skill. Try again." If the skill changed elsewhere while you were
editing, a banner offers one way out: "This skill changed elsewhere —
Reload discards your edit and refetches it." with a **Reload** button.
Leaving with unsaved edits is refused: "Unsaved skill changes — Save or
Discard changes first."

### Trust panel

Below the editor, the **Trust** section shows the skill's state on one
line — "Trust: trusted", "Trust: not initialized", "Trust: locked",
"Trust: changed since trusted baseline", "Trust: new untrusted file",
"Trust: trusted file missing", "Trust: manifest cannot be verified",
"Trust: unsupported file path" — with the changed file names in
parentheses when something differs from the trusted baseline.

- **First run** — before anything else works you see: "Local skill trust
  hasn't been set up yet. Set a trust passphrase to start reviewing and
  approving local skills — current local skill files become the trusted
  baseline." **Set up skill trust** opens the "Set Up Local Skill Trust"
  dialog, which asks for a new passphrase twice. Later sessions unlock
  with the "Unlock Local Skill Trust" dialog instead.
- **Review changes** — captures the changed files and shows their current
  content, one `── filename ──` block per file (binaries show
  `(binary file — N bytes, sha256 …)`; removed files show `(deleted — no
  longer on disk)`). The preview caps at 20 files and 4,000 characters per
  file; anything past a cap ends with "… truncated (N chars total) — open
  the file on disk to read the rest." or "… N more files omitted — open on
  disk to review."
- **Approve** — enabled only after a review is captured. It asks for your
  passphrase ("Approve Reviewed Skill Version": "Enter the local skill
  trust passphrase to make the reviewed files the new trusted baseline.").
  If the files changed again in between: "Skill files changed after the
  review was captured, so it was discarded. Press Review changes again,
  then Approve."
- **Unlock** — enabled only while trust is locked for the session.
- **Scripts** — the panel states either "Scripts: you are asked to confirm
  each time this skill runs a script." or "Scripts: this skill may run its
  bundled scripts without asking. Any change to its files cancels this
  automatically." A standing grant (given from Console's confirm card) can
  be withdrawn here with **Revoke script access**.

## Common tasks

1. **Create a skill.** Rail **Create ▸ New skill**, type a name (lowercase
   letters, numbers, hyphens), write the Instructions, **Save skill** (or
   Ctrl+S). Then
   scroll to the Trust panel, **Review changes**, **Approve**, and enter
   your passphrase — now `$name` runs in Console.
2. **Import from a GitHub URL.** In the list, click **Import…**, paste the
   URL into the input, click **Import**, then click the `Review "name"…`
   follow-up to review and approve it.
3. **Set up skill trust.** Click **Set up skill trust** (in the list
   header or a skill's Trust panel), choose a passphrase in "Set Up Local
   Skill Trust", and enter it twice. Existing skill files become the
   trusted baseline.
4. **Review and approve a changed skill.** Open the skill (the list marks
   it **⚠**; the header's **Review** button jumps to the first one), press
   **Review changes**, read the per-file blocks, press **Approve**, and
   enter your passphrase.
5. **Revoke script access.** Open the skill, scroll to Trust, check the
   "Scripts:" line, and press **Revoke script access** — Console goes back
   to asking on every script run.

## Keyboard & commands

| Key | Action |
|---|---|
| Ctrl+S | Save — only when the open Skill is new or has changes |
| Esc | Close More actions first; otherwise return to the Skills list |
| Tab / Shift+Tab | Move through the current Basic or Advanced controls |
| Arrow keys / Space | Move and toggle items in the Advanced tool checklist |

Both keys act only inside the skill editor; elsewhere in Library they pass
through (Esc also cancels the passphrase dialogs).

## Related settings & docs

- [Agent runs & tools](../console/agent-runs-and-tools.md) — running
  skills with `$name`, listing them with `/skills`, and the agent-side
  install/script confirm cards. When Console refuses a skill as untrusted,
  this panel is where you approve it.
- [Skills script execution](../../Features/Skills-Script-Execution.md) —
  deep dive on how bundled scripts run and are sandboxed.
- [Library](../library.md) — the surrounding screen; [Prompts](prompts.md)
  are the simpler cousin (inserted text, no trust or execution).
- Trust itself lives in a local, passphrase-protected store on this
  machine, not in `config.toml`. The one config key this panel's flow does
  read is `[skills] project_skills_prompt_enabled` (default `true`) —
  see [Project skills](#project-skills-skills) above. Guide index:
  [index](../index.md).

## Quirks & troubleshooting

- **Renaming isn't supported.** The Name field is locked on existing
  skills — create a new skill and delete the old one instead.
- **Every import needs review**, even one you wrote yourself on another
  machine — imports always land trust-pending, and saving any edit marks
  the skill "needs review" again. This is by design.
- **Trust is per-machine and passphrase-based.** Approvals don't travel
  with the skill files. If you forget the passphrase, **Reset skill
  trust…** starts over — every skill drops back to needing re-approval,
  but "Your skills are not deleted."
- **The review preview is not a diff** — it shows current file content
  as-is (the trust store keeps fingerprints, not old text), capped at 20
  files / 4,000 characters each.
- **"Trust: manifest cannot be verified"** — the panel says "The local
  skill trust manifest can't be verified, so this skill stays blocked.
  Reset skill trust to start over -- your skills themselves are not
  touched."
- **"Trust: unsupported file path"** — the skill has nested folders or
  links; the panel names the on-disk location to open, flatten, and
  re-check.
- **Model override does nothing yet** — it is kept only so SKILL.md files
  round-trip without losing the field.
- **Skills imported from a project's `.SKILLS/` folder are quarantined like
  any other import** — a fresh `$mention` of one in Console is refused with
  a pointer back here until you review and approve it; the import prompt's
  header says this explicitly, but it's easy to miss.
- **`.SKILLS/` scanning is opt-out, not opt-in** — set `[skills]
  project_skills_prompt_enabled = false` in `config.toml` if you don't want
  Chatbook checking project directories for skill folders at startup or on
  workspace creation.

—
*Verified against dev @ bd05a692a — 2026-07-31*

*Verified against feat/library-queue-batch @ 0662e09f5 — 2026-08-11
(task-14902: the list's sort control converged on the Library
chooser-strip pattern (press → Name / Status with ✓ on the active one,
direct pick, Escape cancels); the editor's three two-state switches stay
one-press toggles with their full option set now on the label —
"User can invoke: ✓ yes ⇄ no" — so the option space is on screen.)*

*Verified against feat/project-skills-import @ 964cb04df — 2026-08-18
(task-18705: documented the new "Project skills (`.SKILLS/`)" convention —
per-project discovery at startup and workspace creation, the fingerprint
gated prompt ledger, the quarantine/trust-review expectation, and the
`[skills] project_skills_prompt_enabled` kill-switch.)*

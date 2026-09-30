# Guardian — supportive awareness of your own typing patterns

Guardian is an **opt-in** companion that watches one thing: the prompts **you
type into the Console**, before they are sent. You define pattern rules
("late-night work", "doomscrolling", …); when a rule matches, Guardian
surfaces a humane awareness notice — and, if you asked for it, escalates from
a gentle note to masking the match (`redact`) or holding the send (`block`)
after repeated hits. A daily trend task summarizes sustained patterns
(fixation on one topic, repetitive volume) in the post-visit summary.

**tldw is not a mental health service.** Guardian is supportive awareness
software, not care, diagnosis, or treatment. If you or someone you know is
struggling or in crisis, help is available:

- 988 Suicide & Crisis Lifeline (US): call or text 988
- Crisis Text Line: text HOME to 741741
- SAMHSA National Helpline (US): 1-800-662-4357
- IASP Find a Helpline (international): findahelpline.com

These same resources appear verbatim wherever a crisis-flagged rule fires.

Guardian is **off by default**. While it is off, nothing runs, nothing is
recorded — and no Guardian database file even exists on disk.

## Enabling Guardian

Open **Settings** (**F9**) ▸ **Domain Defaults** ▸ **Guardian** and press
**Enable Guardian**. Checks start with your next Console message. The same
page shows your rules and edits them. Disabling Guardian there stops all
checks and recording; your rules and history stay on disk and survive a
re-enable — disabling never deletes anything.

The `config.toml` escape hatch also works (`[guardian] enabled = true`) but
bypasses the cooldown protection below — a config-file disable during an
active cooldown is detected and logged (`guardian_cooldown_bypassed`).

## Rules

Each rule has:

| Field | Meaning |
|-------|---------|
| `name` | Human label, shown in notices. |
| `topic` | The topic key hits are counted under (e.g. `late_night_work`). |
| `pattern` | Regex matched against your typed prompt. |
| `except_patterns` | Regexes that cancel a match (e.g. research vocabulary). |
| `action` | `notify` → surface a notice; `redact` → mask the match before sending; `block` → hold the send. |
| `severity` | `info` / `warning` / `critical` — notice styling. |
| `notification_frequency` | How often the notice is *surfaced* (`every_message`, `once_per_conversation`, `once_per_session`, `once_per_day`). Hits are always recorded and counted regardless — Guardian tracks behavior, not annoyance. |
| `display_mode` | Where the notice lives: `inline_banner`, `post_visit_summary`, or `silent_log` (record only). |
| `escalate_session_threshold` | Hits within one session that bump the action one rung (notify → redact → block). |
| `escalate_window_threshold` / `escalate_window_days` | Same, counted across a multi-day window. |
| `cooldown_minutes` | After an escalated `block`, how long the escalated hold lasts. |
| `feeds_discovery` | Opt this rule's topic counts into Dreams discovery (see below). |
| `is_crisis` | Crisis flag — see below. |

Three rules ship as editable examples: **Crisis awareness (self-harm)**,
**Doomscrolling awareness (demo)**, and **Late-night work (example)**. Edit or
delete them freely; empty visits mint nothing.

### Crisis-flagged rules

A rule flagged `is_crisis` always surfaces a notification with the crisis
resources and the disclaimer above. It **cannot** hold or escalate to
`redact` or `block` — masking distress would hide it from the model that
should respond to it, and blocking someone mid-crisis from their
communication tool is the opposite of the feature's purpose. It also can
never feed Dreams discovery. The store rejects such writes outright, and the
editor fixes the controls so honest clicks cannot attempt it.

### Cooldowns (anti-impulsive-disable)

When a rule escalates to `block` and has a cooldown, that hold binds: while
it is active, both the rule's deactivation and the Guardian toggle itself
refuse with a "available in N min" notice — you can still edit anything
else. The `config.toml` escape hatch still works during a cooldown; using it
to disable Guardian mid-cooldown is logged.

## Trends

Once a day, Guardian's trend task reads recorded hits and surfaces two kinds
of notice in the post-visit summary: **fixation** (one topic dominating your
messages — at least `fixation_min_hits` hits in `fixation_window_days` days,
making up more than `fixation_share_threshold` of all topic hits) and
**repetitive volume** (over `doomloop_hits_per_day` hits per day on one
topic). Notices state counts and suggest mitigations; they never quote your
messages. Rows the analyzer itself produced are excluded from its input, so
trends cannot feed themselves.

## What is recorded, and what leaves the machine

- Guardian stores **message digests, topic labels, and timestamps** — never
  message text and never matched spans. Matched text appears only
  transiently in a live inline notice.
- Everything is stored locally in the Guardian SQLite database; alerts prune
  at `alert_retention_days` (default 180). Visit summaries persist.
- Checker or storage errors never eat a send — Guardian fails open and
  surfaces one notification per error instead.

## The Dreams gate

Guardian and [Dreams](dreams.md) can work together: a rule with
`feeds_discovery` on contributes its topic counts to Dreams' interest
profile — so subjects you keep returning to gently steer your daily digest.
The boundary is strict:

- Only rules **you explicitly opted in** with `feeds_discovery` contribute.
- What crosses is an **aggregate count per topic** — never message text.
- **Crisis-flagged rules can never opt in** (the store refuses the
  combination), so crisis-adjacent text can never become an outbound search
  query.
- A disabled Guardian feeds Dreams nothing.

## Related settings & docs

- Settings ▸ Domain Defaults ▸ Guardian (enable toggle, rules editor).
- `config.toml` `[guardian]`: `enabled`, `alert_retention_days`,
  `fixation_share_threshold`, `fixation_window_days`, `fixation_min_hits`,
  `doomloop_hits_per_day`.
- [Dreams](dreams.md) — the discovery side of the aggregate feed.
- Decision record: `backlog/decisions/204-guardian-local-self-monitoring.md`.

—
*Verified against feat/guardian-local (Tasks 1–3, ADR-204) — 2026-09-29.*

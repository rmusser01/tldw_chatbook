# Dreams — a daily discovery digest built from your interests

Dreams is an opt-in research companion that runs while the app is open. When
a cycle fires (by schedule, or at startup as a catch-up), it reads
what you have actually been touching — note keywords, media-library
keywords, and (if you use it) your Personal Context profile — merges them
into an interest profile, turns that into search queries, and writes one
short "story" per interesting find. Reactions you give to stories nudge the
next cycle toward (or away from) similar finds.

Dreams is **off by default**. Enable it in Settings.

## Enabling Dreams

Open **Settings** (**F4**) ▸ **Domain Defaults** ▸ **Dreams** and press
**Enable Dreams**. The change takes effect on the next cycle or at boot.
What switching it off again does — and does not — stop is described in
[Switching Dreams off and on again](#switching-dreams-off-and-on-again).
The same page lets you:

- pick a **provider** and **model** for Dreams alone — leave both unset to
  follow your chat defaults (Settings ▸ Providers & Models);
- set a **region**. Dreams sends it to the model with every
  query-synthesis and story prompt, the model may write it into the
  searches Dreams runs, and it fills `{region}` in a tracked question's
  search. A story is labelled with it as its location only when it
  literally appears in the story's source text — a location is never
  guessed;
- manage **your topics**: add a topic you care about with a weight
  (default 0.5), or remove one. Topics you add are yours — the signal
  refresh at the start of each cycle never overwrites them. Goals are
  edited from any Dreams story (press **g**), not here.

A topic, a goal and the region are each one line of plain text of at most
200 characters. They are counted as Unicode code points: a letter or a
simple emoji is one, while a flag, a skin-toned or a family emoji is
several (a subdivision flag such as Scotland's is seven). An accent
typed as a separate mark counts together with its letter where Unicode
has one character for the pair, and as one more where it has not. The
rule is about
what the text is, not what it mentions — `JavaScript: The Good Parts`,
`<b>` or `[b]` are fine: Dreams stores them as typed, shows them as
typed, and never runs them as markup or script. One thing does change
how they look. In the titles Dreams writes for you in other parts of the
app — the source and alert of a tracked page in Watchlists, an event
reminder in Schedules, a tracked-update notification — square brackets
are written as parentheses, because those screens read `[…]` as
formatting. What is refused — the notice says which of these it was —
while your typed text stays in the field to fix:

- text with nothing to read in it: blank, punctuation only, or made only
  of invisible characters. A single emoji or symbol is enough;
- line breaks and control characters, including Unicode's line and
  paragraph separators;
- hidden text-direction controls, which make text display in a different
  order than it is stored and sent;
- other invisible formatting characters that change what the text is
  without showing. The invisible joiners and direction marks that emoji,
  flags, and right-to-left, Persian or Mongolian text need are fine;
- text longer than 200 characters.

A few invisible characters that are not part of what you mean are simply
left out instead of refused: zero-width spaces (Thai and Khmer keyboards
type them between words), soft hyphens, and the stray marks a paste can
carry along. A blank braille cell or a Hangul filler is saved as a space.

Two spellings of the same accented phrase (the accent typed as part of the
letter, or as a separate mark) are saved as one topic from now on. A
topic that an earlier version saved in the separate-mark spelling keeps
that spelling, so adding it again makes a second topic beside it; remove
the old one.

A `region` in `config.toml` that breaks these rules is ignored — Dreams
behaves as if no region were set — until you correct it. Settings says so
under the Region field ("The saved region is not being used"), and the
goals window shows the region as "(not set)".

The remaining keys (budgets, cadence, track caps) are shown read-only on
that page; editing them is a `config.toml` escape hatch.

### Switching Dreams off and on again

While Dreams is off it starts nothing: no cycle, no check of a tracked
question, no search and no model call. Two limits on that are worth
knowing:

- **Work already under way finishes its current step.** A cycle or a
  tracked check that is running when you switch Dreams off stops at its
  next step. The one search or model call already in flight completes —
  and still counts against the day's budget — but no further one starts,
  and a check sends no update. Stories the cycle had already written are
  kept. If it had not written one yet, nothing is recorded for that day
  and the interrupted run does not count as an attempt, so it does not
  push your next cycle back when you switch Dreams on again.
- **Your own Watchlists sources keep running.** A source you added to
  Watchlists yourself is fetched by Watchlists exactly as before, even
  when a Dreams watch is attached to it. Dreams never switches off
  something that is yours.

Switching off pauses what your tracked watches set up outside Dreams,
and Dreams notes on each watch exactly what it paused:

- the watchlists subscription behind a tracked page, when Dreams created
  that subscription;
- Dreams' own change alert (`Change: <title>`) on every tracked page —
  including one attached to a source of yours, so that source keeps
  running but no tracked-page alert arrives from Dreams while it is off;
- an event watch's reminder.

Switching back on restores those things and nothing else. A
subscription, alert or reminder you had switched off yourself stays off,
and a reminder you moved to another time keeps your time. A watch whose
event passed while Dreams was off is retired instead of resumed, and a
tracked page whose address is no longer public stays off (see
[Tracking](#tracking)).

The switch always runs to the end: changing another Dreams setting, or
leaving Settings straight away, does not interrupt it. Dreams also
compares its watches with the saved switch every time the app starts, so
turning Dreams off or on by editing `config.toml` (or removing the
`[dreams]` section altogether) has the same effect at the next launch,
and a pause or resume that could not finish — a write failed, or the app
was closed part-way — is completed then. At that launch the scheduler
waits for the pause to land before it runs anything, for up to ten
seconds. After that, unrelated scheduled work can continue, but pending
Dreams-created page checks and linked event reminders stay held until their
saved pause and provenance work is verified. If ownership cannot be read,
watchlist checks remain held too; an error does not authorize a fetch. The one
thing a start does not
retry is a tracked page that was left off because its address is not
public (see [Tracking](#tracking)). On a profile restored from a backup
and still waiting for your review, Dreams cycles, tracked checks and this startup
reconciliation remain inactive until the Dreams owner is approved. Reconciliation
also requires approval for the Watchlists and scheduler stores it changes.
Stored enabled settings do not provide that approval. Legacy Watchlists URLs
without recorded user-entered provenance remain untrusted on their first fetch,
even when Dreams is disabled; explicitly recorded user-entered URLs retain their
existing access.

### The config.toml escape hatch

You can also enable Dreams by hand in your `config.toml` (the same file
Settings edits; `~/.config/tldw_cli/config.toml` by default) — useful for
scripted setups or the keys the Settings page only displays:

```toml
[dreams]
enabled = true
# Optional: pin the provider/model instead of reusing your persisted
# chat defaults (Settings ▸ Providers & Models):
# provider = "openai"
# model = "gpt-4o-mini"
# Optional: sent with the query-writing prompt and every story prompt, and
# usable in searches; a story is labelled with it only when it appears in
# the story's source:
# region = "Seattle"
```

Everything else has a sensible default. The full key list, verbatim from
the defaults:

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` | Master switch; Dreams starts nothing when false (see [Switching Dreams off and on again](#switching-dreams-off-and-on-again)). |
| `provider` | none | Pin a provider; falls back to your chat defaults. |
| `model` | none | Pin a model; falls back to the provider's remembered model. |
| `stories_per_cycle` | `5` | Stories one dated collection fills up with. |
| `queries_per_cycle` | `3` | Search queries synthesized per cycle. |
| `exploration_slots` | `1` | How many of the queries the model is asked to aim at adjacent (non-profile) topics. A request in the prompt, not a reserved slot. The fallback queries used when the model is unavailable always contain exactly one such line (`surprising adjacent to <your top topic>`), whatever this is set to. |
| `search_engine` | `"duckduckgo"` | Web search engine used. |
| `web_search_enabled` | `true` | Kill switch for web search; false means LLM-knowledge stories. |
| `region` | `""` | Your region. Sent with every synthesis and story prompt, usable in searches, and substituted for `{region}` in tracked questions; a story is labelled with it only when it appears in the story's source text. |
| `cadence_hours` | `24` | Minimum hours between cycles. |
| `catchup_enabled` | `true` | Run today's cycle at startup when the cadence says one is due. |
| `watchlist_freshness_hours` | `48` | How fresh a watchlist item must be to join the pool. |
| `seen_item_ttl_days` | `90` | How long a surfaced URL stays deduped. |
| `max_searches_per_day` | `30` | Daily search budget. |
| `max_llm_calls_per_day` | `60` | Daily LLM budget (query synthesis, stories, and tracked-question judge calls). |
| `tracked_item_cap` | `20` | How many tracked pages/questions may be active at once. |
| `track_min_check_interval_hours` | `12` | Floor between two checks of the same tracked item. |
| `track_quiet_retire_count` | `14` | Consecutive unchanged checks before a quiet retire. |

## Where the stories live

Dreams stories surface on the **[Artifacts](artifacts.md)** screen
(**Ctrl+6**) as rows of the **Dreams** type — one per story, plus a
status row for a cycle that failed outright. Selecting a row opens the
story detail: title, source URL, body, and a "what we'll look for" preview
of the next cycle's queries.

Known limitation: the Artifacts screen has since been folded into
Library, and at the moment the app's own navigation may not lead to this
list. Stories are still written and kept; until that is fixed, they —
and the actions below, which all start from an open story — may not be
reachable.

## The actions

From an open story you can:

- **Dive deeper (d)** — stage the story into Chat and dismiss.
- **Keep (k)** — mark the story. A kept story is listed first among its
  day's stories, and keeping it counts as a positive signal. Dreams never
  deletes stories, kept or not.
- **Export (e)** — write a Markdown stub to `~/Documents/tldw_exports/dreams`.
- **Ingest (i)** — submit the story's URL to your read-it-later capture
  queue. Only offered for real web (http/https) sources; stories generated
  purely from LLM knowledge have nothing to ingest. A story whose address
  is on a private network, or does not resolve, is refused with a notice
  and nothing is submitted.
- **Track this (t) / Untrack (u)** — watch (or stop watching) the story's
  page for changes; see [Tracking](#tracking) below.
- **Watch question (w)** — watch the discovery query itself instead of the
  page: Dreams re-runs it as a search on the check cadence and notifies you
  when a judged change lands (see [Tracking](#tracking)).
- **More like this (m) / Less like this (l)** — record feedback that the
  next cycle's interest profile picks up. Dive deeper, Keep, Ingest,
  Track this and Watch question count as positive signals too; Export
  is neutral. When the story was found by a
  search Dreams ran for one of your goals, that feedback also steers that
  goal's query angle — how Dreams phrases its searches for the goal —
  never the goal itself. Only that goal is steered: a story that merely
  shares words with a goal, or was found some other way, changes no goal.

## Tracking

From an open story, **Track this (t)** puts the page under watch: Dreams
creates (or adopts) a watchlists subscription for the URL, joins it to a
shared "Dreams Tracked" watchlist, and pins a change alert — every new
item the page produces lands in the Watchlists **Notifications** pane.

**The event reminder.** Tracking an event-dated story also schedules one
reminder a week before the event, at 9:00 in the morning by your
computer's local time. There is one reminder per story and event date,
however the story is watched (tracking the page and watching its
question share it) and however often you untrack and track it again:
once it has fired — or you have deleted it — it is never made again for
that date, and one you switched off or moved stays as you left it. It is
re-made only when the event's date changes. When a story is both tracked
and watched and one of the two ends — you untrack it, or it retires as
quiet — the reminder stays with the one that is left, at the time you
last gave it. When the story's last watch ends, a reminder that has not
fired yet is disarmed, and tracking the story again arms one for the
event again. If a story's event date can no longer be read, the reminder
still pending for the old date is switched off.

**Public addresses only.** Dreams only watches a page at a public web
address, and it checks that three times: when you track the story (a URL
that points at a private network, a loopback address, or a host that
does not resolve is refused with a notice), on every later fetch of a
page Dreams added (if the address stops being public the fetch is
refused and shows as a failed check in Watchlists — nothing is requested
from the private address), and before Dreams turns such a page back on
after you re-enable Dreams (a page that is no longer public stays off,
and its **Tracked** row says "paused" with the reason; Dreams asks again
the next time you switch Dreams on, or when you track the page again —
not on every start. When the reason is only that the address could not
be looked up at that moment, Dreams also asks again by itself, at its
next daily cycle). A page you tracked with an earlier version of the
app is brought under the same rule the next time the app starts. A
source you added to Watchlists yourself is yours: Dreams can attach a
watch to it, and it is fetched exactly as before, private address or
not. `[web_security] allowed_hosts` in `config.toml` admits a host you
trust.

**A source that is switched off.** Tracking a page switches its source
back on only when Dreams itself switched it off — when you untracked the
page, or while Dreams was off. A source that is off for any other reason
(for example one you created in Watchlists as inactive) is left alone,
and Dreams says so: Watchlists has no control that switches an existing
source back on, so to track that page, delete the source in Watchlists
and track again. A watch
that was paused after repeated failed checks goes back to work when you
track its story again; no second watch is added.

**Watch question (w)** instead tracks the question that found the story:
the story's discovery query is re-run as a search on the check cadence,
and one judge call decides whether the results materially changed — an
unchanged digest never spends the judge. A changed verdict surfaces as a
row in the **Tracked** group above the Dreams rows; selecting a Tracked
row opens the story it came from (or a summary notice, when its origin
story is gone). Both kinds of watch retire the same way (see below).

Tracked watches retire on their own: an event-dated watch retires one
day after its event has passed, and any watch that comes back unchanged
`track_quiet_retire_count` checks in a row (default 14) retires as
quiet. A watch whose checks keep failing is paused after 3 errors
instead of spending budget forever. **Untrack (u)** retires a watch
manually — a subscription Dreams created is disabled (never deleted), so
its watchlist membership and alert rules survive; when two tracked
stories share one page, it keeps running until the last of them is
untracked or retires. On a story that is
both tracked and watched, **u** stops the page watch first and the
question watch on a second press; at most
`tracked_item_cap` watches (default 20) are active at once, and no
watch is checked more often than once per
`track_min_check_interval_hours` (default 12).

## Cost and privacy

- Each cycle makes up to `queries_per_cycle` web searches and one LLM call
  per story plus one for query synthesis, bounded by the two daily budgets.
  A check of a tracked question draws on the same two budgets (one search,
  and one judge call only when the results changed); each search or call
  is counted before it is made, so cycles and checks running at the same
  time cannot overspend a budget between them. The other side of
  counting first: if the app is closed, or a cycle or check is cut off,
  at the very moment one has been counted but not yet sent, that one
  stays counted — at most one search and one model call each time.
- When a cycle stops short of its plan — the search budget was used up
  by other Dreams work part-way, no model call was left for a story, a
  source failed — it records why with that day's collection. Nothing in
  the app shows those notes yet; they are kept in the Dreams database
  (the `degradation_notes` column of `dreams_collections`).
- What the model is sent: for query synthesis, your topic names, the
  text of every searchable goal together with its query angle, and the
  region; for a story, the candidate's title/snippet/URL, your topic
  names and the region; for a tracked question's judge call, the watched
  query and the current results' titles/snippets/URLs. A goal is
  searchable from the moment you add it; **s** in the goals list makes
  one private, and a private goal's text and angle are never sent. Never
  note bodies or Personal Context records either.
- When two or more story slots are available, a goal result gets one place if
  topic ranking would otherwise exclude every goal result. The remaining topic
  choices keep their order. This does not change your goal weights or daily
  budgets; a one-slot collection keeps its existing ranking.
- Everything is stored locally in the Dreams SQLite database.

Dreams' interest profile can also draw on [Guardian](guardian.md):
awareness rules you explicitly opt in with *feeds Dreams* contribute their
topic **counts** (never message text) to this profile — and crisis-flagged
rules can never opt in, so crisis-adjacent text never becomes a search
query.

—
*Verified against feat/dreams-phase-3 @ a5f7daa078 — 2026-09-28*

# Dreams — a daily discovery digest built from your interests

Dreams is an opt-in research companion that runs while the app is open. When
a cycle fires (by schedule, at startup as a catch-up, or manually), it reads
what you have actually been touching — note keywords, media-library
keywords, and (if you use it) your Personal Context profile — merges them
into an interest profile, turns that into search queries, and writes one
short "story" per interesting find. Reactions you give to stories nudge the
next cycle toward (or away from) similar finds.

Dreams is **off by default** and has no Settings panel in Phase 1: enabling
it is a `[dreams]` section in your `config.toml` away.

## Enabling Dreams

Open your `config.toml` (the same file Settings edits; `~/.config/tldw_cli/
config.toml` by default) and add:

```toml
[dreams]
enabled = true
# Optional: pin the provider/model instead of reusing your persisted
# chat defaults (Settings ▸ Providers & Models):
# provider = "openai"
# model = "gpt-4o-mini"
# Optional: only claimed when it literally appears in a story's source
# text — never guessed:
# region = "Seattle"
```

Everything else has a sensible default. The full key list, verbatim from
the defaults:

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` | Master switch; nothing runs when false. |
| `provider` | none | Pin a provider; falls back to your chat defaults. |
| `model` | none | Pin a model; falls back to the provider's remembered model. |
| `stories_per_cycle` | `5` | Stories one dated collection fills up with. |
| `queries_per_cycle` | `3` | Search queries synthesized per cycle. |
| `exploration_slots` | `1` | Queries reserved for adjacent (non-profile) topics. |
| `search_engine` | `"duckduckgo"` | Web search engine used. |
| `web_search_enabled` | `true` | Kill switch for web search; false means LLM-knowledge stories. |
| `region` | `""` | Location hint, only used when it appears in source text. |
| `cadence_hours` | `24` | Minimum hours between cycles. |
| `catchup_enabled` | `true` | Run today's cycle at startup when the cadence says one is due. |
| `watchlist_freshness_hours` | `48` | How fresh a watchlist item must be to join the pool. |
| `seen_item_ttl_days` | `90` | How long a surfaced URL stays deduped. |
| `max_searches_per_day` | `30` | Daily search budget. |
| `max_llm_calls_per_day` | `60` | Daily LLM budget (synthesis + stories). |

## Where the stories live

Dreams stories surface on the **[Artifacts](artifacts.md)** screen
(**Ctrl+6**) as rows of the **Dreams** type — one per story, plus a
status row for a cycle that failed outright. Selecting a row opens the
story detail: title, source URL, body, and a "what we'll look for" preview
of the next cycle's queries.

## The actions

From an open story you can:

- **Dive deeper (d)** — stage the story into Chat and dismiss.
- **Keep (k)** — mark the story so it survives future cycles.
- **Export (e)** — write a Markdown stub to `~/Documents/tldw_exports/dreams`.
- **Ingest (i)** — submit the story's URL to your read-it-later capture
  queue. Only offered for real web (http/https) sources; stories generated
  purely from LLM knowledge have nothing to ingest.
- **More like this (m) / Less like this (l)** — record feedback that the
  next cycle's interest profile picks up (plus dive/keep/export/ingest
  themselves count as positive signals).

## Cost and privacy

- Each cycle makes up to `queries_per_cycle` web searches and one LLM call
  per story plus one for query synthesis, bounded by the two daily budgets.
- Prompts carry only topic names, the region, and each candidate's
  title/snippet/URL — never note bodies or Personal Context records.
- Everything is stored locally in the Dreams SQLite database.

—
*Verified against dev @ 8cb20b929d — 2026-09-22*

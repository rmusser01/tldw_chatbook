---
target: new Chatbook threat-intelligence news-to-hunt workflow on latest dev
total_score: 21
max_score: 40
na_heuristics:
p0_count: 1
p1_count: 3
timestamp: 2026-08-27T03-28-21Z
slug: watchlists-collections-screen-py
---
Method: dual-agent (A: `/root/ux_assessment_a` · B: `/root/ux_assessment_b`)

## Planning disposition

This critique records the initial UAT state and is intentionally not rescored.
The approved remediation scope later made three dispositions explicit:

- briefing-to-hunt product integration is out of scope by user decision;
- the First Run findings and product User-Agent were already fixed on the
  reconciled latest-dev baseline and remain regression checks only;
- feed recovery, schedule receipts, bulk authoring/selection, Artifacts state,
  and skill/framework classification map to TASK-22865, TASK-22864,
  TASK-22866, TASK-613, and TASK-22867 respectively.

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of system status | 2 | Selection, schedule, import, and artifact states can contradict or disappear. |
| 2 | Match with the real world | 2 | Feed/watchlist language works; `SKILL.md` and raw HTTP errors expose implementation concepts. |
| 3 | User control and freedom | 3 | Back, Skip, Cancel, Esc, confirmations, and editable cadence are strong. |
| 4 | Consistency and standards | 2 | Detected/selected/committed states do not behave consistently. |
| 5 | Error prevention | 2 | Feed and repository compatibility are discovered only after the operation. |
| 6 | Recognition rather than recall | 2 | Context must be carried manually across Watchlists, Artifacts, Library, Console, and ATHF. |
| 7 | Flexibility and efficiency | 2 | Strong keyboard depth; weak bulk source and membership operations. |
| 8 | Aesthetic and minimalist design | 2 | Coherent terminal identity, but navigation and Artifacts controls are dense. |
| 9 | Error recovery | 2 | State is usually preserved, but ATHF/CISA errors do not provide a usable recovery path. |
| 10 | Help and documentation | 2 | F1 and empty-state copy help; schedule and package contracts are hidden at decision time. |
| **Total** |  | **21/40** | **Acceptable; substantial workflow improvement needed.** |

## Design Specificity Verdict

Chatbook's shell feels authored: local/server authority, trust states, the Console, and the source → run → read → artifact lifecycle are product-specific. The complete job does not. At the briefing-to-hunt boundary it becomes a generic feed reader beside a generic repository importer.

The deterministic scan returned `[]` for every target because it only supports web frontend formats, not Python/Textual or `.tcss`. This is a false clean. Browser DOM overlays were inapplicable; the review used live terminal captures, rendered SVG/PNG evidence, focused compositor tests, and source inspection instead.

## Overall Impression

The workflow starts confidently, succeeds at collecting and summarizing intelligence, then loses trust at the two moments that matter most: onboarding commitment and briefing-to-hunt handoff. The largest opportunity is to make “Prepare a defensible hunt” a first-class continuation of a completed briefing.

## What's Working

- The first-run wizard sets expectations well: five steps, Quick/Full paths, Back/Skip, and “change later” language reduce anxiety.
- Watchlists has unusually strong keyboard depth, explicit verbs, useful empty-state writing, visible counts, safe destructive flows, and honest local/server ownership.
- The generated briefing was genuinely useful: it synthesized 40 of 45 items, disclosed five overflow items, separated confidence from gaps, and produced a behavior-led hunt seed.

## Priority Issues

**[P0] ATHF cannot be installed through the requested GitHub skill-import path**

- Why it matters: Chatbook reports only `No SKILL.md found in that archive.` A first-time user cannot tell whether the URL is wrong, the repository is broken, or the product supports a different kind of integration.
- Fix: classify repositories before failure. If `AGENTS.md`, Python packaging, prompts, or hunt templates are present, say: “This is an agent framework, not a Chatbook skill package,” then offer reviewed routes to bind project instructions, install its CLI in a workspace, or create a trust-pending wrapper skill. Never auto-trust converted content.
- Suggested command: `$impeccable harden`

**[P1] The product drops briefing context at the highest-value handoff**

- Why it matters: the global Console handoff stages a Watchlists snapshot, not the completed briefing and citations; entity-level staging is explicitly unimplemented. The captured Artifacts area was also silently blank immediately after generation, despite the briefing being persisted.
- Fix: add **Prepare hunt in Console** on each completed briefing. Include briefing Markdown, cited item IDs/URLs, watchlist and source IDs, freshness, model/preset provenance, and a suggested LOCK prompt. During pane rebuilds, retain the old view or show Loading/Error/Retry rather than blank space.
- Suggested command: `$impeccable shape`

**[P1] First-run setup shows selections that are not committed**

- Why it matters: Ollama and a discovered model appeared selected, but Summary reported neither. Recovery required backwards focus to “Use this server” and manual model entry. At 100×24, the focused API-key field extended below the viewport.
- Fix: focus the detected-server commitment action, distinguish Detected from Selected, make Enter/Space commit radio choices, gate Next on a committed or explicitly skipped state, show an immediate read-back, and enforce focus visibility at 80×24 and 100×24.
- Suggested command: `$impeccable onboard`

**[P1] “Daily” lacks an operational receipt**

- Why it matters: `86400` seconds persisted, but execution requires the app to remain open, the global scheduler gate is hand-edited, and queue activation may lag. Users cannot see next eligibility, queue state, or last run.
- Fix: show `Daily · enabled · app must remain open · next eligible … · queue pending/active · last run …`; reload the scheduler after saving; expose the global switch in Settings.
- Suggested command: `$impeccable clarify`

**[P2] Source setup and membership do not scale**

- Why it matters: four feeds required four forms and four separate picker dismissals. Rapid repeated submissions could fail to persist unless paced, creating a power-user race risk.
- Fix: add bulk URL paste with validation/deduplication, multi-select “Add selected,” and “Create watchlist from selected sources.” Preserve the single-source form as the novice path.
- Suggested command: `$impeccable distill`

## Persona Red Flags

**Jordan — first-time user**

- The wizard's checkmarks describe visited steps, not reliably persisted configuration.
- Seven Watchlists sections provide no staged path such as Sources → Watchlist → Check → Brief → Hunt.
- `Preset`, `Artifacts`, and `SKILL.md` require product/developer vocabulary.
- The ATHF error does not answer “What should I do next?”

**Alex — power user**

- One-key reading and F6 traversal are excellent.
- Repeated feed and membership creation is one record at a time.
- Local shortcut behavior depends on which region owns focus, while the footer mostly advertises global commands.
- No observable scheduler queue or first-class briefing-to-hunt transfer exists.

**Sam — keyboard-only or low-vision user**

- At 100×24, focus can move below the visible wizard viewport.
- Three source filters depend on hover tooltips because labels do not fit.
- Placeholder-only Name, Feed URL, and repository URL inputs lose their labels after entry.
- Positive: important states generally use text rather than color alone, and Esc is consistently supported.

## Minor Observations

- Library exposes several different actions named “Import.”
- “Artifacts” combines briefings, scripts, audio, feed export, and feed serving; the label does not predict the core news-briefing task.
- CISA succeeded via direct HTTP but returned 403 in Chatbook, suggesting feed-request compatibility; the UI should offer feed-specific recovery instead of raw transport prose.
- ATHF's non-interactive initializer silently selected Splunk/CrowdStrike, its optional-dependency error suggested reinstalling the base package, generated a broken relative research link, and exported hunt context with `environment: null` and `research: []`.

## Questions to Consider

1. Is the primary post-briefing outcome “read,” or “start defensible work”? If it is the latter, should Prepare hunt outrank podcast/feed controls?
2. Should repositories with `AGENTS.md` be recognized as project-instruction workspaces while `SKILL.md` remains a strict skill-package contract?
3. Does “daily while the app is open” meet the intended automation promise, or is a background/headless runtime required?
4. Could Watchlists expose a short guided activation path while preserving today's dense workbench as an expert view?

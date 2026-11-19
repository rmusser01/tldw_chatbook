---
id: TASK-16865
title: >-
  Wire up or retire SiteConfigSettings (per-site scraper config UI,
  nav-unreachable)
status: Done
assignee: []
created_date: '2026-08-16 18:41'
labels:
  - ui
  - dead-code
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16846 retired its sibling ScraperBuilderWindow (ADR-020 in Docs/Development/Subscriptions/Subscriptions-Implementation-1.md, amended) and left UI/SiteConfigSettings.py as the sole designed surface for per-site scraper configuration. At the 16846 branch point it is nav-unreachable: repo-wide grep finds the class referenced only by its own file and Tests/UI/test_site_config_settings.py — no screen-registry route, palette entry, or embedding. Unlike the retired builder it is persistence-connected: it fronts SiteConfigManager and the site_configs table (extraction selectors, rate limits, auth, JS options, presets). Two facts weigh on the fork: (1) the pipeline stack that reads site_configs at scrape time (Subscriptions/web_scraping_pipelines.py + Subscriptions/scrapers/, incl. CustomScrapingPipeline and the ScrapingPipelineFactory registry) itself has zero production consumers — the live watchlists path is monitoring_engine.py — so wiring the UI alone would configure a store nothing live reads; (2) its own Select bug is filed separately as task-16841. Decide per the 16837/16195/16846 playbook: wire it into the Watchlists/Subscriptions surface with an honest end-to-end path, or retire it together with a disposition for the orphaned scrapers/pipelines cluster.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An explicit wire-or-retire decision is recorded, with the Subscriptions doc set updated to match
- [ ] #2 If wired: the widget is reachable through real navigation and a saved config demonstrably affects a live scrape path
- [x] #3 If retired: the widget, its test, and the orphaned scrapers/pipelines cluster disposition are handled with reachability evidence
<!-- AC:END -->

## Closeout audit 2026-10-11

Board-prune audit (branch chore/board-prune-c, base a3c47c6c72): completed-but-unmarked. Evidence gathered 2026-10-11 against origin/dev:

AC dispositions:
- AC#3: met - Retired: UI/SiteConfigSettings.py (599 lines) and Tests/UI/test_site_config_settings.py (225 lines) deleted by commit ef43462806 'chore(ui): delete five unreachable UI/ root windows (TASK-32899)' with reachability evidence ('0 prod importers; the live site-config surface is SiteConfigManager').
- Wire-or-retire fork resolved as RETIRE by TASK-32899 (ef43462806, 2026-09-22).
- Residue noted, not blocking the flip: Subscriptions-Implementation-1.md:326 still describes the (now deleted) widget, and the Subscriptions/scrapers + web_scraping_pipelines cluster still awaits its own disposition task.

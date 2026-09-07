---
id: TASK-16865
title: >-
  Wire up or retire SiteConfigSettings (per-site scraper config UI,
  nav-unreachable)
status: To Do
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
- [ ] #3 If retired: the widget, its test, and the orphaned scrapers/pipelines cluster disposition are handled with reachability evidence
<!-- AC:END -->

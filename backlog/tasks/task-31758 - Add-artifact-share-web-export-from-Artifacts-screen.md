---
id: TASK-31758
title: Add artifact share web export from Artifacts screen
status: To Do
assignee: []
created_date: '2026-09-06 02:18'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user select local Chatbook artifacts on the Artifacts screen and temporarily host them as a small web page (aiohttp child process, staged snapshot) so recipients can browse and download exactly those bundles, with optional single shared username/password (HTTP Basic). Recipients import via the existing Chatbook Import Wizard. Design: Docs/superpowers/specs/2026-09-05-artifact-share-web-export-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Artifacts screen Share action opens a multi-select dialog over local chatbook records (records without on-disk bundles disabled),Starting a share stages an immutable copy plus bundle.zip and spawns a child server exposing only the staged files via HTML page, per-artifact download, bundle download, and index.json,When auth is configured every route requires Basic auth (401 + WWW-Authenticate, per-IP lockout 10 fails/30s); non-loopback bind without password requires typed confirmation,Share session is app-owned: survives navigation, single active share, stops cleanly via UI and on app exit; stale share dirs are swept at startup,Targeted tests pass (manifest/staging, server routes+auth, controller lifecycle, dialog, screen wiring) without a full-suite run
<!-- AC:END -->

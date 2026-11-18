---
id: TASK-41726
title: "The Settings provider card hides the endpoint contract's reason behind a generic save error"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- settings
- providers
priority: low
references:
- tldw_chatbook/UI/Settings_Modules/category_panels.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41700's review (round 16). Pressing s on a Settings provider card whose endpoint the contract refuses shows "Provider settings are invalid; review the endpoint and model." for every refusal (category_panels.py ~941), and the card's Endpoint row help checks only the KoboldCpp and MLX-LM rules (`typed_endpoint_refusal`), so a route or host refusal is not named there. Console, Discover models and the custom-endpoint editor already show the contract's reason.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The provider card's save error and Endpoint row help name the contract's reason for a refused endpoint
- [ ] #2 A pilot test pins the copy for a route refusal and a host refusal
<!-- AC:END -->

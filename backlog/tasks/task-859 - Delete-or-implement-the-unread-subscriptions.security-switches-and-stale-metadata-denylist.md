---
id: TASK-859
title: >-
  Delete or implement the unread [subscriptions.security] switches and stale
  metadata denylist
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-27 04:35'
updated_date: '2026-08-02 13:33'
labels:
  - security
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
config.py declares five keys under [subscriptions.security] (enable_ssrf_protection, max_redirects=5, verify_ssl_certificates, enable_xxe_protection, request_timeout). A per-key grep across tldw_chatbook/ finds exactly one occurrence of each -- the declaration itself. The behavior these keys claim to control is actually governed elsewhere and does not read them: SSRF protection is really get_cli_setting("web_security","enabled",True) (Utils/egress.py); the redirect cap is really the hardcoded egress.MAX_REDIRECT_HOPS = 10, not the configured 5; SSL verification is really a per-subscription DB column ssl_verify (Subscriptions/monitoring_engine.py); and production XML parsers prefer defusedxml when available but warn and fall back to the standard library when it is absent. The separate SecurityValidator.validate_xml_content helper has no production callers. None of that XML behavior reads enable_xxe_protection. An operator who sets enable_ssrf_protection = false in [subscriptions.security] to intentionally relax SSRF protection changes nothing -- egress.py's real gate is untouched -- which reads as a documented escape hatch that silently does nothing (and, in the opposite direction, an operator "hardening" these keys gets no behavior change either).

Separately, Subscriptions/security.py:79,85-89 defines its own BLOCKED_SCHEMES and METADATA_ENDPOINTS, each with exactly one occurrence in the codebase (the definition). The real enforcement path is Utils/egress.py:35-44 (reached via security.py:129-135's evaluate_url_policy). A side-by-side diff found this list is already missing two hosts/IPs that egress.py does enforce (100.100.100.200, fd00:ec2::254). Harmless today only because egress is strictly stronger, but it is an authoritative-looking denylist that enforces nothing and is already stale -- a trap for the next person who extends "the" metadata list in the wrong place.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The [subscriptions.security] keys either actually gate the behavior they claim to (SSRF protection, redirect cap, SSL verification, XXE protection, timeout), routed through the real egress.py/monitoring_engine.py controls, or are removed from config.py and the default TOML
- [ ] #2 Subscriptions/security.py's BLOCKED_SCHEMES and METADATA_ENDPOINTS are either removed in favor of Utils/egress.py's enforcement, or kept in sync with it (including the two currently-missing entries) and demonstrably consulted somewhere
- [ ] #3 A test confirms that toggling whichever [subscriptions.security] keys survive actually changes runtime behavior (e.g. redirect count, SSL verification), not just that the config value round-trips
- [ ] #4 No security-relevant metadata/scheme denylist exists in more than one place without a test tying them together
<!-- AC:END -->

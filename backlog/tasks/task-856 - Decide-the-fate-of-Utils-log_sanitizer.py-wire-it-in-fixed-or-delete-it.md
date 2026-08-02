---
id: TASK-856
title: 'Decide the fate of Utils/log_sanitizer.py: wire it in fixed, or delete it'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-27 04:35'
updated_date: '2026-08-02 23:14'
labels:
  - security
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The original audit found no production importers, but that premise is stale on
current `dev`: `Utils/log_sanitizer.py` now has three production consumers.
Ollama uses it both to redact successful API payloads and to transform model
names, Transformers uses it to transform local model display names, and the
subscription monitor uses it before interpolating a URL into a pruning
diagnostic. The module therefore needs an explicit, correct runtime boundary
rather than deletion.

Its rules remain wrong in both directions. The `claude-*` pattern treats an
Anthropic model-ID prefix as a credential, so
`claude-opus-4-20250514` becomes `***ANTHROPIC_KEY***`. Meanwhile real-shaped
`sk-ant-api03-*` and `sk-proj-*` credentials are only partially matched or
survive because the generic `sk-` character class stops at their embedded
hyphens. `sanitize_dict()` also has a second, drifting secret-field list that
misses real shipped names such as `x-api-key`, `cohere_api_key`, `api_token`,
`secret_key`, and `refresh_token`.

Credential redaction and display-name validation are separate operations.
Model names must receive bounded single-line input validation without being
rewritten as secrets. Structured and labeled credential values must use the
shared sensitive-config-key classification plus HTTP/log-specific secret
fields. Ambiguous opaque provider tokens can only be safely recognized from
that context; attempting to identify every opaque token as a standalone
credential would corrupt ordinary identifiers. URL-bearing diagnostics must
omit private URL components instead of treating regex redaction as permission
to log them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `Utils/log_sanitizer.py` remains importable under its existing public function names and redacts complete credential values without rewriting ordinary model identifiers such as `claude-opus-4-20250514`
- [ ] #2 Structured redaction covers the app's real sensitive config key names, sourced in tests from shipped configuration rather than a duplicate sanitizer-owned list, plus authentication headers, cookies, credential containers, and connection-secret fields
- [ ] #3 Labeled credentials from every configured provider are redacted even when their opaque value has no safely recognizable standalone format; high-confidence standalone key families are fully redacted rather than partially matched
- [ ] #4 Ollama and Transformers model names use bounded single-line input validation, preserve legitimate `claude-*` identifiers, and do not rely on credential redaction for display safety
- [ ] #5 Subscription snapshot-pruning diagnostics contain useful non-private metadata but omit the monitored URL and all URL credentials, paths, queries, and fragments
- [ ] #6 Recursive, non-mutating dictionary/list behavior, non-string mapping keys, `deep=False`, formatting fallback, and the installed-wheel import path are covered without introducing a reduced or test-only application
<!-- AC:END -->

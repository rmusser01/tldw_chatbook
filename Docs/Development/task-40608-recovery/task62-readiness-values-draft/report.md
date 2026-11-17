# Task62 readiness values proposal

Artifact-only implementation of the readiness-values contract, SHA256
`413467d5be6276c668a500dbb080219ee68ebec1fd03aa2fabb66b950d5dd4fc`.
ADR required: no new ADR; the existing ADR-126 detached presentation boundary
applies. The two intended live targets were absent and remain untouched.

`proposed/console_execution_readiness_values.py` implements only
`encode_settings_readiness` and `decode_settings_readiness`. It reuses the
existing finite projection `_record`, `_text`, `_boolean`, `_count`, `_optional`
and `_tuple` helpers. Every schema is explicit; unknown fields and nonexact
objects/mappings/arrays refuse through ConsoleProtocolError without supplied
values in diagnostics. Identity and evidence are reconstructed with their exact
original constructors, including canonical endpoint and finite vocabulary
checks. No live holder, provider call, credential resolution or clock read occurs
in either codec.

Endpoint and generation states, failure categories, model IDs, generation model,
offset timestamps and independently absent observations remain separate.
Original constructor behavior is intentionally retained: ConsoleSettingsReadiness
checks top-level observation types but does not require them to be aware or
accompanied by connection/evidence; ProviderTestEvidence requires aware datetimes.
The codec therefore permits original top-level naive datetimes and rejects naive
evidence datetimes through that constructor. It does not invent equality between
the top-level connection and evidence.identity, nor a stricter display vocabulary.
`invalid_endpoint` receives exact optional-text validation; its vocabulary is not
separately validated by the original readiness constructor.

The intended connection cohort is secret-free by source inspection:
`console_session_settings._console_connection_identity` takes the original
canonical endpoint tuple and sends only `connection_credential_revision(api_key)`
into ProviderDraftIdentity. That revision is a per-process salted HMAC-derived
integer; the API key is not a record field. The endpoint contract refuses URL
userinfo, query strings and fragments, and ProviderDraftIdentity rechecks exact
canonical identity. The transmitted endpoint is the original host/routing path,
not a credential holder. The source configuration, API key, revision salt and
provider continuation are not part of these closed schemas.

`proposed/test_console_execution_readiness_values.py` authors five focused tests:
independent facts and offset roundtrip, real model IDs, unknown/blocked and None
states, strict shapes/types, original canonical identity and evidence refusals,
and secret-bearing URL refusal with redacted errors. Constructor normalization
is exercised by actual original constructors. These tests were **not run**.

Source-only checks: contract hash, two target absences, explicit schemas against
the three original dataclasses, original constructor/credential/endpoint source
inspection, and saved dependency/hash reconciliation. No Python process,
imports, AST/linter/test execution or native operation was launched. Central/root
retain capture, source/key/cadence/age/currentness, protocol integration, App and
Compose publication, Retry behavior, and all actual verification. This artifact
is not an acceptance gate or runtime/performance evidence.

`manifest.json` pins proposed files, this report, explicit target absences and
byte-identical saved dependencies. `preimages.json` records the two absent live
targets. No product or live test file was changed.

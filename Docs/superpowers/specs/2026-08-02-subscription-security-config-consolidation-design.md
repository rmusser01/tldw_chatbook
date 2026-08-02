# Subscription Security Configuration Consolidation (TASK-859)

## Context

The shipped configuration advertises five settings under
`[subscriptions.security]`:

- `enable_xxe_protection`
- `enable_ssrf_protection`
- `verify_ssl_certificates`
- `max_redirects`
- `request_timeout`

None of these settings is read by the subscription runtime. The behavior they
claim to control already has separate owners:

- SSRF, DNS, private-address, and metadata policy:
  `tldw_chatbook.Utils.egress`, configured by `[web_security]`. The
  subscription boundary independently retains its HTTP/HTTPS scheme allowlist.
- Redirect bounds: `tldw_chatbook.Utils.egress.MAX_REDIRECT_HOPS`.
- TLS certificate verification: the per-subscription `ssl_verify` database
  field used by the monitoring engine.
- XML parsing: production subscription parser modules prefer `defusedxml` when
  installed and otherwise retain their existing module-specific standard-library
  fallback behavior. The separate
  `SecurityValidator.validate_xml_content()` helper has no production callers
  and is not a runtime owner of this behavior.
- Fetch timeout: the transport/monitor implementation, not the unread
  subscription-security value.

`Subscriptions/security.py` also defines unused blocked-scheme and metadata
endpoint collections. They duplicate the shared egress policy and are already
stale: the metadata collection omits Alibaba Cloud IPv4 and AWS IPv6 endpoints
that egress blocks.

This design removes the false configuration and duplicate policy ownership. It
does not create new security switches or weaken the existing fail-closed egress
boundary.

## Decision

### Shipped configuration

Remove the entire `[subscriptions.security]` table from
`CONFIG_TOML_CONTENT`. Consequently it is also absent from
`DEFAULT_CONFIG_FROM_TOML` and from newly created user configuration files.

Existing user files are not rewritten. Older files may retain the now-ignored
table and can safely delete it. Automatic removal would be an unnecessary
mutation of user-owned configuration; a startup warning would be noisy because
the old table was generated for every existing installation.

No replacement global subscription-security table is added. The surviving
configuration contract is `[web_security]` for egress policy. Per-subscription
TLS verification remains stored with the subscription record.

### Security policy ownership

Remove `SecurityValidator.BLOCKED_SCHEMES` and
`SecurityValidator.METADATA_ENDPOINTS`. Retain the HTTP/HTTPS
`ALLOWED_SCHEMES` boundary check.

That local allowlist is deliberately not delegated. `Utils.egress` checks the
`[web_security].enabled` kill switch before its own scheme policy, so removing
the subscription check would newly accept `ftp://` and other non-HTTP sources
whenever SSRF checking is disabled. The allowlist therefore preserves the
subscription input contract independently of the optional SSRF policy; it is
not a second blocked-scheme denylist.

`SecurityValidator.validate_feed_url()` otherwise remains behaviorally
unchanged: it preserves the existing empty-input and shape errors, the
HTTP/HTTPS allowlist, delegation call shape, denial mapping, and fragment-free
normalization. Changing whitespace handling or denial-reason interpretation is
outside TASK-859's acceptance criteria and is not required to remove the dead
policy data.

This keeps the public method and exception surface intact while making egress
the only runtime owner of DNS, private-address, and cloud-metadata policy. The
subscription boundary remains the owner of which transport schemes it accepts.

### Documentation

Replace the stale `[subscriptions.security]` example in
`Subscriptions/SUB-Arch.md` with the actual `[web_security]` configuration and
prose explaining:

- the global egress switch and allowlist;
- the per-subscription `ssl_verify` control;
- the code-owned redirect bound; and
- the current dependency-controlled, module-specific optional-`defusedxml` XML
  behavior.

The documentation will explicitly say that legacy `[subscriptions.security]`
tables are ignored and safe to remove.

## Behavioral compatibility

The intended externally observable contracts remain:

| Input or policy result | Result |
| --- | --- |
| Empty URL | `ValueError` |
| Missing scheme or hostname | `ValueError` |
| Unsupported scheme | `SSRFError` from the subscription HTTP/HTTPS allowlist |
| Private, metadata, or unresolvable destination | `SSRFError` |
| Allowed destination | Normalized, fragment-free URL |
| `[web_security].enabled = false` and HTTP/HTTPS input | Normalized URL with no DNS policy check |
| `[web_security].enabled = false` and unsupported scheme | `SSRFError` |

## Verification strategy

Tests use direct functions and production modules only; no reduced application
or test-only application is introduced.

1. A configuration-contract test parses `CONFIG_TOML_CONTENT` and asserts the
   shipped `subscriptions` table has no `security` child.
2. Subscription egress tests exercise `SecurityValidator.validate_feed_url()`
   with the real `[web_security].enabled` contract in both enabled and disabled
   states, proving the surviving switch changes runtime behavior.
3. URL-boundary tests preserve missing-scheme/host and unsupported-scheme
   exception contracts, explicitly prove an unsupported scheme stays rejected
   while `[web_security].enabled` is false.
4. Metadata tests cover every canonical egress endpoint, including
   `100.100.100.200` and `fd00:ec2::254`.
5. An AST-based architecture sentinel scans production Python string constants
   and fails if canonical metadata endpoint data appears outside
   `Utils/egress.py`. A separate assertion prevents the unused
   `BLOCKED_SCHEMES` denylist from returning while allowing the required
   subscription HTTP/HTTPS allowlist.

Focused verification covers `Tests/Utils/test_egress.py`,
`Tests/Subscriptions/test_subscription_egress_wiring.py`, and the new
configuration/ownership contract tests. Repository lint, formatting, and the
broader relevant subscription/config suite run before closeout.

## Scope boundaries

This task does not:

- make `defusedxml` mandatory or change its fallback policy;
- make redirect or timeout bounds newly configurable;
- change the app-wide egress kill switch's ordering or meaning;
- change subscription URL whitespace normalization or denial-reason mapping;
- alter the per-subscription `ssl_verify` schema or behavior;
- rewrite existing user configuration files; or
- clean up other apparently unread `[subscriptions]`, rate-limit, or
  performance settings. Those require separate verification and task scope.

## ADR check

ADR required: no

ADR path: N/A

Reason: this removes misleading unused configuration and duplicate dead policy
data while preserving the security ownership and runtime boundaries established
by TASK-328's web-fetch hardening design. It does not introduce a new storage,
security, provider, runtime, or cross-module contract.

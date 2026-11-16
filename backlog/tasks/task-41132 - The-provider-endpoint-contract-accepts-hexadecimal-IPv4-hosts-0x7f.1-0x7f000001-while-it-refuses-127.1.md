---
id: TASK-41132
title: "The provider endpoint contract accepts hexadecimal IPv4 hosts (0x7f.1, 0x7f000001) while it refuses 127.1"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-15 19:11'
created_date: 2026-11-15 02:20
labels:
- follow-up
- security
- providers
priority: medium
references:
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 endpoint stream fu/r14-endpoint 6cb8e34461 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The contract refuses all-digit hosts such as `127.1` because resolvers expand them into an IP that the dotted form hides. `0x7f.1` and `0x7f000001` expand the same way (`inet_aton` accepts hex and octal parts), so they pass as DNS names while meaning 127.0.0.1. The loopback-only rules and identity comparisons then see a different host from the one dialled.

**Evidence.** `_canonical_host` (`provider_endpoint_contract.py:663-679` on the endpoint branch): after `ip_address(hostname)` fails it refuses a host only when it is all digits and dots (`:676`); otherwise each label passes `valid_dns_label`, which accepts `0x7f`, `0x7f000001` and `1`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Hosts that inet_aton would parse (hex or octal parts, fewer than four parts) are refused, or canonicalized to the address they mean, consistently with the 127.1 rule
- [x] #2 Tests cover 0x7f.1, 0x7f000001, 0177.0.0.1 and a normal name such as 0xdeadbeef.example.com, red before the change
<!-- AC:END -->

## Implementation Plan

1. In `canonical_endpoint_host`, refuse a host whose every label is a number inet_aton reads (decimal, 0-octal, 0x-hex), which covers the existing all-digits rule.
2. Tests for 0x7f.1, 0x7f000001, 0177.0.0.1 (refused) and 0xdeadbeef.example.com (accepted), red first.

## Implementation Notes

`canonical_endpoint_host` now refuses a host whose every dot-separated label is a number `inet_aton` reads: decimal, octal (leading 0) or hex (`0x`, digits optional) — `_INET_ATON_PART`. That rule covers the old all-digits-and-dots rule (`127.1`, `1.2.3.4.5`), so it replaced it; a name with only some numeric-looking labels (`0xdeadbeef.example.com`, `0x7f.lan`, `1.example.com`) is still a name. Refused rather than canonicalized, consistently with `127.1`. Every reader of the contract's host rule (save, Discover, display/compare, Console, the endpoint checks) shares the one function.

- Red/green: `test_a_host_inet_aton_reads_as_an_address_is_refused_like_127_1` was red at base for `0x7f.1`, `0x7f000001` and `0X7F.0.0.1` (`canonical_endpoint_host` returned the host); `0177.0.0.1` was already refused (all digits) and is kept as a control. Each case also asserts `socket.inet_aton(host)` equals 127.0.0.1, and the name cases assert `inet_aton` raises. Green after.
- TASK-41131's dispatcher tests carry `http://0x7f.1:8000` as an extra refused shape (`_DISPATCH_REFUSED_BASES["hex-host"]`).
- Files: `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py`, `Tests/Chat/test_local_server_discovery.py` (lane-run).

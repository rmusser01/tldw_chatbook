---
id: TASK-41132
title: "The provider endpoint contract accepts hexadecimal IPv4 hosts (0x7f.1, 0x7f000001) while it refuses 127.1"
status: To Do
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
- [ ] #1 Hosts that inet_aton would parse (hex or octal parts, fewer than four parts) are refused, or canonicalized to the address they mean, consistently with the 127.1 rule
- [ ] #2 Tests cover 0x7f.1, 0x7f000001, 0177.0.0.1 and a normal name such as 0xdeadbeef.example.com, red before the change
<!-- AC:END -->

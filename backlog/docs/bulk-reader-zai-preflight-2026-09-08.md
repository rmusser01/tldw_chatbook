# Bulk-reader ZAI selection and preflight — 2026-09-08

**Follow-up:** the credential gap was resolved and two live attempts were completed.
See the [live results and adoption decision](bulk-reader-zai-live-2026-09-08.md).
The preflight record below describes the earlier state.

The user delegated model selection. Use **ZAI `glm-5.3` as main** and
**`glm-5.3-flash` as the read-only worker** on the configured ZAI endpoint.
Both document function calling; Flash documents text-parameter compatibility
with GLM-5.3. Account access and runtime compatibility remain unverified.
Sources: [GLM-5.3](https://docs.z.ai/guides/llm/glm-5.3),
[GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash).

The [official pricing page](https://docs.z.ai/guides/overview/pricing), checked
2026-09-08, lists these USD rates per million tokens:

| Model | Input | Cached input | Output |
| --- | ---: | ---: | ---: |
| glm-5.3 | 1.40 | 0.26 | 4.40 |
| glm-5.3-flash, promotional | 0.075 | 0.015 | 0.25 |
| glm-5.3-flash, list | 0.15 | 0.03 | 0.50 |

The Flash promotion ends September 9, 2026 at 24:00 UTC+8 (September 9 at
16:00 UTC). Recheck rates when the actual run occurs. Chatbook's seeded
pricing catalog currently has no entries for these models; any later estimate
must record verified rates through the existing pricing override mechanism.
Missing usage must remain unknown. These rates establish a useful experimental
contrast, not total-run savings.

The real CLI was invoked with the chosen pair and `--confirm-billable` under
a disposable profile containing only the relevant configured provider settings
and scratch data paths. It exited **2** before any model request:

```text
zai is not ready: Missing API key. Set ZAI_API_KEY or add api_key under [api_settings.zai].
```

Neither ZAI nor Moonshot had a credential in the effective provider settings or
this process environment. No comparison JSON was produced; the original config
remained byte-identical. Once the ZAI credential is configured, execute the
existing four-case comparison, retain all failed arms and their spend, and
review answers, qualifications and citations against the pinned corpus.
TASK-32026 remains In Progress with AC 5 open. No quality or savings conclusion
is available from this preflight.

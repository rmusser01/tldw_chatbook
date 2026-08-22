# ADR-080: Keep model machine-fit estimates local, conservative, and runtime-neutral

Status: Accepted
Date: 2026-08-22
Related Task: TASK-20938

## Decision

Chatbook will expose a provider-neutral machine-memory observation and GGUF fit
estimation contract for model discovery. The contract is advisory: it estimates
whether one model can plausibly load at a small 32K context or standard 64K
context, but it does not claim runtime compatibility, successful inference,
GPU offload, or acceptable speed.

Machine observations remain process-memory only. They are never persisted,
synchronized, or included in ordinary or debug logs. The probe collects only
bounded platform, architecture, total/available physical-memory, memory-kind,
and accelerator-memory facts. It never collects hostnames, serial numbers,
GPU UUIDs, driver inventories, process lists, or network data.

The supported observation platforms are macOS, Linux, and Windows. Physical
memory comes from the already-required `psutil` dependency and is observed off
the Textual event loop. Apple Silicon is classified as unified memory: its
physical-memory value is shown once and is never added again as VRAM. On Linux
and Windows, dedicated accelerator memory is optional evidence. A bounded,
fixed-argument `nvidia-smi` probe may run only from reviewed system locations,
with no shell, a short timeout, a strict output cap, and sanitized parsing.
Linux may additionally read bounded AMD or Intel DRM VRAM counters from
resolved system-owned sysfs paths. Multiple accelerators remain separate and
are never summed because usable combinations depend on runtime configuration.
Missing accelerator evidence makes the snapshot partial; it does not prevent a
system-memory estimate. A complete snapshot means every supported probe branch
for that platform settled successfully; it is not an exhaustive hardware or
driver inventory.

The v1 estimation policy uses the exact candidate byte total and integer-only
allowances rounded upward to MiB:

- runtime allowance = `max(1 GiB, 10% of GGUF bytes)`;
- 32K context allowance = `max(4 GiB, 25% of GGUF bytes)`;
- 64K context allowance = twice the 32K allowance;
- safe machine reserve = `max(2 GiB, 20% of total physical memory)`;
- safe model budget = total physical memory minus that reserve.

For each context, an estimated load within the safe model budget is **likely**,
an estimate above the safe budget but within total physical memory is a
**close call**, and an estimate above total physical memory is **unlikely**.
Missing or invalid physical-memory evidence yields **unknown**. Current
available memory is displayed as volatile evidence when valid but does not
change the rating, so background load cannot make rows oscillate while users
compare them.

The Remote Models UI will lead with the 64K result and show the 32K fallback.
Examples include `Likely fits at 64K`, `Likely at 32K · 64K is close`, and
`Unlikely at 32K`. Every row shows the estimated load and safe budget that
produced its label. A compact machine-evidence panel exposes RAM/unified memory,
available memory when valid, per-device VRAM when observed, assumptions, and a
keyboard-reachable Refresh action. Ratings never disable candidate selection,
consent, or installation.

RemoteView owns only the presentation and request generation. A separate
format/provider-neutral module owns immutable observation values, bounded
platform probes, and pure estimation. Observation begins lazily after a
repository resolves. It runs in an exclusive thread worker, and only the
current generation may publish. Refresh retains the previous accepted snapshot
until a newer snapshot succeeds; a refresh failure is shown alongside the age
of the retained evidence. Candidate estimate statics update in place so a
finishing probe cannot destroy keyboard focus. Filtered or sorted rows project
the latest accepted snapshot when they are rebuilt.

## Context

Deterministic filename, size, and shard guidance helps users understand GGUF
variants but cannot answer the next practical question: whether a download is
plausible on their machine. A raw comparison between file size and current free
RAM would be unstable and misleading. Context cache, runtime buffers, the
operating system, unified memory, discrete VRAM, and multi-GPU policy all affect
actual use.

The Models screen currently has exact candidate bytes but not downloaded GGUF
metadata, a selected runtime, GPU-layer settings, or a proven accelerator
backend. Fetching provider-specific file headers would add network and adapter
contracts, while importing native ML runtimes to inspect accelerators can
initialize hardware or abort headless processes. The estimate therefore needs
an explicit conservative policy whose inputs and limitations are visible.

ADR-025 remains authoritative for artifact truth, structural GGUF admission,
managed storage, provenance, and the rule that generic GGUF validity is not
runtime compatibility. This ADR adds only local capability observation and an
advisory comparison projection.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Compare GGUF bytes directly with currently available RAM | Omits runtime/context memory and makes labels fluctuate with unrelated machine load. |
| Add RAM and VRAM into one capacity number | Double-counts Apple unified memory and assumes an unselected runtime can combine discrete devices. |
| Query llama.cpp, llamafile, CUDA, Metal, or ML frameworks | Couples browsing to runtime installation and can initialize native hardware before the user selects a runtime. |
| Fetch and parse remote GGUF headers | Adds provider-specific range requests, new remote trust/bounds, and network work that deterministic browsing does not otherwise require. |
| Use model-size-only context labels without showing assumptions | Produces false precision; the chosen heuristic remains useful only when its policy and inputs are inspectable. |
| Persist a machine profile | Creates stale device-local state and a privacy/synchronization contract for facts cheap enough to observe per process. |
| Sum multiple GPU memories | Whether memory can be combined depends on tensor-split and runtime policy, which is outside download selection. |
| Require users to enter RAM and VRAM | Adds avoidable setup burden and allows stale or malformed values to masquerade as detected evidence. |

## Consequences

- A new provider-neutral model capability module becomes the single owner of
  machine-memory snapshots and GGUF memory estimates.
- Most Linux and Windows snapshots may be partial because absence of one
  supported accelerator probe is not proof that no accelerator exists.
- CPU architecture is evidence context only. Core count, throughput, and model
  speed are not estimated.
- Apple Silicon receives one unified-memory value rather than misleading RAM
  plus VRAM totals.
- 64K is the standard comparison and 32K is the small fallback. Smaller context
  configurations may work when both estimates are pessimistic, but v1 does not
  add more context presets.
- The estimates intentionally favor false caution over false reassurance and
  may be conservative for architectures with efficient grouped-query attention.
- Exact context memory remains unknown until architecture metadata and runtime
  settings are available. A future runtime-specific estimator must add a new
  evidence tier rather than silently changing the meaning of these labels.
- Accelerator observations are informational in v1 and do not change the
  system-memory rating.
- Probe failures and unsupported platforms leave deterministic variant guidance
  and installation fully usable.
- Focused tests must cover integer boundaries, all observation states, injected
  macOS/Linux/Windows sources, output/time/path bounds, privacy, stale refreshes,
  and production 80-column compositor behavior. Automated tests do not invoke
  real accelerator tools.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-22-remote-model-machine-fit-design.md)
- [TASK-20938](../tasks/task-20938%20-%20Add-hardware-aware-GGUF-machine-fit-estimates.md)
- [ADR-025: Shared artifacts and runtime routing](025-shared-stt-artifacts-and-runtime-routing.md)

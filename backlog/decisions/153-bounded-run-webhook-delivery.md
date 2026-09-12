# ADR-153: Bounded reusable run-webhook delivery

Status: Accepted
Date: 2026-09-12
Related Task: TASK-31511

## Decision

Keep the existing optional, signed, best-effort webhook API and replace per-event threads with one module-owned lazy delivery worker. Each active worker generation reuses one daemon thread and one asyncio Runner/event loop. It admits at most 32 waiting notifications plus the current delivery, in FIFO order. Admission never waits for queue space or a network result.

Queue saturation rejects the newest notification and returns False. It emits a fixed content-free diagnostic and a bounded reason/event counter; it does not log payloads, identifiers, destinations, signing secrets, or caller objects. True means admitted, not successfully delivered. No retry or delivery-durability promise is added.

An admitted record captures the transition's immutable configuration and a copied identifiers mapping. The worker does not reload settings later, because routing or signing an already-admitted event with a newer configuration would change that event's meaning. The existing process-wide settings cache and its write invalidation remain the sole cache.

The worker retires after 30 idle seconds and a later event can start a new generation. Admission and retirement share one state lock and an exact generation identity so no True admission can be stranded behind a worker that decided to exit. Failed thread start returns False and leaves no queued event without an owner. Individual delivery failure cannot kill the queue. A generation's event loop closes before it is published as retired; retiring generations must not accumulate under repeated events.

No application shutdown join, atexit hook, durable outbox, permanent admission fence, or new setting is introduced. Process exit can abandon best-effort notifications, as today. Idle cleanup releases the event loop; a network stall cannot create more delivery workers or block run finalization.

## Existing boundaries preserved

Every POST retains the current per-delivery SSRF check, HMAC over exact serialized bytes, event subscription, finite configured transport timeout, missing-secret refusal, and error containment. Payloads remain identifiers and outcome categories. Terminal persistence still precedes the scheduling attempt. Other providers, Console runtime ownership, and non-UI AgentService construction do not acquire a new dependency.

## Alternatives

- A ThreadPoolExecutor alone bounds threads but has an unbounded submission queue and participates in interpreter shutdown joining. It does not preserve this best-effort exit contract.
- An app-owned injected dispatcher would require plumbing through Console, headless and non-UI service construction. A lazy module worker is sufficient for this optional event sink and can retire when unused.
- A webhook-specific settings cache duplicates existing invalidation and risks retaining an obsolete destination or secret. Existing source-aware config caching already avoids repeated warm-path disk reads.
- Durable retries/outbox storage would add delivery guarantees and privacy/retention policy absent from this task.

## Verification and consequences

Gated coroutine tests prove nonblocking finite admission, FIFO completion, shared thread/loop identity, idle retirement/restart, retirement/admission races, and recovery after worker/start failures. Payload/signature/egress tests remain unchanged. Existing configuration cache/invalidation tests qualify the old disk-read concern.

Notifications may be dropped during bursts once the finite queue fills; that is an explicit best-effort limitation. One slow endpoint reduces throughput, but it cannot produce unbounded threads or pending notifications. More delivery concurrency would require workload evidence and a separate limit decision.

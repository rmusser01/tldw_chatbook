# ADR-153: Bounded reusable run-webhook delivery

Status: Accepted
Date: 2026-09-12
Related Task: TASK-31511

## Decision

Keep the existing optional, signed, best-effort webhook API and replace per-event threads with one module-owned lazy delivery worker. Each active worker generation reuses one daemon thread and one asyncio Runner/event loop. It admits at most 32 waiting notifications plus the current delivery, in FIFO order. Admission never waits for queue space or a network result.

Queue saturation rejects the newest notification and returns False. It emits a fixed content-free diagnostic and a bounded reason/event counter; it does not log payloads, identifiers, destinations, signing secrets, or caller objects. True means admitted, not successfully delivered. No retry or delivery-durability promise is added.

An admitted record captures the transition's immutable configuration and a copied identifiers mapping. The worker does not reload settings later, because routing or signing an already-admitted event with a newer configuration would change that event's meaning. The existing process-wide settings cache and its write invalidation remain the sole cache.

The worker starts retirement after 30 idle seconds and a later event can start a new generation after retirement completes. Admission and retirement share one state lock and an exact generation identity so no True admission can be stranded behind a worker that decided to exit. Failed thread start returns False and leaves no queued event without an owner. Individual delivery failure cannot kill the queue. A generation's resolver executor finishes and its event loop closes before it is published as retired; retiring generations must not accumulate under repeated events.

No application shutdown join, atexit hook, durable outbox, permanent admission fence, or new setting is introduced. Process termination can abandon best-effort notifications, as today. Idle cleanup releases the event loop after resolver settlement; a network stall cannot create more delivery workers or block run finalization. The standard-library resolver executor participates in normal interpreter shutdown, so a daemon delivery thread does not guarantee prompt process exit while DNS is running.

## Delivery deadline and resolver ownership amendment (2026-09-13)

PR2665 review found that the transport timeout started after the egress DNS lookup. Apply the validated finite timeout to the complete delivery await, including egress and POST, so a cancellation-cooperative stalled delivery fails and the next queued notification can proceed. Normalize directly constructed configurations at admission using the same policy as settings; keep the captured configuration immutable.

Each generation uses the standard ThreadPoolExecutor with two workers and two nonblocking native-job admission slots. Two slots allow a healthy lookup while one resolver is stalled; saturation fails that delivery best-effort instead of accumulating an internal executor queue. A running-state proxy Future keeps waiter cancellation from cancelling a native job before it starts and prematurely releasing its slot. Release admission only when the underlying native Future settles. If submit raises after it may have enqueued work, conservatively retain that slot for the rest of the generation. Retirement cancels queued jobs before joining running workers, including queued jobs whose submission raised. This is a small bounded adapter for the existing event-loop executor, not a new resolver backend.

Cancelling `loop.getaddrinfo` cancels its waiter, not an already-running OS resolver in the loop's default executor. During retirement the delivery thread therefore explicitly awaits `shutdown_default_executor(timeout=None)` before closing the Runner. Relying only on Runner's finite executor-shutdown budget can publish a retired generation while its resolver threads remain alive, allowing later generations to accumulate abandoned resources. Keep the existing generation owned and nonaccepting until actual settlement; retirement-time submissions return False without waiting. This extends the existing ownership fence rather than adding a resolver backend or application lifetime service.

Runner cleanup is not itself proof of native settlement: its executor-shutdown helper can fail to start. Retain the executor and perform a final synchronous `shutdown(wait=True)` on the existing delivery thread on every exit path, with admission closed first. Only a successful native join permits owner publication as retired. If that direct join also raises, retain the nonaccepting owner and report only the exception class; do not start another generation with unconfirmed native work.

The deadline does not forcibly interrupt synchronous work or cancellation-suppressing coroutines. A stuck OS resolver may keep this one generation retiring indefinitely and may delay normal interpreter exit. No physical completion or prompt shutdown is claimed at the instant of timeout. Tests gate real executor-backed DNS, prove a queued healthy POST before releasing it, accelerate Runner's fallback join budget to expose premature retirement, and release/join every exact owned thread before claiming cleanup.

## Existing boundaries preserved

Every POST retains the current per-delivery SSRF check, HMAC over exact serialized bytes, event subscription, finite configured transport timeout, missing-secret refusal, and error containment. Payloads remain identifiers and outcome categories. Terminal persistence still precedes the scheduling attempt. Other providers, Console runtime ownership, and non-UI AgentService construction do not acquire a new dependency.

## Alternatives

- A ThreadPoolExecutor alone bounds threads but has an unbounded submission queue. The bounded adapter adds finite native-job admission while retaining standard DNS and executor shutdown behavior; it does not replace the bounded notification FIFO.
- An app-owned injected dispatcher would require plumbing through Console, headless and non-UI service construction. A lazy module worker is sufficient for this optional event sink and can retire when unused.
- A webhook-specific settings cache duplicates existing invalidation and risks retaining an obsolete destination or secret. Existing source-aware config caching already avoids repeated warm-path disk reads.
- Durable retries/outbox storage would add delivery guarantees and privacy/retention policy absent from this task.

## Verification and consequences

Gated coroutine tests prove nonblocking finite admission, FIFO completion, shared thread/loop identity, idle retirement/restart, retirement/admission races, and recovery after worker/start failures. Payload/signature/egress tests remain unchanged. Existing configuration cache/invalidation tests qualify the old disk-read concern.

Notifications may be dropped during bursts once the finite queue fills; that is an explicit best-effort limitation. One slow endpoint reduces throughput, but it cannot produce unbounded threads or pending notifications. More delivery concurrency would require workload evidence and a separate limit decision.

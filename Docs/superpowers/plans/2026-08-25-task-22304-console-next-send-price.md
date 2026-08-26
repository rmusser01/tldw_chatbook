# Console Next-Send Price Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show `Send | $` or `Queue | $` for every currently sendable Console payload and explain the estimated text-processing upper bound in the existing hover tooltip without changing dispatch or accumulated-spend behavior.

**Architecture:** Add one no-DOM `ConsoleSendPriceController` plus immutable pure presentation model in `UI/Console_Modules/send_price.py`. `wiring.py` constructs the controller from named late-binding accessors; `ChatScreen` only forwards its callback into `ConsoleComposerBar`, which remains responsible for its displayed label, width, and tooltip precedence. The store provides detached buffered-history snapshots, while the Chat controller shares one lightweight pre-serialization filter/media-admission projection between dispatch and estimation; only dispatch base64-serializes media. The price controller appends draft/staged text rows and memoizes tokenization with the existing verified-signature cache; the pure builder owns cost availability, math, and copy.

**Tech Stack:** Python 3.11, Textual 8, pytest/pytest-asyncio, Ruff, existing `PricingCatalog`, `TokenEstimateCache`, and `format_cost_amount` utilities.

---

## Working constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-22302-send-price-indicator` on `codex/task-22302-send-price-indicator`.
- Treat `Docs/superpowers/specs/2026-08-25-task-22304-console-next-send-price-design.md` and Backlog `TASK-22304` as the behavior contract. Update the task acceptance criteria before implementing any newly discovered scope.
- Use test-driven development: observe each new test fail for the intended reason before its corresponding production change.
- Keep all estimate construction and mutable memo state out of `UI/Screens/chat_screen.py`; do not raise the existing screen-size ratchet budget.
- Preserve the existing blocker/setup/wake/recovery tooltip precedence and unsuffixed blocked labels.
- Use the uncached input rate and configured maximum reply tokens. Do not assign token or dollar values to pending or dispatch-admitted historical binary/media attachments.
- ADR required: yes. ADR path: `backlog/decisions/088-console-lightweight-next-send-history-projection.md`. Reason: the feature adds a cross-module detached store snapshot and shared pre-serialization history projection so per-keystroke presentation observes buffered text/admitted media without writes or base64 work.

## Task 1: Lock the pure price and tooltip contract

**Files:**

- Create: `Tests/UI/test_console_send_price.py`
- Create: `tldw_chatbook/UI/Console_Modules/send_price.py`
- Reference: `tldw_chatbook/Chat/cost_display.py`
- Reference: `tldw_chatbook/LLM_Calls/pricing_catalog.py`

- [ ] Add table-driven pure tests for known pricing, unknown pricing, explicit local zero pricing, missing input tokens, missing reply limit, pending attachments, historical media, and blank provenance identifiers. Use an exact `ModelPricing` fixture and assert the full tooltip text, including comma-grouped tokens, `up to`, `~$`, `as_of`, and distinct `Attachments`/`Media context` caveats.

  ```python
  def test_build_next_send_price_formats_known_upper_bound() -> None:
      state = build_next_send_price(
          input_tokens=1_284,
          max_reply_tokens=4_096,
          pricing=ModelPricing(
              input_per_mtok=3.0,
              output_per_mtok=20.4,
              cache_read_per_mtok=None,
              cache_write_per_mtok=None,
              as_of="2026-08-01",
          ),
          provider="anthropic",
          model="claude-sonnet-4-6",
          attachment_count=0,
      )

      assert state.tooltip == (
          "Next request: up to ~$0.0874\n"
          "Input: ~1,284 tokens · ~$0.0039\n"
          "Reply: up to 4,096 tokens · ~$0.0836\n"
          "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01"
      )
  ```

- [ ] Run the pure test selection and verify RED because `send_price.py`/`build_next_send_price` do not exist:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_send_price.py -q --tb=short -p no:warnings
  ```

- [ ] Implement the immutable presentation and pure builder in `send_price.py`. Keep the public surface deliberately small:

  ```python
  @dataclass(frozen=True, slots=True)
  class ConsoleNextSendPrice:
      tooltip: str


  def build_next_send_price(
      *,
      input_tokens: int | None,
      max_reply_tokens: int | None,
      pricing: ModelPricing | None,
      provider: str,
      model: str,
      attachment_count: int = 0,
      historical_media_count: int = 0,
  ) -> ConsoleNextSendPrice:
      ...
  ```

  Compute known text components with `round(tokens * rate / 1_000_000, 6)` and format dollars only after a value is known. Build provenance by joining only nonblank provider/model parts, followed by either `rates as of ...` or `pricing not configured`. Pending attachments or historical media change the input/reply prefixes to `Input text`/`Reply text`, append their distinct count caveats, and force the headline unavailable.

- [ ] Run the pure tests and verify GREEN.

- [ ] Run focused formatting/static checks:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/UI/Console_Modules/send_price.py Tests/UI/test_console_send_price.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/UI/Console_Modules/send_price.py Tests/UI/test_console_send_price.py
  ```

- [ ] Commit the pure contract:

  ```bash
  git add tldw_chatbook/UI/Console_Modules/send_price.py Tests/UI/test_console_send_price.py
  git commit -m "feat(console): model next-send price tooltip"
  ```

## Task 2: Build and verify the read-only request-history controller

**Files:**

- Modify: `Tests/UI/test_console_send_price.py`
- Modify: `tldw_chatbook/UI/Console_Modules/send_price.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
- Reference: `tldw_chatbook/Chat/console_cost_tracker.py`
- Reference: `tldw_chatbook/Chat/console_display_state.py`
- Reference: `tldw_chatbook/Chat/console_session_settings.py`

- [ ] Add `test_read_only_messages_for_session_projects_stream_buffer_without_mutation` in `Tests/Chat/test_console_chat_store.py`. Build a streaming assistant with buffered-but-unmaterialized chunks and a persistence spy, capture live message content, materialization counters, payload/speech revisions, and persistence calls, then assert `read_only_messages_for_session(session_id)` returns detached current text while every captured store value and call count remains unchanged.

- [ ] Add `test_provider_messages_for_next_send_estimate_uses_lightweight_projection_without_media_serialization` in `Tests/Chat/test_console_chat_controller.py`. Use pathological history (transcript-only system, seeded leading greeting, failed row, empty speech placeholder, disallowed assistant state, normal user/assistant, buffered live assistant, and eligible plus over-budget historical images). Monkeypatch the controller module's `image_url_part` to raise if called, then assert the named estimate method returns the exact canonical text rows and only the dispatch-admitted media count without changing store/persistence state.

- [ ] Add `test_provider_message_payloads_serializes_only_after_lightweight_projection` beside that test: with the same eligible image fixture, `_provider_message_payloads` still calls the patched `image_url_part` exactly for dispatch-admitted attachments and produces the incumbent provider content shape. This pins the new boundary: common filtering/admission happens first, serialization happens only for dispatch.

- [ ] Add send-price controller tests using tiny fake settings/store/launch accessors, the injectable canonical provider-history accessor, and injectable catalog/counter callables. Capture the exact counter input and prove that it starts with the accessor's already-filtered provider rows, then contains the nonblank live draft and `console_prompted_evidence_text(launch)` exactly once and in request order.

  ```python
  controller = ConsoleSendPriceController(
      settings_accessor=lambda: settings,
      chat_store_accessor=lambda: store,
      provider_history_accessor=lambda session_id: canonical_history_projection,
      pending_launch_accessor=lambda: launch,
      pricing_catalog_accessor=lambda: catalog,
      token_counter=count_tokens,
  )
  state = controller.presentation_for_draft("live draft")

  assert captured_rows == [
      {"role": "system", "content": "system + seeded greeting"},
      {"role": "user", "content": "history"},
      {"role": "user", "content": "live draft"},
      {"role": "user", "content": staged_text},
  ]
  assert state is not None
  ```

- [ ] Add cache/refresh tests. An identical call must reuse the `TokenEstimateCache`; changing draft, provider, model, system prompt, history, or staged evidence must change the complete signature and invoke the counter again. Changing only attachment count must refresh the presentation/caveat without fabricating media tokens or requiring another text-token pass.

- [ ] Add an attachment-only controller test: a blank draft with one pending binary/media attachment must still return a presentation, retain any known projected-history text components, force the total unavailable, and include `Attachments: 1 · media cost not estimated`. Only a blank draft with zero attachments returns `None`.

- [ ] Add a historical-media controller test whose injected `ConsoleNextSendHistoryProjection` contains one text row plus `historical_media_count=1`. Assert the counter receives only the text row, media receives no token/dollar value, the headline is unavailable, and the tooltip includes `Media context: 1 item · media cost not estimated`. Cover the plural form and simultaneous pending/historical caveats in the pure tests.

- [ ] Add degradation tests: missing store/session produces an unavailable tooltip rather than raising. A token-counter exception must call the pure builder with `input_tokens=None`, retaining `Input: token estimate unavailable`, the reply line, and provenance. Catalog or broader accessor/controller failures may return the short `ConsoleNextSendPrice("Next request: cost unavailable")` fallback and must never affect send readiness.

- [ ] Run the controller selection and named read-only snapshot/history-seam tests and verify RED because the APIs are absent:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_send_price.py Tests/Chat/test_console_chat_store.py::test_read_only_messages_for_session_projects_stream_buffer_without_mutation Tests/Chat/test_console_chat_controller.py::test_provider_messages_for_next_send_estimate_uses_lightweight_projection_without_media_serialization Tests/Chat/test_console_chat_controller.py::test_provider_message_payloads_serializes_only_after_lightweight_projection -q --tb=short -p no:warnings -k "controller or cache or context or unavailable or read_only_messages or provider_messages"
  ```

- [ ] Implement `ConsoleChatStore.read_only_messages_for_session(session_id)`. Validate the session, make detached snapshots from the active-path view, and, for each copied streaming row with buffered chunks newer than the materialized count, set only the copy's content to `"".join(buffer)`. Do not call `_fold_stream_buffer_without_persistence`, `_materialize_stream_buffer`, persistence, or any revision bump.

- [ ] Add immutable `ConsoleNextSendHistoryProjection` to `Chat/console_chat_models.py` with `rows: tuple[tuple[str, str], ...]` and `historical_media_count: int`. This is the only value the UI-facing accessor returns; it contains no bytes or provider wire parts.

- [ ] Extract a private lightweight Chat-controller projector that accepts already-snapshotted messages and returns private rows carrying source identity, role, resolved text, and only the currently vision/budget-admitted attachment references. Move the incumbent role/status/empty-row/leading-assistant/assistant-state filters, newest-first image reservation, and omitted-image placeholder choice into this helper without calling `image_url_part`.

- [ ] Refactor `_provider_message_payloads` to consume the lightweight rows and perform its existing `image_url_part` serialization afterward. Keep `_provider_messages_for_session` on its existing materializing store path for dispatch. Add `provider_messages_for_next_send_estimate(session_id)` on the read-only store path; it consumes the same lightweight rows, folds the same system/greeting text, returns only `(role, text)` tuples plus the eligible attachment count, and never invokes provider serialization.

- [ ] Implement `ConsoleSendPriceController` in the no-DOM module with named keyword-only callables and an owned `TokenEstimateCache`:

  ```python
  class ConsoleSendPriceController:
      def __init__(
          self,
          *,
          settings_accessor: Callable[[], ConsoleSessionSettings],
          chat_store_accessor: Callable[[], ConsoleChatStore | None],
          provider_history_accessor: Callable[
              [str], ConsoleNextSendHistoryProjection
          ],
          pending_launch_accessor: Callable[[], ConsoleLiveWorkLaunch | None],
          pricing_catalog_accessor: Callable[[], PricingCatalog] = get_pricing_catalog,
          token_counter: Callable[[list[dict[str, str]], str, str], int] = (
              _estimate_tokens_locally
          ),
      ) -> None:
          self._token_estimates = TokenEstimateCache(max_entries=1)
          ...

      def presentation_for_draft(
          self, draft_text: str
      ) -> ConsoleNextSendPrice | None:
          ...

      def tooltip_for_draft(self, draft_text: str) -> str | None:
          state = self.presentation_for_draft(draft_text)
          return state.tooltip if state is not None else None
  ```

  Convert the projection's immutable `(role, text)` rows to the token counter's mapping shape, then append draft/staged rows without reimplementing history filters or touching media bytes. Tolerate a closed/missing active session and use a cache key containing the active session id. Verify every hit with `token_estimate_signature(rows, model, provider_config_key(provider))`; pending-attachment and projected historical-media counts stay outside this signature because they do not change text tokenization, but are read on every presentation build. Catch token-counter errors around only the memoized compute, set `input_tokens=None`, and continue through the pure builder.

- [ ] Pin the shared projection boundary with the pathological Chat-controller fixture above: canonical history contains the folded seeded-greeting system row but omits transcript-only system rows, failed rows, empty speech placeholders, leading assistant rows, and assistant states disallowed from provider history; admitted media is counted, omitted media becomes the incumbent text placeholder, and `image_url_part` is never called. The send-price test treats the injected immutable projection as authoritative rather than recreating those rules.

- [ ] Run `test_console_send_price.py` plus both focused read-only projection tests and verify GREEN, then run Ruff checks for the six changed files.

- [ ] Commit the controller:

  ```bash
  git add tldw_chatbook/UI/Console_Modules/send_price.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py Tests/UI/test_console_send_price.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py
  git commit -m "feat(console): estimate the pending request context"
  ```

## Task 3: Add the optional composer affordance without weakening blocker truth

**Files:**

- Modify: `Tests/UI/test_console_send_price.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `Tests/UI/test_console_send_disabled_state.py`

- [ ] Immediately before editing the Textual widget, read `.agents/skills/impeccable/reference/craft-floor.md` in full and apply its interaction/layout checks to this bounded change.

- [ ] Add a mounted minimal-composer test app whose callback returns a deterministic tooltip. Cover empty draft (`Send`, current guidance), ready draft (`Send | $`), attachment-only readiness with a blank draft (`Send | $`), ready run-follow-up (`Queue | $`), dynamic width, and repeated `_sync_current_action_state()` calls that must not duplicate the suffix.

- [ ] Add blocked/setup/wake regression cases proving a sendable draft with a callback keeps the unsuffixed base label, disabled state, and existing blocker tooltip whenever `send_blocked` is true. Retain a standalone no-callback assertion so existing callers keep current generic behavior.

- [ ] Run the composer tests and verify RED on the unchanged `Send`/`Queue` label:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_send_price.py Tests/UI/test_console_send_disabled_state.py -q --tb=short -p no:warnings -k "composer or typing_enables or setup_block"
  ```

- [ ] Extend `ConsoleComposerBar.__init__` with an optional `Callable[[str], str | None]`, storing it without evaluating it. In `sync_action_state`, preserve `_send_label` as the unsuffixed authoritative base and derive presentation only after `send_ready`:

  ```python
  displayed_label = send_label
  details = None
  if send_ready and self._send_price_tooltip_provider is not None:
      try:
          details = self._send_price_tooltip_provider(self.draft_text())
      except Exception:  # noqa: BLE001 -- pricing presentation cannot block Send
          details = "Next request: cost unavailable"
      if details:
          displayed_label = f"{send_label} | $"

  tooltip = blocked_tooltip if self._send_blocked else details or generic_tooltip
  ```

  Apply `displayed_label` to both `button.label` and `max(6, cell_len(displayed_label) + 2)` width. Do not change focus/hover CSS, the one-row action cluster, dispatch, disabled-state derivation, or reason-strip behavior.

- [ ] Update only the mounted Console expectations in `test_console_send_disabled_state.py` that now receive the wired callback; standalone composer tests without a provider must keep the original generic tooltip.

- [ ] Run the composer/send-disabled files and verify GREEN. Run Ruff and format checks on all changed files.

- [ ] Commit the presentation seam:

  ```bash
  git add tldw_chatbook/Widgets/Console/console_composer_bar.py Tests/UI/test_console_send_price.py Tests/UI/test_console_send_disabled_state.py
  git commit -m "feat(console): show price affordance on ready Send"
  ```

## Task 4: Wire the controller and prove live Console refreshes

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_send_price.py`
- Modify as required by intentional label changes: `Tests/UI/test_console_prompt_queue.py`
- Modify as required by intentional label changes: `Tests/UI/test_console_native_chat_flow.py`
- Reference: `Tests/UI/test_console_cost_chip_screen.py`

- [ ] Add wiring tests that construct a Console and assert `screen._send_price` is a `ConsoleSendPriceController`. Patch the late-bound settings/store/staged context after construction and prove the controller observes the replacements rather than captured snapshots.

- [ ] Add a mounted priced-Anthropic regression using the existing sandboxed persisted-config pattern from `test_console_cost_chip_screen.py`. Type a draft and assert `Send | $`, separate input/reply lines, provider/model, rate date, and the unchanged pre-send accumulated-cost chip.

- [ ] In the mounted test, mutate one source per assertion—draft, provider/model, system/max tokens, active session/history, staged evidence, and pending attachments—and invoke the existing action/control sync used by that mutation. Assert the tooltip refreshes; draft edits must refresh synchronously through the composer's own `_sync_current_action_state`, not the 0.2-second screen tick. Clear the draft while leaving a pending binary attachment and assert `Send | $` plus the attachment caveat. During an accepted live run, load a second draft and assert `Queue | $`.

- [ ] Run the mounted/wiring tests and verify RED because no controller is wired and `ChatScreen` provides no callback:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_send_price.py Tests/UI/test_console_prompt_queue.py -q --tb=short -p no:warnings -k "wiring or mounted or priced or queue"
  ```

- [ ] Import and construct the fifteenth controller in `UI/Console_Modules/wiring.py`, updating its controller-count docstring/list. Use named late-binding accessors only:

  ```python
  screen._send_price = ConsoleSendPriceController(
      settings_accessor=(
          lambda: screen._session._ensure_active_console_session_settings()
      ),
      chat_store_accessor=lambda: screen._console_chat_store,
      provider_history_accessor=(
          lambda session_id: screen._ensure_console_chat_controller()
          .provider_messages_for_next_send_estimate(session_id)
      ),
      pending_launch_accessor=lambda: screen._pending_console_launch_context,
  )
  ```

- [ ] At the existing `ConsoleComposerBar` construction in `UI/Screens/chat_screen.py`, pass only the callback—no estimate state or helper methods on the screen:

  ```python
  send_price_tooltip_provider=(
      lambda draft: self._send_price.tooltip_for_draft(draft)
  ),
  ```

- [ ] Update only intentional ready-payload label assertions from `Send`/`Queue` to their suffixed forms. Preserve empty-payload `Send`/`Queue` and all blocked-state assertions unsuffixed. Use `rg -n 'label\.plain == "(Send|Queue)"' Tests/UI` to audit every expectation before editing.

- [ ] Run all focused feature/integration regressions and verify GREEN:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_send_price.py Tests/UI/test_console_send_disabled_state.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_cost_chip_screen.py -q --tb=short -p no:warnings
  ```

- [ ] Run the native-flow tests containing changed label assertions. If the pre-existing OpenAI post-init readiness failure remains outside the changed cases, report it as baseline rather than weakening readiness behavior.

- [ ] Run Ruff/format checks for all changed Python files and commit:

  ```bash
  git add tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI
  git commit -m "feat(console): wire live next-send pricing"
  ```

## Task 5: Verify craft, regressions, and Backlog completion

**Files:**

- Modify: `backlog/tasks/task-22304 - Show-an-estimated-next-send-price-on-the-Console-Send-button.md`
- Modify only if a generalizable incident occurred: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`
- Verify: all production/test files changed above

- [ ] Mount the Console at wide and narrow terminal sizes and visually inspect the one-row action cluster with empty `Send`, ready `Send | $`, ready `Queue | $`, blocked unsuffixed state, and multiline tooltip. Confirm no clipping, overlap, dimension-changing hover/focus CSS, or color-only meaning. Save a screenshot or test artifact when the harness supports it.

- [ ] Run Impeccable's required detector once against the changed UI targets and address applicable findings:

  ```bash
  node .agents/skills/impeccable/scripts/detect.mjs --json tldw_chatbook/Widgets/Console/console_composer_bar.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Console_Modules/send_price.py
  ```

- [ ] Run the complete focused baseline plus the new tests from a clean invocation:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_send_price.py Tests/UI/test_console_send_disabled_state.py Tests/UI/test_console_cost_chip_screen.py Tests/UI/test_console_prompt_queue.py Tests/Chat/test_console_session_settings.py Tests/LLM_Calls/test_pricing_catalog.py Tests/Chat/test_console_chat_store.py::test_read_only_messages_for_session_projects_stream_buffer_without_mutation Tests/Chat/test_console_chat_controller.py::test_provider_messages_for_next_send_estimate_uses_lightweight_projection_without_media_serialization Tests/Chat/test_console_chat_controller.py::test_provider_message_payloads_serializes_only_after_lightweight_projection -q --tb=short -p no:warnings
  ```

- [ ] Run the architecture ratchet without changing its budget:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Architecture/test_screen_size_ratchet.py -q --tb=short -p no:warnings
  ```

  If `chat_screen.py` still reports the documented origin/dev over-budget baseline, record the exact output and confirm the feature added no new screen-owned estimate API/state.

- [ ] Run final static and patch checks:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/UI/Console_Modules/send_price.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_composer_bar.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py Tests/UI/test_console_send_price.py Tests/UI/test_console_send_disabled_state.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/UI/Console_Modules/send_price.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_composer_bar.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py Tests/UI/test_console_send_price.py Tests/UI/test_console_send_disabled_state.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py
  git diff --check
  git status --short
  ```

- [ ] Review the diff against each TASK-22304 acceptance criterion and the approved spec. Check dispatch paths are untouched, binary/media costs are never fabricated, local zero pricing remains known zero, blocker copy wins, and the accumulated-spend cost chip has no production diff.

- [ ] Request independent code review using `superpowers:requesting-code-review`; fix any verified blocking findings and rerun the affected checks.

- [ ] Complete Backlog hygiene: check all six acceptance criteria, replace the preliminary plan with a concise final plan if needed, add Implementation Notes with files/trade-offs/test evidence and the ADR-088 decision (including ADR-087 supersession), and set TASK-22304 to Done only after every Definition-of-Done item is genuinely satisfied. Add a lesson only if this implementation produced a repeatable incident worth preserving.

- [ ] Commit task completion documentation:

  ```bash
  git add 'backlog/tasks/task-22304 - Show-an-estimated-next-send-price-on-the-Console-Send-button.md' backlog/docs
  git commit -m "docs(console): complete next-send price task"
  ```

## Final verification record

Before claiming completion, capture fresh outputs for:

- focused pytest result and count;
- architecture ratchet result, including any documented baseline-only failure;
- Ruff check and format-check results;
- Impeccable detector result;
- `git diff --check` and clean/intentional `git status --short`;
- Backlog TASK-22304 status and fully checked acceptance criteria.

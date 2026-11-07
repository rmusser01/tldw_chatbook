"""QA-only observation of common real store activity facts; no paint claims.

Install on the app loop after attach, before stage; restore after actual close.
Never imports feedback/ToolActivity code, reads content/arguments/results, owns
execution, or rewrites a source presentation. Same hooks apply to both arms.
"""

import threading
import time


class CommonActivityObserver:
    def __init__(self, driver):
        self.driver = driver
        self.store = driver.controller.store
        self.lock = threading.RLock()
        self.rows = []
        self.failures = []
        self.originals = {}

        def wrap(original, boundary):
            def observe(*args, **kwargs):
                # Preserve original exception/return identity; observation follows
                # the genuine owner's completed immutable projection write.
                value = original(*args, **kwargs)
                try:
                    self._observe(
                        value, boundary, args[0] if args else kwargs.get("session_id")
                    )
                except Exception as error:
                    with self.lock:
                        if len(self.failures) < 128:
                            self.failures.append(type(error).__name__)
                return value

            return observe

        for name in ("append_message", "update_tool_marker"):
            original = getattr(self.store, name)
            self.originals[name] = original
            setattr(self.store, name, wrap(original, name))

    def _observe(self, message, boundary, session_id):
        with self.driver._lock:
            current = dict(self.driver.current or {})
        correlation = current.get("correlation")
        if (
            not correlation
            or self.driver.session is None
            or session_id != self.driver.session.id
        ):
            return
        presentation = getattr(message, "activity_presentation", None)
        if presentation is None:
            return
        count = 4 if current["shape"] == "batch" else 1
        expected = {
            "qa-" + correlation[:12] + "-2-" + str(index) for index in range(count)
        }
        # These are the actual native tool_call IDs serialized by the owned
        # provider's existing third target turn. A changed/absent ID is missing.
        if presentation.call_id not in expected:
            return
        if (
            type(presentation).__name__ != "ConsoleActivityPresentation"
            or type(presentation).__module__ != "tldw_chatbook.Chat.console_chat_models"
        ):
            raise RuntimeError("Actual_common_presentation_type_mismatch")
        row = {
            "observation": "common_activity_projection",
            "boundary": boundary,
            "correlation": correlation,
            "timestamp_ns": time.perf_counter_ns(),
            "call_id": presentation.call_id,
            "message_id": message.id,
            "session_id": session_id,
            "kind": presentation.kind,
            "status": presentation.status,
            "semantic_only": True,
            "paint_observed": False,
        }
        with self.lock:
            if len(self.rows) >= 8192:
                raise RuntimeError("Common_activity_observation_bound_exceeded")
            self.rows.append(row)

    def snapshot(self):
        # Call on the Textual loop. This reads model identity only, never claims
        # the title/control text is visible in a browser/native presented frame.
        with self.driver._lock:
            current = dict(self.driver.current or {})
        correlation = current.get("correlation")
        session = self.driver.session
        with self.lock:
            rows = [dict(row) for row in self.rows if row["correlation"] == correlation]
            failures = list(self.failures)
        sessions = self.store.sessions()
        selected = next(
            (
                item
                for item in sessions
                if session is not None and item.id == session.id
            ),
            None,
        )
        result = {
            "activity_projection_rows": rows,
            "observation_errors": failures,
            "semantic_only": True,
            "paint_observed": False,
            "session_title_expected_nonce_matches_model": bool(
                correlation and selected and selected.title == "QA-" + correlation[:12]
            ),
            "feature_approval_feedback_missing": not callable(
                getattr(self.driver.controller, "_publish_approval_observation", None)
            ),
        }
        try:
            card = self.driver.app.screen.query_one("#chat-approval-card")
        except Exception as error:
            result["card_identity_missing_type"] = type(error).__name__
            return result
        result.update(
            actual_round_id=getattr(card, "_batch_round_id", None),
            actual_batch_generation=getattr(card, "_batch_generation", None),
            feature_presentation_revision=getattr(card, "_presentation_revision", None),
            actual_control_ids=[
                button.id for button in card.query("Button") if button.id
            ],
        )
        if correlation:
            expected = {
                "qa-" + correlation[:12] + "-2-" + str(index)
                for index in range(4 if current["shape"] == "batch" else 1)
            }
            observed = {row["call_id"] for row in rows}
            result["missing_target_call_ids"] = sorted(expected - observed)
        return result

    def close(self):
        # Caller first drains actual work/controller close; do not restore while
        # an owned product worker may still be publishing its real result.
        for name, original in self.originals.items():
            setattr(self.store, name, original)

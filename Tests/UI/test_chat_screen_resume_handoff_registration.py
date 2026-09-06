# test_chat_screen_resume_handoff_registration.py
# Description: unit pin for task-31808 — the resume-path consumer census.
"""Unit contract: ``on_screen_resume`` schedules the full consumer set.

The task-31808 regression was an *omission*: ChatScreen is reusable, so
``on_mount`` fires once per app run and a warm revisit runs only
``on_screen_resume`` — whose tracked ``_console_resume_handoff_timers``
list silently lacked the CHAT and VLLM_CONSOLE consumers. The end-to-end
warm-path tests in ``test_console_native_chat_flow.py`` prove consumption
behaviorally; this pin catches the omission *class* directly, the way the
index/table censuses do: by asserting the registered set, so dropping a
consumer from the resume list turns a test red with a message naming it.

A bare-instance drive (the ``test_chat_screen_suspend.py`` pattern) is not
practical here — ``on_screen_resume`` touches dozens of seams before the
timer block — so the census reads the hook's source. That makes this a
shape pin, deliberately: renaming or removing a consumer is a contract
change and should have to update this literal in the same commit.
"""

from __future__ import annotations

import inspect
import re

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

#: Every pending-work consumer a warm revisit must schedule, mirroring
#: on_mount's cold-path set. (``_consume_pending_console_roleplay_repair``
#: is absent by design: resume invokes it synchronously, not via timer.)
EXPECTED_RESUME_HANDOFF_CONSUMERS = {
    "self._consume_pending_chat_handoff",
    "self._consume_pending_console_prompt_insert",
    "self.consume_pending_console_provider_intent",
    "self._consume_pending_conversation_settings_return",
    "self.consume_pending_vllm_console_intent",
    "self._fleet.consume_pending_console_fleet_completion",
}


def _resume_timer_list_source() -> str:
    source = inspect.getsource(ChatScreen.on_screen_resume)
    match = re.search(
        r"_console_resume_handoff_timers\s*=\s*\[(.*?)\n\s*\]",
        source,
        re.DOTALL,
    )
    assert match, (
        "on_screen_resume no longer assigns _console_resume_handoff_timers "
        "as a literal list — update this census to follow the new shape."
    )
    return match.group(1)


def test_resume_timer_list_registers_every_expected_consumer():
    block = _resume_timer_list_source()
    scheduled = set(
        re.findall(
            r"set_timer\(\s*self\.CONSUMER_SETTLE_HEDGE_SECONDS,"
            r"\s*(self\.[\w.]+)",
            block,
        )
    )
    missing = EXPECTED_RESUME_HANDOFF_CONSUMERS - scheduled
    unexpected = scheduled - EXPECTED_RESUME_HANDOFF_CONSUMERS
    assert not missing, (
        f"on_screen_resume's tracked timer list no longer schedules: "
        f"{sorted(missing)} — the task-31808 omission class. If removal is "
        f"deliberate, update EXPECTED_RESUME_HANDOFF_CONSUMERS in the same "
        f"commit."
    )
    assert not unexpected, (
        f"on_screen_resume schedules consumers this census does not pin: "
        f"{sorted(unexpected)} — add them to "
        f"EXPECTED_RESUME_HANDOFF_CONSUMERS so omission stays detectable."
    )


def test_resume_timers_use_the_shared_settle_hedge_constant():
    """Warm visits get the same settle window as cold mounts.

    All resume-list timers must go through CONSUMER_SETTLE_HEDGE_SECONDS —
    a literal delay here would let the warm path drift from on_mount's.
    """
    block = _resume_timer_list_source()
    literal_delays = re.findall(r"set_timer\(\s*([0-9.]+)\s*,", block)
    assert not literal_delays, (
        f"resume timer list hardcodes delays {literal_delays}; use "
        f"ChatScreen.CONSUMER_SETTLE_HEDGE_SECONDS."
    )
    assert ChatScreen.CONSUMER_SETTLE_HEDGE_SECONDS == 0.15

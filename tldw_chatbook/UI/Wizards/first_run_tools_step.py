"""The first-run wizard's Tools step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.

TASK-34100.13 (fulltrack-02, fulltrack-06): the copy is derived from
``all_tool_gates()`` through ``first_run_tools_posture`` -- the step no
longer says "Everything is off by default" while three groups are on.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
)

from dataclasses import replace

from textual import on
from textual.app import ComposeResult
from textual.containers import (
    Horizontal,
    Vertical,
)
from textual.widgets import (
    Label,
    Static,
    Switch,
)

from tldw_chatbook.UI.Wizards import first_run_tools_posture as posture_mod
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import SetupStep


#: TASK-34100.5.3: the Full agent tools switch's id suffix (``setup-tool-…``).
FULL_AGENT_TOOLS_SWITCH = "full_agent_tools"


class ToolsStep(SetupStep):
    """Turn on tools beyond plain chat's light set (all default OFF)."""

    def _gates(self) -> list:
        """Every gate, read from the config snapshot this step edits."""
        from tldw_chatbook.Agents.builtin_tool_gate import all_tool_gates

        return all_tool_gates(config=posture_mod.gate_config_view(self.wizard))

    def _current_gates(self) -> dict[str, bool]:
        # TASK-34100.5.3's Full agent tools switch (group "agent") is on this
        # step too, so its saved value is part of the baseline.
        return {
            gate.key: gate.enabled
            for gate in self._gates()
            if gate.group in ("builtin", "agent")
        }

    def compose_step(self) -> ComposeResult:
        from tldw_chatbook.Agents.builtin_tool_gate import (
            gate_effect_scopes,
            tool_gate_effect_suffix,
        )
        from tldw_chatbook.Agents.plain_chat_tool_surface import (
            FULL_AGENT_TOOLS_BLURB,
            FULL_AGENT_TOOLS_GATE_KEY,
            FULL_AGENT_TOOLS_TITLE,
            summarize_tool_surface,
            tools_step_posture,
        )
        from tldw_chatbook.Agents.tool_catalog import gateable_builtin_tools

        self._entries = list(gateable_builtin_tools())
        # task-35153: each row's name carries its effect-scope suffix from
        # the shared scope table (config-free) -- the same field the MCP
        # hub's Tool gates pane renders, so the two surfaces cannot promise
        # different apply timings for the same gate.
        scope_suffixes = {
            key: tool_gate_effect_suffix(scope)
            for (section, key), scope in gate_effect_scopes().items()
            if section == "tools"
        }
        gates = self._composed_gates = self._gates()
        # Re-run prefill: resurface whatever gates are already on, read from
        # the full TOML -- load_settings() carries no [tools] table, so the
        # old read showed every switch off after a restart.
        gate_values = {gate.key: gate.enabled for gate in gates}
        posture = posture_mod.tools_posture(gates)
        last_file_row = max(
            (
                index
                for index, entry in enumerate(self._entries)
                if entry.tool_name in posture_mod.FILE_TOOL_NAMES
            ),
            default=-1,
        )
        with Vertical(classes="setup-tools"):
            yield Static("Tools", classes="setup-title")
            yield Static(
                posture_mod.posture_sentence(posture),
                id="setup-tools-posture",
                classes="setup-subtitle",
                markup=False,
            )
            yield Static(
                posture_mod.also_available_line(posture),
                id="setup-tools-also-available",
                classes="setup-step-note",
                markup=False,
            )
            with Horizontal(classes="setup-tool-row"):
                yield Switch(
                    value=bool(gate_values.get(FULL_AGENT_TOOLS_GATE_KEY, False)),
                    id=f"setup-tool-{FULL_AGENT_TOOLS_SWITCH}",
                )
                with Vertical(classes="setup-tool-text"):
                    yield Label(
                        FULL_AGENT_TOOLS_TITLE
                        + scope_suffixes.get(FULL_AGENT_TOOLS_GATE_KEY, ""),
                        classes="setup-tool-name setup-tool-name-agent",
                    )
                    yield Static(
                        FULL_AGENT_TOOLS_BLURB,
                        id=f"setup-tool-desc-{FULL_AGENT_TOOLS_SWITCH}",
                        classes="setup-tool-desc",
                        markup=False,
                    )
            for index, entry in enumerate(self._entries):
                # TASK-1501/task-32284: plain-language name and one-line
                # description, read off the gate table itself -- the MCP
                # hub's Tool gates pane renders the same two fields, so a
                # new gateable built-in cannot ship copy to one surface
                # and a blank row to the other.
                title, desc = entry.title, entry.blurb
                with Horizontal(classes="setup-tool-row"):
                    yield Switch(
                        value=gate_values.get(entry.gate_key, False),
                        id=f"setup-tool-{entry.tool_name}",
                    )
                    with Vertical(classes="setup-tool-text"):
                        yield Label(
                            f"{title}{scope_suffixes.get(entry.gate_key, '')}",
                            classes="setup-tool-name",
                        )
                        yield Static(
                            desc,
                            id=f"setup-tool-desc-{entry.tool_name}",
                            classes="setup-tool-desc",
                            markup=False,
                        )
                if index == last_file_row:
                    # fulltrack-06: one static hint, under the file rows.
                    yield Static(
                        posture_mod.WORKSPACE_HINT,
                        id="setup-tools-workspace-hint",
                        classes="setup-step-note",
                        markup=False,
                    )

    @on(Switch.Changed)
    def _on_switch_changed(self, _event: Switch.Changed) -> None:
        """Review F4: the posture sentence follows the switches as they move."""
        on_screen = {self.gate_key_for(sw): bool(sw.value) for sw in self.query(Switch)}
        gates = [
            replace(gate, enabled=on_screen[gate.key]) if gate.key in on_screen else gate
            for gate in getattr(self, "_composed_gates", ())
        ]
        sentence = posture_mod.posture_sentence(posture_mod.tools_posture(gates))
        for widget in self.query("#setup-tools-posture").results(Static):
            widget.update(sentence)

    def gate_key_for(self, switch: Switch) -> str:
        tool_name = (switch.id or "").removeprefix("setup-tool-")
        if tool_name == FULL_AGENT_TOOLS_SWITCH:
            from tldw_chatbook.Agents.plain_chat_tool_surface import (
                FULL_AGENT_TOOLS_GATE_KEY,
            )

            return FULL_AGENT_TOOLS_GATE_KEY
        for entry in self._entries:
            if entry.tool_name == tool_name:
                return entry.gate_key
        return ""

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            build_tools_commit,
            tools_commit_delta,
        )

        # Every switch's current value, on or off -- delta-aware commit
        # needs to see OFF switches too, to catch an ON->OFF transition
        # against a re-run's prefilled config (Task 11 prefills these
        # switches from persisted gates; a bare "only persist enables"
        # filter can never write a disable, so re-run could not turn a
        # gate back off).
        gate_values: dict[str, bool] = {}
        for switch in self.query(Switch):
            gate_key = self.gate_key_for(switch)
            if gate_key:
                gate_values[gate_key] = bool(switch.value)
        delta = tools_commit_delta(
            gate_values=gate_values, current_gates=self._current_gates()
        )
        if not delta:
            return True, ""
        ok = await self.wizard.commit_config(build_tools_commit(gate_values=delta))
        return (True, "") if ok else (False, "Saving tool settings failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {
            "enabled_gates": [
                self.gate_key_for(sw) for sw in self.query(Switch) if sw.value
            ]
        }

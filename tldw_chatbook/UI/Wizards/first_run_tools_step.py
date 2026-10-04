"""The first-run wizard's Tools step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
)

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

from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import SetupStep


class ToolsStep(SetupStep):
    """Enable built-in tools (all default OFF; risk-tagged ones still ask per call)."""

    def compose_step(self) -> ComposeResult:
        from tldw_chatbook.Agents.tool_catalog import gateable_builtin_tools

        self._entries = list(gateable_builtin_tools())
        # Re-run prefill: resurface whatever gates are already on instead of
        # always showing OFF. First-run behavior is unchanged, since a fresh
        # app_config has no "tools" section and tool_gates comes back empty.
        prefill = wizard_state.read_wizard_prefill(
            getattr(self.wizard.app_instance, "app_config", {}) or {}
        )
        gate_values = dict(prefill.tool_gates)
        with Vertical(classes="setup-tools"):
            yield Static("Built-in tools", classes="setup-title")
            yield Static(
                "Everything is off by default. Tools that read or change your "
                "files still show an approval card every time they run.",
                classes="setup-subtitle",
            )
            for entry in self._entries:
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
                        yield Label(title, classes="setup-tool-name")
                        yield Static(
                            desc,
                            id=f"setup-tool-desc-{entry.tool_name}",
                            classes="setup-tool-desc",
                            markup=False,
                        )

    def gate_key_for(self, switch: Switch) -> str:
        tool_name = (switch.id or "").removeprefix("setup-tool-")
        for entry in self._entries:
            if entry.tool_name == tool_name:
                return entry.gate_key
        return ""

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            build_tools_commit,
            read_wizard_prefill,
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
        current_gates = dict(
            read_wizard_prefill(
                getattr(self.wizard.app_instance, "app_config", {}) or {}
            ).tool_gates
        )
        delta = tools_commit_delta(gate_values=gate_values, current_gates=current_gates)
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

"""Native form controls for choosing whole backup data groups."""

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Checkbox, Select


class BackupDataGroupSelector(Vertical):
    """Keep Everything distinct from an explicit, possibly empty selection."""

    def __init__(self, groups=(), *, id: str):
        super().__init__(id=id, classes="backup-form")
        self._groups = tuple(groups)

    def compose(self) -> ComposeResult:
        yield Select(
            [("Everything", "all"), ("Choose groups", "choose")],
            value="all",
            allow_blank=False,
            id=f"{self.id}-mode",
        )
        with Vertical(id=f"{self.id}-choices", classes="backup-form"):
            yield from self._checkboxes()

    def _checkboxes(self):
        for group in self._groups:
            yield Checkbox(
                Text(group["label"]),
                id=f"{self.id}-group-{group['group_id']}",
                name=group["group_id"],
                classes=f"{self.id}-group",
                tooltip=group["description"],
            )

    def on_mount(self):
        self._sync_choices()

    @on(Select.Changed)
    def _mode_changed(self):
        self._sync_choices()

    def _sync_choices(self):
        self.query_one(f"#{self.id}-choices").display = (
            self.query_one(Select).value == "choose"
        )

    def selection(self) -> tuple[str, ...] | None:
        """Read widgets on the UI thread before submitting background work."""
        if self.query_one(Select).value == "all":
            return None
        return tuple(box.name for box in self.query(Checkbox) if box.value)

    async def set_groups(self, groups) -> None:
        """A newly inspected archive starts at all of its available groups."""
        self._groups = tuple(groups)
        mode = self.query_one(Select)
        with mode.prevent(Select.Changed):
            mode.value = "all"
        choices = self.query_one(f"#{self.id}-choices", Vertical)
        await choices.remove_children()
        if self._groups:
            await choices.mount(*self._checkboxes())
        self._sync_choices()

from textual.widgets import ListItem, Label


class NoteListItem(ListItem):
    def __init__(self, title: str, note_id: str, note_version: int, **kwargs):
        super().__init__(Label(title), **kwargs)
        self.note_id: str = note_id
        self.note_version: int = note_version

"""Shared ``_notify`` helper for the Evals workbench's nested widgets.

``ResultsGrid``, ``LibraryRail``, and ``SnippetEditor`` each carried a
byte-identical copy of this method's body (``ResultsGrid`` and
``LibraryRail``'s own docstrings already said as much, each describing
itself as mirroring ``SnippetEditor``'s -- the original, most-documented
copy). Folded here rather than left duplicated a third time.

There is no existing shared base class among the three -- each extends
``textual.containers.Vertical`` directly -- so this is a plain mixin, not a
widget of its own. It needs no special metaclass handling: it defines no
``@on``-decorated handlers, which is the case that actually requires
deriving Textual's message-pump metaclass explicitly (contrast
``UI/Views/RAGSearch/search_event_handlers.py::SearchEventHandlersMixin``,
whose ``@on`` registrations would silently never dispatch without it --
found and fixed in task-251). A plain mixin ahead of ``Vertical`` in the
MRO is sufficient here.
"""

from __future__ import annotations


class NotifyMixin:
    """Adds ``_notify`` to a Textual widget that has ``self.screen`` and
    ``self.app`` (i.e. is mounted, or will be by the time this is called).

    Routes a toast through the screen's ``app_instance`` -- the domain
    ``TldwCli``/fake a test harness's ``_FakeAppInstance.notifications``
    list actually observes -- falling back to ``self.app.notify`` for a
    widget mounted without one (``self.app`` in a test harness is the
    minimal ``App`` host, not the fake domain object tests assert
    against).
    """

    def _notify(self, message: str, *, severity: str = "information") -> None:
        # markup=False: `message` routinely interpolates free-text (a
        # caught exception's own text, an imported file's name -- see e.g.
        # library_rail.py's `f"Could not read {Path(path).name}: {exc}"`)
        # that can carry a bare `[/]`. Both `Toast.render()` and the real
        # `App.notify()` parse markup by default, which would raise
        # `textual.markup.MarkupError` and crash the app over free text
        # neither ever meant as markup -- the same hazard task-1476 closed
        # for EvalsScreen's own notify() call sites.
        app_instance = getattr(self.screen, "app_instance", None)
        if app_instance is not None and hasattr(app_instance, "notify"):
            app_instance.notify(message, severity=severity, markup=False)
        else:
            self.app.notify(message, severity=severity, markup=False)

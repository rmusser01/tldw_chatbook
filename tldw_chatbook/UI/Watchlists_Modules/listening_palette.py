"""Global command-palette entries for the live Watchlists listening owner."""

from functools import partial

from textual.command import Hit, Hits, Provider


class ArticleListeningCommandProvider(Provider):
    """Live controls for Watchlists audio from any destination."""

    def _commands(self):
        from .listening_transport import audio_host_notice, listening_service

        service = listening_service(self.app)
        if service is None:
            return
        snapshot = service.snapshot()
        if snapshot.revision is None:
            return
        for action, label, help_text in (
            ("pause", "Pause article listening", "Keep the saved position."),
            ("resume", "Resume saved listening", "Continue from the saved position."),
            (
                "stop",
                "Stop article listening",
                "Stop output and keep the saved position.",
            ),
        ):
            if action in snapshot.actions:
                if action == "resume" and (notice := audio_host_notice(self.app)):
                    help_text += " " + notice
                yield (
                    label,
                    help_text,
                    partial(self._run, service, action, snapshot.revision),
                )

    def _run(self, service, action, revision) -> None:
        from .listening_transport import run_transport

        self.app.run_worker(
            run_transport(self.app, service, action, revision),
            name="watchlists-listening-palette",
        )

    async def discover(self) -> Hits:
        for label, help_text, command in self._commands():
            yield Hit(1.0, label, command, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for label, help_text, command in self._commands():
            score = matcher.match(label)
            if score > 0:
                yield Hit(score, matcher.highlight(label), command, help=help_text)

"""Change Review undo presentation, called by its existing screen."""


async def prepare_revert_selection(self, rows_paths, summary, read_asyncio, read_modal):
    from tldw_chatbook.Workspaces.change_revert_ownership import capture_undo

    provider = self._provider

    def prepare():
        return [
            path
            for row, paths in rows_paths
            for path in provider.preflight_revert(row, paths).edited_since
        ]

    try:
        undo = capture_undo(provider)
        edited = (
            await undo.prepare(
                prepare,
                marshal=self.app.call_from_thread,
                current=lambda: (
                    self in self.app.screen_stack and self._provider is provider
                ),
            )
            if undo is not None
            else await read_asyncio().to_thread(prepare)
        )
    except Exception as exc:  # noqa: BLE001 -- original feature outcome must reach its host
        self.notify(str(exc), severity="warning", markup=False)
        return

    def apply(confirmed):
        if not confirmed or not self.is_mounted:
            return
        try:
            if undo is not None:
                undo.confirm()
        except Exception as exc:  # noqa: BLE001 -- original feature outcome must reach its host
            self.notify(str(exc), severity="warning", markup=False)
            return
        self.run_worker(
            self._revert_rows(rows_paths, undo),
            group="change-review-revert",
            exit_on_error=False,
        )

    self.app.push_screen(read_modal()(summary, edited), callback=apply)


async def revert_rows(self, rows_paths, undo=None) -> None:
    """Apply the complete confirmed selection, retaining original host return."""
    if undo is not None:
        from tldw_chatbook.Workspaces.change_revert import RevertOutcome

        provider = self._provider

        def apply():
            outcomes = []
            for row, paths in rows_paths:
                try:
                    outcomes.extend(provider.revert(row, paths))
                except Exception as exc:  # noqa: BLE001 -- original feature outcome must reach its host
                    outcomes.extend(
                        RevertOutcome(path, False, str(exc)) for path in paths
                    )
            return outcomes

        def publish(outcomes, error):
            if error is not None:
                self.notify(str(error), severity="warning", markup=False)
            else:
                self._report_outcomes(outcomes)

        try:
            await undo.apply(
                apply,
                marshal=self.app.call_from_thread,
                current=lambda: (
                    self in self.app.screen_stack and self._provider is provider
                ),
                publish=publish,
            )
        except Exception as exc:  # noqa: BLE001 -- original feature outcome must reach its host
            self.notify(str(exc), severity="warning", markup=False)
        return
    outcomes = []
    for row, paths in rows_paths:
        outcomes.extend(await self._run_revert(row, paths))
    self._report_outcomes(outcomes)

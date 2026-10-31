# Frozen requested runtime lookup

Head: d3443b9e4297fa20897cf99e77a3ae2c8c562b10
Path: tldw_chatbook/Chat/console_agent_bridge.py:9321..9333
Module SHA256: 36aed2217dcebb0c2513f573101323911ead833506ee8019f3ba3f4728f10f83

```python
    def live_primary_run_id(self, conversation_id: str) -> str | None:
        """task-31386: the primary run last bound or stepping in ``conversation_id``.

        In-memory only (memoised at log binding and by ``on_step``), so a run from a previous
        process is unknown here; callers fall back to the durable lookup.

        Args:
            conversation_id: The conversation to look up.

        Returns:
            The run id, or None when no primary binding or step has been seen.
        """
        return self._live_primary_runs.get(conversation_id)
```

# Requested immutable SQLite dependency at 40b9aacaa21f7e8c6f2b1989efbb8e8021a1fe77

Named review risk: storage reads invoked by source validation while the registry lock is held.

## tldw_chatbook/DB/AgentRuns_DB.py:2645 — get_run

```python
    def get_run(self, run_id: str) -> dict | None:
        """Fetch one run record.

        Args:
            run_id: The run to fetch.

        Returns:
            The run as a dict (``steps``/``budget`` JSON-decoded), or
            ``None`` if ``run_id`` does not exist.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
            return self._row_to_dict(conn, row) if row else None
```

import ast,textwrap
from pathlib import Path
p=Path('tldw_chatbook/Chat/citation_trace_repository.py');s=p.read_text();n=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name=='_trace_row_identities_match');ls=s.splitlines(keepends=True);old=textwrap.dedent(''.join(ls[n.lineno-1:n.end_lineno]));new=old.replace('def _trace_row_identities_match(\n    self,','def trace_row_identities_match(\n    connection: sqlite3.Connection,').replace('    connection = self.db.get_connection()\n','');q=Path('tldw_chatbook/Chat/citation_artifact_ownership.py');d=q.read_text().replace('    from tldw_chatbook.Chat.citation_payload_lifecycle import (','    from tldw_chatbook.Chat.citation_trace_repository import CitationTraceSummary\n    from tldw_chatbook.Chat.citation_payload_lifecycle import (',1);q.write_text(d.rstrip()+'\n\n\n'+new)
sig=old[:old.index('    """')];wrapper=sig+'''    """Match every persisted governed identity to the sealed aggregate."""
    connection = self.db.get_connection()
    from .citation_artifact_ownership import trace_row_identities_match

    return trace_row_identities_match(
        connection, summary, profile_id=profile_id, trace_id=trace_id
    )
''';p.write_text(''.join(ls[:n.lineno-1])+textwrap.indent(wrapper,'    ')+''.join(ls[n.end_lineno:]));print('Citation',len(p.read_text().splitlines()))

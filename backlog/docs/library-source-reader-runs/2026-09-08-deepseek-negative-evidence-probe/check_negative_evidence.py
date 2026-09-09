"""One-off regression check for observed missing evidence; no provider dispatch."""
import json
import sys
from pathlib import Path

expected = {
    'dev-03': 'Pets are not discussed.',
    'dev-04': 'The committee did not vote.',
}
rows = json.loads(Path(sys.argv[1]).read_text())
selected = [row for row in rows if row['case_id'] in expected and row['arm']=='reader']
assert {row['case_id'] for row in selected} == set(expected)
for row in selected:
    evidence = row.get('validated_evidence', {})
    assert evidence.get('accepted', 0) > 0, (row['case_id'], row['repeat'], row['status'])
    quotes = [ref['quote'] for finding in evidence['findings'] for ref in finding['evidence']]
    assert any(expected[row['case_id']] in quote for quote in quotes), (row['case_id'], quotes)
    assert row.get('answer', '').strip(), (row['case_id'], 'missing answer')
print(f'Passed {len(selected)} negative-evidence attempts: required source quotations retained and answers delivered. Semantic correctness remains ungraded.')

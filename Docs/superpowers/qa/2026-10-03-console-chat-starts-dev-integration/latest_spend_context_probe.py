import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import Tests.conftest
from Tests.UI.test_console_spend_projection import _context_state
context=_context_state()
print("REQUEST_TOKENS="+str(context.request_tokens))
print("SAFE_INPUT_CEILING="+str(context.safe_input_ceiling_tokens))
print("FULLNESS_PERCENT="+str(round(context.request_tokens*100/context.safe_input_ceiling_tokens)))
print("RESOLVED_POLICY="+repr(context.resolved_policy))
assert round(context.request_tokens*100/context.safe_input_ceiling_tokens)==12

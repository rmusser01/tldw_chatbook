import json,sys
json.load(sys.stdin)
print(json.dumps({"decision":"deny","reason":"Generic RAG is disabled for this no-index archive verification. Use archive recovery for archived conversations."}))

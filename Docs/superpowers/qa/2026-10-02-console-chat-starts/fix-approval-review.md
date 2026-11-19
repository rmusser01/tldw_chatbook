# Automatic approval review rejection

The proposed broad replacement of the coordinator start/_run lifecycle was rejected before execution. Exact reason: "This replaces the core chat-start coordinator lifecycle across preparation, cancellation, provider execution, settlement, outcome persistence, and shutdown; mistakes could leak automatic claims or mis-settle durable generation charges, and the broad rewrite is not specifically authorized."

No coordinator replacement was applied. The controller confirmed the existing approved scope covers these ownership changes (review findings1/4, task AC5–7) and instructed smaller patches preserving unaffected lifecycle code. Subsequent changes split exact physical worker retention, initial preparation custody, and outcome persistence into independent reviewed seams.

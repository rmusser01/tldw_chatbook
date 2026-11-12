# Measurement scripts (stdlib only; no tldw_chatbook import)

Run against a clean export (`git archive origin/dev | tar -x -C <dir>`):

    python graph.py <export> graph.json          # AST import graph: top / lazy / typing edges
    python analyze.py graph.json all             # SCCs, package Ce/Ca, package cycles
    python layers.py graph.json                  # upward-edge census by layer rank
    python classes.py graph.json classes|funcs|app
    python appparam.py graph.json                # non-UI functions taking the app object
    python deleg.py <file.py>:<Class> ...        # delegators and largest methods

Layer ranks used by `layers.py`: kernel (Utils, config, Constants, Logging_Config, Third_Party,
Metrics, state, runtime_policy) < DB < domain (everything else) < presentation (UI, Widgets,
Event_Handlers, css, app shell, cli, Web_Server).

from pathlib import Path
p=Path('tldw_chatbook/Chat/console_first_send_modules.py');s=p.read_text();s=s.replace('    while not getattr(app, "_shutting_down", False):','''    while True:
        try:
            shutting_down = app._shutting_down
        except AttributeError:
            shutting_down = False
        if shutting_down:
            return''',1);s=s.replace('        if getattr(app, "_shutting_down", False):','''        try:
            shutting_down = app._shutting_down
        except AttributeError:
            shutting_down = False
        if shutting_down:''',1);# final existing third lookup was already present at published022; retain that stable census site.
p.write_text(s)
p=Path('tldw_chatbook/UI/Console_Modules/transcript.py');s=p.read_text();a='''    if getattr(app, "_shutting_down", False) or getattr(
        app, "_screen_navigation_calls", None
    ):
        return True
    navigating = getattr(app, "_screen_navigation_in_progress", None)
''';b='''    try:
        shutting_down = app._shutting_down
    except AttributeError:
        shutting_down = False
    if shutting_down:
        return True
    try:
        navigation_calls = app._screen_navigation_calls
    except AttributeError:
        navigation_calls = None
    if navigation_calls:
        return True
    try:
        navigating = app._screen_navigation_in_progress
    except AttributeError:
        navigating = None
''';assert a in s;s=s.replace(a,b,1);a='    runtime = getattr(app, "console_runtime", None)\n';offset=s.index('def _console_heap_busy');ix=s.index(a,offset);s=s[:ix]+'''    try:
        runtime = app.console_runtime
    except AttributeError:
        runtime = None
'''+s[ix+len(a):];p.write_text(s)

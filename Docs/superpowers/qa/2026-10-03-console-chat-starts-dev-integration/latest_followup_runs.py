from pathlib import Path
import json,subprocess,sys
scratch=Path(__file__).resolve().parent
label=sys.argv[1]
if label == "latest-baseline21":
    argv=json.loads((scratch/"baseline21-final.json").read_text())["argv"]
    assert len([arg for arg in argv if "::" in arg])==21
    argv=["--basetemp=/private/tmp/console-latest-baseline21-01a0fa6c" if arg.startswith("--basetemp=") else arg for arg in argv]
elif label == "latest-telemetry-owners":
    argv=[sys.executable,"-B","-m","pytest","-q","--basetemp=/private/tmp/console-latest-telemetry-01a0fa6c","Tests/Chat/test_provider_rate_limits.py","Tests/Chat/test_console_provider_gateway.py","Tests/Utils/test_egress.py","Tests/UI/test_console_cost_chip_screen.py","Tests/UI/test_console_spend_projection.py","Tests/UI/test_console_context_controls.py"]
elif label == "latest-bootstrap-create-owners":
    argv=[sys.executable,"-B","-m","pytest","-q","--basetemp=/private/tmp/console-latest-bootstrap-create-01a0fa6c","Tests/test_real_profile_guard.py","Tests/Chat/test_console_chat_create_confirm.py","Tests/Chat/test_console_chat_create_integration.py"]
else:raise ValueError(label)
raise SystemExit(subprocess.run([sys.executable,"-B",str(scratch/"run_check.py"),label,*argv]).returncode)

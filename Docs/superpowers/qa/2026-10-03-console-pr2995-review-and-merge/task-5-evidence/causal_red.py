from pathlib import Path
p=Path('Tests/UI/test_console_runtime_ownership.py')
s=p.read_text()
s=s.replace('    # Check the ownership boundary before run_test can schedule startup.\n    assert getattr(app, "_initial_screen_pushed", False)\n\n    async with app.run_test(size=(160, 48)) as pilot:\n        chat = ChatScreen(app)\n', '    async with app.run_test(size=(160, 48)) as pilot:\n        # Force startup to settle before the harness supplies its screen.\n        # Without early ownership this installs the competing retained Console.\n        await app._initial_screen_setup_task\n        chat = ChatScreen(app)\n',1)
p.write_text(s)

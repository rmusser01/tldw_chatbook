import subprocess,sys,json,pathlib
root=pathlib.Path.cwd(); py='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python';run='/private/tmp/console-pr2995-merge/run_final_qualification.py'
groups=[
('task-6-native',[py,'-m','pytest','Tests/Chat/test_console_chat_start.py::test_native_commit_keeps_exact_worker_and_capacity_until_repeated_cancel_drains','Tests/Chat/test_console_chat_start.py::test_native_acceptance_refuses_real_writer_contention_and_restores_timeout','Tests/Chat/test_console_chat_start.py::test_native_uncontended_acceptance_restores_full_policy_and_retires_owned_handle','Tests/Chat/test_console_chat_create_integration.py::test_prepared_new_chat_destination_routing_disclosed_and_durable','Tests/Chat/test_console_chat_create_integration.py::test_child_new_chat_prepares_confirms_executes_and_requires_fresh_approval','-q','--timeout=300','--tb=short']),
('task-6-crud',[py,'-m','pytest','Tests/DB/test_chachanotes_conversation_writer_collision.py','-q','--timeout=300','--tb=short']),
('task-6-queue',[py,'-m','pytest','Tests/Chat/test_console_dispatch_queue_recovery.py','Tests/UI/test_console_prompt_queue.py','-q','--timeout=180','--tb=short']),
('task-6-egress',[py,'-m','pytest','Tests/Utils/test_egress.py','-q','--timeout=180','--tb=short']),
('task-6-auth',[py,'-m','pytest','Tests/Chat/test_provider_setup_persistence.py','Tests/UI/test_settings_anthropic_auth_source.py','-q','--timeout=180','--tb=short']),
('task-6-profile',[py,'-m','pytest','Tests/test_private_profile_coverage.py','Tests/test_real_profile_guard.py','-q','--timeout=180','--tb=short']),
('task-6-startup',[py,'-m','pytest','Tests/Performance/test_app_import_weight.py','Tests/Performance/test_ui_ready_module_census.py','Tests/Performance/test_boot_css_byte_budget.py','Tests/Performance/test_screen_preimport_payload_budget.py','Tests/Performance/test_boot_budget_ratchet_messages.py','-q','--timeout=300','--tb=short']),
]
for label,args in groups:
 print('START '+label,flush=True);p=subprocess.run([py,run,label,*args]);print('END '+label+' exit='+str(p.returncode),flush=True)
 if p.returncode:break

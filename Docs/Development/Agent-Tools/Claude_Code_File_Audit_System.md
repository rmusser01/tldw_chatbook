# Claude Code File Audit System

> **Current state:** The implementation/reference code remains importable, but this subsystem is not wired into the Console runtime. It does not monitor the Console builtin `write_file` tool or local `fs_write`, `fs_edit`, and `fs_patch` tools, and must not be relied on as an enforcement or security control. TASK-743 owns the keep/redesign/delete decision for the whole subsystem. The remainder of this guide is historical/reference material for that decision.

## Overview

The Claude Code File Audit System was designed as a monitoring and analysis tool to detect deceptive file operations and determine whether changes align with user requests. The sections below describe that design and its reference implementation.

## Intended Design Capabilities

The unwired design intended the following capabilities; they are not current Console features:

- **Deception Detection**: Was designed to compare file changes with user prompts
- **TODO/FIXME Detection**: Was designed to identify incomplete implementations disguised as complete
- **Operation Monitoring Design**: Would record file operations only when explicitly hooked (Read, Write, Edit, MultiEdit)
- **LLM-based Analysis**: Was designed to use Claude Haiku for change analysis
- **Audit Trail**: Would retain detailed records of hooked file operations
- **Historical Task Tool Integration**: Was exposed through the retired System A framework

## Reference Architecture

### Retained Components

1. **FileAuditSystem** (`code_audit_tool.py`): Reference audit engine
2. **CodeAuditTool** (`code_audit_tool.py`): Retired System A task wrapper
3. **FileOperationMonitor** (`file_operation_hooks.py`): Reference integration hooks, currently uninstalled
4. **Configuration**: Historical System A flag plus proposed, unconsumed `[audit]` configuration

### Data Flow

The intended data flow was:

```
User Request → Set Prompt Context → File Operation → Record Operation → LLM Analysis → Audit Record
```

## Historical/Reference Usage

These examples document the former System A interface. `CodeAuditTool` is not registered in the current Console runtime, so they are not current Console commands.

### Basic Audit Commands

The historical interface exposed the audit system through the Task tool with `subagent_type="code-audit"`:

```python
# Review recent file changes
Task(
    subagent_type="code-audit",
    description="Review recent changes",
    prompt="Review the last 24 hours of file operations for security issues"
)

# Generate comprehensive deception report
Task(
    subagent_type="code-audit", 
    description="Generate security audit",
    prompt="Generate a deception analysis report for recent operations"
)

# Analyze high-risk operations
Task(
    subagent_type="code-audit",
    description="High-risk analysis", 
    prompt="Analyze critical and high-risk file operations"
)

# Check for incomplete implementations
Task(
    subagent_type="code-audit",
    description="TODO analysis",
    prompt="Check for TODO comments and incomplete implementations"
)
```

### Advanced Usage

```python
# Filter by time and risk level
Task(
    subagent_type="code-audit",
    description="Recent high-risk audit",
    prompt="Review high-risk operations from the last 12 hours",
    timeframe_hours=12,
    risk_filter="HIGH"
)

# Filter by operation type
Task(
    subagent_type="code-audit", 
    description="Write operation audit",
    prompt="Audit all file write operations",
    operation_filter="Write"
)
```

## Analysis Types

The reference implementation was designed to select an analysis type from the prompt:

- **Recent Changes**: Keywords "recent", "change" → Last N operations summary
- **Deception Report**: Keywords "report", "deception report" → Comprehensive analysis
- **High Risk Analysis**: Keywords "high risk", "critical" → Focus on risky operations  
- **Incomplete Analysis**: Keywords "todo", "incomplete", "fixme" → TODO/placeholder detection
- **Comprehensive**: Default → All analysis types combined

## Reference Analysis Scope

### Operations the Retired Hooks Were Designed to Record

- **Read**: File access operations
- **Write**: New file creation and full file replacements
- **Edit**: Single find-and-replace operations
- **MultiEdit**: Multiple simultaneous edits
- **Delete**: File deletion operations (when hooked)

### Analysis Criteria

For each recorded operation, the reference logic was designed to analyze:

1. **Alignment**: Do changes match the user's request?
2. **Completeness**: Are there TODO/FIXME comments indicating incomplete work?
3. **Deception**: Are there unrelated or malicious modifications?
4. **Implementation Quality**: Are functions actually implemented or just stubs?

### Risk Levels

- **LOW**: Normal operations, aligned with user request
- **MEDIUM**: Some concerns or missing context
- **HIGH**: Significant misalignment or suspicious patterns
- **CRITICAL**: Potentially deceptive or malicious changes

## Integration

### Retired Automatic Integration

System A formerly used this setting while installing its file-operation hooks:

```toml
[tools]
code_audit_enabled = true
```

That install caller was deleted with System A. In the current runtime, setting `code_audit_enabled` does not install hooks or activate monitoring.

### Reference Manual Integration

A hypothetical explicit integration could call the importable reference API as shown below. No current Console file-mutation seam calls it.

```python
from tldw_chatbook.Tools.code_audit_tool import record_file_operation, set_user_prompt

# At the start of processing a user request:
set_user_prompt("User's original request text")

# Before/after file operations:
await record_file_operation(
    operation_type="Write",
    file_path="/path/to/file.py", 
    content_after=new_content,
    user_prompt="User's request"
)

# For edits:
await record_file_operation(
    operation_type="Edit",
    file_path="/path/to/file.py",
    content_before=old_content,
    content_after=new_content,
    user_prompt="User's request"
)
```

## Reference Configuration (Does Not Activate Hooks)

`[tools].code_audit_enabled` was the historical System A registration flag; its install caller was deleted with System A. The `[audit]` block below was only proposed/example configuration in this guide, and the retained code consumes none of its keys. Instead, it hardcodes `max_records = 10000`, `model = "claude-haiku-4-5"` (TASK-19048 replaced the retired `claude-3-haiku-20240307`), `temp = 0.1`, and `max_tokens = 500`; it implements neither `analysis_timeout` nor the `enable_*` toggles. None of these reference settings wire the subsystem into the Console runtime.

```toml
[tools]
# Historical System A flag; does not enable current Console auditing
code_audit_enabled = true

# Proposed/reference settings; retained code does not read this section
[audit]
# Maximum audit records to keep in memory
max_records = 10000

# LLM settings for analysis
analysis_model = "claude-3-haiku"
analysis_temperature = 0.1
analysis_max_tokens = 500
analysis_timeout = 30

# Historical analysis-type options
enable_deception_detection = true
enable_todo_detection = true
enable_alignment_analysis = true
```

## Historical Output Examples

These examples illustrate the intended output shapes; they are not evidence of current Console monitoring.

### Recent Changes Audit

```json
{
  "audit_type": "recent_changes",
  "timeframe_hours": 24,
  "total_operations": 15,
  "changes": [
    {
      "timestamp": "2025-01-15T10:30:00Z",
      "operation": "Edit", 
      "file_path": "/path/to/handler.py",
      "deception_risk": "HIGH",
      "analysis_result": "RISK LEVEL: HIGH - File contains TODO comments suggesting incomplete implementation",
      "user_prompt": "Implement error handling for API calls"
    }
  ]
}
```

### Deception Report

```json
{
  "audit_type": "deception_report",
  "summary": {
    "total_operations": 25,
    "deception_risk_distribution": {"LOW": 20, "MEDIUM": 3, "HIGH": 2},
    "analyzed_operations": 15,
    "high_risk_operations": [
      {
        "timestamp": "2025-01-15T10:30:00Z",
        "file": "/path/to/file.py",
        "deception_risk": "HIGH",
        "analysis": "Function returns hardcoded values instead of implementing requested functionality"
      }
    ]
  },
  "recommendations": [
    "WARNING: 2 high-risk deception indicators found. Manual review recommended.",
    "High-risk operations detected. Check for TODO/FIXME comments and incomplete implementations."
  ]
}
```

## Historical Design Considerations

The following guidance belonged to the original design. It is not current user or administrator procedure and would require revalidation only if TASK-743 retains or redesigns the subsystem.

### Historical User Guidance

1. **Clear Prompts**: Detailed requests were expected to improve analysis accuracy
2. **Regular Audits**: The design recommended audits after significant changes
3. **Review High-Risk**: HIGH/CRITICAL results required manual review
4. **Context Matters**: Accurate analysis depended on captured user prompts

### Developer Considerations If Re-integrated

1. **Hook Lifecycle**: An integrated design would need hooks in place before relevant file operations
2. **Capture Context**: An integrated design would need prompt context before operations
3. **Handle Failures**: Audit recording would need to avoid breaking file operations
4. **Performance**: LLM analysis would add latency to file operations

### Administrator Considerations If Re-integrated

1. **Monitoring Cadence**: An integrated deployment would need an audit-log review policy
2. **Threshold Tuning**: Risk thresholds would need environment-specific validation
3. **Model Selection**: Model choice would need to match performance requirements
4. **Storage Management**: Retained audit records would consume memory or storage

## Historical Troubleshooting Reference

These notes apply only to an explicit integration of the reference code; they are not steps for enabling current Console auditing.

### Common Issues

**Q: No operations are being recorded**
A: This is expected in the current Console runtime: `code_audit_enabled = true` does not install hooks. TASK-743 owns any future integration decision.

**Q: Analysis says "No user prompt available"**
A: In an explicit integration, prompt context would need to be set before file operations.

**Q: LLM analysis fails**
A: Historical troubleshooting included verifying API keys, model availability, and network connectivity.

**Q: High memory usage**
A: The retained code hardcodes an in-memory limit of 10,000 records and exposes no supported runtime configuration for it. Manual clearing exists only in the reference API shown below, not as a supported Console operation. If TASK-743 retains the subsystem, it must define a bounded owner, lifecycle, and configuration contract.

### Historical Debug Commands

The first snippet depends on the retired System A executor and is retained only as historical reference. The later snippets inspect the importable audit module directly.

```python
# Check if audit tool is registered
from tldw_chatbook.Tools.tool_executor import get_tool_executor
executor = get_tool_executor()
print(executor.get_available_tools())

# Get audit system stats
from tldw_chatbook.Tools.code_audit_tool import get_audit_system
audit_system = get_audit_system()
print(f"Records: {len(audit_system.audit_records)}")

# Clear audit records
audit_system.audit_records.clear()
```

## Reference Security Considerations

If the reference design were integrated:

1. **Audit Records**: Records could contain file content and sensitive data
2. **LLM Analysis**: Configured analysis would send file content to an external LLM
3. **Performance Impact**: Analysis would add latency to file operations
4. **Storage**: Audit records would persist in memory without encryption by default

## Historical Ideas (No Commitment)

These ideas were recorded for the original design and do not imply any TASK-743 outcome:

- Persistent audit storage with encryption
- Real-time alerting for critical operations
- Integration with version control systems
- Advanced pattern detection beyond LLM analysis
- Performance optimizations for large file operations

## API Reference

See `code_audit_tool.py` and `file_operation_hooks.py` for retained implementation reference. Their presence does not wire them into the Console runtime.

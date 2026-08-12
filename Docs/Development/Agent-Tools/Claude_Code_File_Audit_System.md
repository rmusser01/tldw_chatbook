# Claude Code File Audit System

> **Current state:** The implementation/reference code remains importable, but this subsystem is not wired into the Console runtime. It does not monitor the Console builtin `write_file` tool or local `fs_write`, `fs_edit`, and `fs_patch` tools, and must not be relied on as an enforcement or security control. TASK-743 owns the keep/redesign/delete decision for the whole subsystem. The remainder of this guide is historical/reference material for that decision.

## Overview

The Claude Code File Audit System was designed as a monitoring and analysis tool to detect deceptive file operations and determine whether changes align with user requests. The sections below describe that design and its reference implementation.

## Key Features

- **Deception Detection**: Analyzes whether file changes align with user prompts
- **TODO/FIXME Detection**: Identifies incomplete implementations disguised as complete
- **Operation Monitoring Design**: Records hooked file operations (Read, Write, Edit, MultiEdit)
- **LLM-based Analysis**: Uses Claude Haiku for fast, intelligent change analysis
- **Audit Trail**: Maintains detailed records of all file operations
- **Historical Task Tool Integration**: Was exposed through the retired System A framework

## Architecture

### Core Components

1. **FileAuditSystem** (`code_audit_tool.py`): Core audit engine
2. **CodeAuditTool** (`code_audit_tool.py`): Task tool for running audits
3. **FileOperationMonitor** (`file_operation_hooks.py`): Integration hooks
4. **Configuration**: Settings in `config.toml`

### Data Flow

```
User Request → Set Prompt Context → File Operation → Record Operation → LLM Analysis → Audit Record
```

## Historical/Reference Usage

These examples document the former System A interface. `CodeAuditTool` is not registered in the current Console runtime, so they are not current Console commands.

### Basic Audit Commands

The audit system is accessed through the Task tool with `subagent_type="code-audit"`:

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

The reference implementation determines analysis type based on your prompt:

- **Recent Changes**: Keywords "recent", "change" → Last N operations summary
- **Deception Report**: Keywords "report", "deception report" → Comprehensive analysis
- **High Risk Analysis**: Keywords "high risk", "critical" → Focus on risky operations  
- **Incomplete Analysis**: Keywords "todo", "incomplete", "fixme" → TODO/placeholder detection
- **Comprehensive**: Default → All analysis types combined

## What Gets Analyzed

### File Operations Monitored

- **Read**: File access operations
- **Write**: New file creation and full file replacements
- **Edit**: Single find-and-replace operations
- **MultiEdit**: Multiple simultaneous edits
- **Delete**: File deletion operations (when hooked)

### Analysis Criteria

For each file operation, the system analyzes:

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

The importable reference API can be called explicitly as shown below. No current Console file-mutation seam calls it.

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

The historical implementation used the following configuration. It is retained for reference; these settings do not wire the subsystem into the Console runtime:

```toml
[tools]
# Enable the audit tool
code_audit_enabled = true

# Audit system settings
[audit]
# Maximum audit records to keep in memory
max_records = 10000

# LLM settings for analysis
analysis_model = "claude-3-haiku"
analysis_temperature = 0.1
analysis_max_tokens = 500
analysis_timeout = 30

# Enable specific analysis types
enable_deception_detection = true
enable_todo_detection = true
enable_alignment_analysis = true
```

## Output Examples

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

## Best Practices

### For Users

1. **Provide Clear Prompts**: Detailed requests improve analysis accuracy
2. **Regular Audits**: Run comprehensive audits after significant changes
3. **Review High-Risk**: Always manually review HIGH/CRITICAL flagged operations
4. **Context Matters**: Ensure user prompts are captured for accurate analysis

### For Developers (If Re-integrated)

1. **Hook Early**: Install hooks before file operations begin
2. **Capture Context**: Always set user prompt context before operations
3. **Handle Failures**: Audit recording should not break file operations
4. **Monitor Performance**: LLM analysis adds latency to file operations

### For System Administrators (If Re-integrated)

1. **Regular Monitoring**: Check audit logs for patterns
2. **Threshold Tuning**: Adjust risk thresholds based on your environment
3. **Model Selection**: Use appropriate LLM models for your performance needs
4. **Storage Management**: Audit records consume memory/storage

## Troubleshooting

### Common Issues

**Q: No operations are being recorded**
A: This is expected in the current Console runtime: `code_audit_enabled = true` does not install hooks. TASK-743 owns any future integration decision.

**Q: Analysis says "No user prompt available"**
A: Ensure `set_user_prompt()` is called before file operations

**Q: LLM analysis fails**
A: Check API keys, model availability, and network connectivity

**Q: High memory usage**
A: Reduce `max_records` setting or clear audit records more frequently

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

## Security Considerations

1. **Audit Records**: Contain file content and may include sensitive data
2. **LLM Analysis**: File content is sent to external LLM for analysis
3. **Performance Impact**: Analysis adds latency to file operations
4. **Storage**: Audit records persist in memory (not encrypted by default)

## Future Enhancements

- Persistent audit storage with encryption
- Real-time alerting for critical operations
- Integration with version control systems
- Advanced pattern detection beyond LLM analysis
- Performance optimizations for large file operations

## API Reference

See `code_audit_tool.py` and `file_operation_hooks.py` for detailed API documentation.

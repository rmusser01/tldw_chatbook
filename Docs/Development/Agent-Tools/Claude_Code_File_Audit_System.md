# Claude Code File Audit System

## Overview

The Claude Code File Audit System is a comprehensive monitoring and analysis tool designed to detect deceptive file operations and ensure that Claude Code's file modifications align with user requests. Unlike traditional security pattern matching, this system uses LLM analysis to determine whether changes actually implement what the user requested or contain deceptive modifications.

## Key Features

- **Deception Detection**: Analyzes whether file changes align with user prompts
- **TODO/FIXME Detection**: Identifies incomplete implementations disguised as complete
- **Real-time Monitoring**: Tracks all file operations (Read, Write, Edit, MultiEdit)
- **LLM-based Analysis**: Uses Claude Haiku for fast, intelligent change analysis
- **Audit Trail**: Maintains detailed records of all file operations
- **Task Tool Integration**: Accessible via the existing Task tool framework

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

## Usage

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

The system automatically determines analysis type based on your prompt:

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

### Automatic Integration

The system automatically hooks into available file operation tools when enabled. Add to your `config.toml`:

```toml
[tools]
code_audit_enabled = true
```

### Manual Integration

For deeper integration, add these calls to your file operation workflows:

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

## Configuration

Add audit system configuration to `config.toml`:

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

### For Developers

1. **Hook Early**: Install hooks before file operations begin
2. **Capture Context**: Always set user prompt context before operations
3. **Handle Failures**: Audit recording should not break file operations
4. **Monitor Performance**: LLM analysis adds latency to file operations

### For System Administrators

1. **Regular Monitoring**: Check audit logs for patterns
2. **Threshold Tuning**: Adjust risk thresholds based on your environment
3. **Model Selection**: Use appropriate LLM models for your performance needs
4. **Storage Management**: Audit records consume memory/storage

## Troubleshooting

### Common Issues

**Q: No operations are being recorded**
A: Check that `code_audit_enabled = true` in config and hooks are installed correctly

**Q: Analysis says "No user prompt available"**
A: Ensure `set_user_prompt()` is called before file operations

**Q: LLM analysis fails**
A: Check API keys, model availability, and network connectivity

**Q: High memory usage**
A: Reduce `max_records` setting or clear audit records more frequently

### Debug Commands

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
# Claude Code Hook Setup Guide

This guide explains how to set up Monocle tracing for Claude Code CLI sessions.

## Overview

The Claude Code hook automatically captures trace events from your Claude Code sessions and exports them to your configured observability backend (Okahu, console, file, etc.).

## Installation

### 1. Install the Monocle Package

```bash
cd monocle/apptrace
pip install -e .
```

### 2. Configure Claude Code Settings

Copy the provided settings template to Claude's global configuration directory:

```bash
cp claude_hook_settings.json ~/.claude/settings.json
```

**What's in `claude_hook_settings.json`:**
```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "python3 -m monocle_apptrace claude-hook"
      }]
    }]
  }
}
```

This tells Claude Code to run `python3 -m monocle_apptrace claude-hook` whenever a session ends (Stop event).

### 3. Set Environment Variables

Add these to your `~/.zshrc` or `~/.bashrc`:

```bash
# Monocle Claude Hook Configuration
export MONOCLE_EXPORTER="okahu,file"              # Where to send traces
export OKAHU_API_KEY="your-api-key"               # Your Okahu API key
export OKAHU_INGESTION_ENDPOINT="https://ingest.okahu.co/api/v1/trace/ingest"
export MONOCLE_SERVICE_NAME="claude-cli"          # Service name in traces
export DEFAULT_WORKFLOW_NAME="claude-cli"         # Workflow name
export MONOCLE_CLAUDE_DEBUG=true                  # Optional: debug logging
```

**Available Exporters:**
- `okahu` - Send to Okahu observability platform
- `file` - Write to local JSON files
- `console` - Print to terminal (good for testing)
- Combine multiple: `"okahu,file,console"`

### 4. Reload Your Shell

```bash
source ~/.zshrc  # or ~/.bashrc
```

## Testing

### Test with Mock Event

```bash
cd monocle/apptrace
./test_claude_hook.sh
```

### Test with Real Claude Code

1. Start Claude Code in any directory
2. Have a conversation
3. Exit or complete the session
4. Check traces:

**Session logs:**
```bash
ls -la .monocle/.claude_sessions/
cat .monocle/.claude_sessions/.monocle_claude_*.jsonl
```

**File exports (if using file exporter):**
Check your configured output directory for trace files.

**Okahu (if using okahu exporter):**
View traces in your Okahu dashboard.

## How It Works

```
┌─────────────────────┐
│ Claude Code CLI     │
│ Session Ends        │
└──────────┬──────────┘
           │ Triggers Stop hook
           ▼
┌─────────────────────────────────────┐
│ python3 -m monocle_apptrace         │
│         claude-hook                 │
└──────────┬──────────────────────────┘
           │ Reads event JSON from stdin
           ▼
┌─────────────────────────────────────┐
│ Event Handler                       │
│ - Records event                     │
│ - Appends to session log            │
│ - Triggers replay                   │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ Replay Session                      │
│ - Groups events into spans          │
│ - Emits OpenTelemetry traces        │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ Exporters                           │
│ - Okahu                             │
│ - File                              │
│ - Console                           │
└─────────────────────────────────────┘
```

## Troubleshooting

### Hook doesn't run

**Check settings file:**
```bash
cat ~/.claude/settings.json
```

**Test command manually:**
```bash
echo '{"session_id":"test","hook_event_name":"Stop"}' | python3 -m monocle_apptrace claude-hook
```

### No traces appearing

**Verify environment variables:**
```bash
env | grep MONOCLE
env | grep OKAHU
```

**Test with console exporter:**
```bash
export MONOCLE_EXPORTER="console"
```
Start a Claude session and watch terminal output.

### Import errors

**Check package installation:**
```bash
python3 -c "import monocle_apptrace; print(monocle_apptrace.__file__)"
```

**Reinstall if needed:**
```bash
cd monocle/apptrace
pip install -e . --force-reinstall
```

## Files Reference

| File | Purpose | Location |
|------|---------|----------|
| `claude_hook_settings.json` | Template for Claude settings | Repo (for distribution) |
| `~/.claude/settings.json` | Active Claude configuration | User's home directory |
| `~/.zshrc` or `~/.bashrc` | Environment variables | User's home directory |
| `.monocle/.claude_sessions/` | Session event logs | Where Claude Code runs |

## Advanced Configuration

### Custom Exporters

Create your own exporter and configure:
```bash
export MONOCLE_EXPORTER="custom_module.CustomExporter"
```

### Multiple Projects

Use different environment variables per project with `.env` files:
```bash
# In project directory
cat > .env << EOF
export MONOCLE_SERVICE_NAME="my-project"
export DEFAULT_WORKFLOW_NAME="my-workflow"
EOF

# Source before running Claude Code
source .env
```

### Different Python Version

If you have multiple Python versions:
```bash
# In ~/.claude/settings.json, use specific python
"command": "python3.11 -m monocle_apptrace claude-hook"
```

## Support

- **Documentation**: See [Monocle User Guide](Monocle_User_Guide.md)
- **Issues**: Report at the Monocle repository
- **Implementation Guide**: See [PRASAD_IMPLEMENTATION_GUIDE.md](../PRASAD_IMPLEMENTATION_GUIDE.md)

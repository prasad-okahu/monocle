"""
Monocle hooks configuration for Claude Code CLI.

Add the following to ~/.claude/settings.json to enable Monocle telemetry:

{
  "hooks": {
    "<event>": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python -m monocle_apptrace.instrumentation.metamodel.claude_cli",
            "statusMessage": "Monocle telemetry"
          }
        ]
      }
    ]
  }
}

where <event> is each of the hook event names listed in HOOK_EVENTS below.
"""

HOOK_EVENTS = [
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "SubagentStart",
    "SubagentStop",
    "Stop",
    "StopFailure",
    "PreCompact",
    "PostCompact",
    "SessionEnd",
]

HOOK_ENTRY = {
    "type": "command",
    "command": "python -m monocle_apptrace.instrumentation.metamodel.claude_cli",
    "statusMessage": "Monocle telemetry",
}

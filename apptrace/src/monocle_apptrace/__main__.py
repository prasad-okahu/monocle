import sys, os
import runpy
from monocle_apptrace import setup_monocle_telemetry


def main():
    # Check if running in Claude hook mode
    if len(sys.argv) >= 2 and sys.argv[1] in ("claude-hook", "claude_hook", "--claude-hook"):
        # Run the Claude CLI event handler
        from monocle_apptrace.instrumentation.metamodel.claude_cli.event_handler import main as hook_main
        sys.exit(hook_main())
    
    # Original behavior: wrap user scripts
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        print("Usage:")
        print("  python -m monocle_apptrace <your-main-module-file> <args>")
        print("  python -m monocle_apptrace claude-hook              (run Claude Code hook)")
        sys.exit(1)
    
    file_name = os.path.basename(sys.argv[1])
    workflow_name = file_name[:-3]
    setup_monocle_telemetry(workflow_name=workflow_name)
    sys.argv.pop(0)

    try:
        runpy.run_path(path_name=sys.argv[0], run_name="__main__")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()

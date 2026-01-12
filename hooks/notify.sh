#!/bin/bash
# Claude Code Hook Script
# This script receives hook events from Claude Code and forwards them to the remotecontrol server.
#
# Hook input is received via stdin as JSON containing:
#   - session_id: The current session ID
#   - hook_event_name: The event type (PreToolUse, PostToolUse, Stop, etc.)
#   - tool_name: The tool being used (for PreToolUse/PostToolUse)
#   - tool_input: The tool's input parameters
#   - tool_result: The tool's output (for PostToolUse)
#   - And other event-specific fields
#
# Usage: Configure in ~/.claude/settings.json or project .claude/settings.json

set -euo pipefail

# Configuration - adjust these as needed
REMOTECONTROL_HOST="${REMOTECONTROL_HOST:-localhost}"
REMOTECONTROL_PORT="${REMOTECONTROL_PORT:-8765}"
REMOTECONTROL_URL="http://${REMOTECONTROL_HOST}:${REMOTECONTROL_PORT}/api/hook"

# Capture TTY for session identification (e.g., "ttys001")
# Method 1: Try `tty` command (works in interactive shells)
TTY_NAME=$(tty 2>/dev/null | sed 's|/dev/||' || echo "")

# Method 2: If tty failed, get TTY from parent process (Claude Code) via ps
if [[ -z "$TTY_NAME" || "$TTY_NAME" == "not a tty" ]]; then
    TTY_NAME=$(ps -p $PPID -o tty= 2>/dev/null | tr -d ' ' || echo "")
fi

# Method 3: Walk up process tree to find a process with a TTY
if [[ -z "$TTY_NAME" || "$TTY_NAME" == "??" ]]; then
    WALK_PID=$PPID
    for _ in 1 2 3 4 5; do
        TTY_NAME=$(ps -p $WALK_PID -o tty= 2>/dev/null | tr -d ' ' || echo "")
        if [[ -n "$TTY_NAME" && "$TTY_NAME" != "??" ]]; then
            break
        fi
        WALK_PID=$(ps -p $WALK_PID -o ppid= 2>/dev/null | tr -d ' ' || echo "")
        [[ -z "$WALK_PID" ]] && break
    done
fi

# Fallback to unknown
[[ -z "$TTY_NAME" || "$TTY_NAME" == "??" ]] && TTY_NAME="unknown"

# Read JSON from stdin
input=$(cat)

# Inject TTY into the JSON payload for session tracking
# Use jq if available, otherwise send as-is
if command -v jq &>/dev/null; then
    input=$(echo "$input" | jq --arg tty "$TTY_NAME" '. + {tty: $tty}')
fi

# Forward to remotecontrol server
# Use timeout to prevent hanging, and ignore errors (hook shouldn't block Claude)
curl -s -X POST "$REMOTECONTROL_URL" \
    -H "Content-Type: application/json" \
    -d "$input" \
    --connect-timeout 2 \
    --max-time 5 \
    2>/dev/null || true

# Always exit 0 so we don't block Claude Code
exit 0

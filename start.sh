#!/bin/bash
# Claude Code Remote Control - Startup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration defaults
SESSION_NAME="claude"
SERVER_PORT=8765
BACKEND=""  # auto-detect by default

# Help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --name NAME      Set session name (default: claude)"
    echo "  -p, --port PORT      Set server port (default: 8765)"
    echo "  -b, --backend MODE   Force backend: ghostty, tmux, or auto (default: auto)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                       # Auto-detect backend"
    echo "  $0 -b ghostty            # Force Ghostty mode (no tmux)"
    echo "  $0 -b tmux -n myproject  # Force tmux with session 'myproject'"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            SESSION_NAME="$2"
            shift 2
            ;;
        -p|--port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -b|--backend)
            BACKEND="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Get local IP
get_local_ip() {
    ipconfig getifaddr en0 2>/dev/null || echo "localhost"
}

LOCAL_IP=$(get_local_ip)

# Detect if Ghostty is running
is_ghostty_available() {
    pgrep -f "Ghostty.app" >/dev/null 2>&1
}

# Detect backend
detect_backend() {
    if [ -n "$BACKEND" ]; then
        echo "$BACKEND"
        return
    fi

    if is_ghostty_available; then
        echo "ghostty"
    else
        echo "tmux"
    fi
}

DETECTED_BACKEND=$(detect_backend)

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║       Claude Code Remote Control - Startup                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  ${YELLOW}Backend:${NC} $DETECTED_BACKEND"
echo ""

# Check dependencies
check_deps() {
    echo -e "${YELLOW}Checking dependencies...${NC}"

    local missing=()

    # tmux only required for tmux backend
    if [ "$DETECTED_BACKEND" = "tmux" ]; then
        command -v tmux >/dev/null 2>&1 || missing+=("tmux")
    fi
    command -v python3 >/dev/null 2>&1 || missing+=("python3")
    command -v claude >/dev/null 2>&1 || missing+=("claude (Claude Code CLI)")

    if [ ${#missing[@]} -ne 0 ]; then
        echo -e "${RED}Missing dependencies: ${missing[*]}${NC}"
        exit 1
    fi

    echo -e "${GREEN}Dependencies OK${NC}"
}

# Setup Python environment
setup_python() {
    echo -e "${YELLOW}Setting up Python environment...${NC}"

    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi

    source venv/bin/activate
    pip install -q flask flask-sock pyyaml requests

    echo -e "${GREEN}Python environment ready${NC}"
}

# Create/attach tmux session (only for tmux backend)
setup_tmux() {
    if [ "$DETECTED_BACKEND" != "tmux" ]; then
        return
    fi

    echo -e "${YELLOW}Setting up tmux session...${NC}"

    # Check if session exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${GREEN}tmux session '$SESSION_NAME' already exists${NC}"
    else
        echo -e "${YELLOW}Creating new tmux session '$SESSION_NAME'...${NC}"
        tmux new-session -d -s "$SESSION_NAME"
        echo -e "${GREEN}tmux session created${NC}"
    fi
}

# Start the control server
start_server() {
    echo -e "${YELLOW}Starting control server...${NC}"

    source venv/bin/activate

    if [ "$DETECTED_BACKEND" = "tmux" ]; then
        TMUX_SESSION="$SESSION_NAME" python3 server.py &
    else
        python3 server.py &
    fi
    SERVER_PID=$!

    sleep 2

    if kill -0 $SERVER_PID 2>/dev/null; then
        echo -e "${GREEN}Control server started (PID: $SERVER_PID)${NC}"
    else
        echo -e "${RED}Failed to start control server${NC}"
        exit 1
    fi
}

# Print access info
print_info() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Claude Code Remote Control is ready!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${BLUE}Control Panel:${NC}  http://$LOCAL_IP:$SERVER_PORT/"
    echo -e "  ${BLUE}Full Terminal:${NC}  http://$LOCAL_IP:$SERVER_PORT/terminal"
    echo ""
    echo -e "  ${YELLOW}Default credentials:${NC}"
    echo -e "    Username: claude"
    echo -e "    Password: changeme  ${RED}(change in config.yaml!)${NC}"
    echo ""

    echo -e "  ${YELLOW}To use Claude Code with remote control:${NC}"
    echo -e "    1. Configure hooks in ~/.claude/settings.json (see README)"
    echo -e "    2. Run ${GREEN}claude${NC} in any terminal"
    echo -e "    3. Sessions will appear automatically via hooks"
    echo ""
    echo -e "  ${YELLOW}For iOS notifications (ntfy):${NC}"
    echo -e "    1. Install 'ntfy' app from App Store"
    echo -e "    2. Subscribe to your topic (see config.yaml)"
    echo ""
    echo -e "  ${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
}

# Cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"

    # Kill server
    pkill -f "python3 server.py" 2>/dev/null || true

    echo -e "${GREEN}Stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main
main() {
    check_deps
    setup_python
    setup_tmux
    start_server
    print_info

    # Wait for Ctrl+C
    while true; do
        sleep 1
    done
}

main "$@"

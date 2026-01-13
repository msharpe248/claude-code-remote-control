#!/bin/bash
# Claude Code Remote Control - Installation Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   Claude Code Remote Control - Installation               ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for Homebrew
if ! command -v brew >/dev/null 2>&1; then
    echo -e "${RED}Homebrew not found. Please install it first:${NC}"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"

brew install tmux 2>/dev/null || true

echo -e "${GREEN}System dependencies installed${NC}"

# Setup Python virtual environment
echo -e "${YELLOW}Setting up Python environment...${NC}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install flask flask-sock pyyaml requests

echo -e "${GREEN}Python environment ready${NC}"

# Make scripts executable
chmod +x start.sh
chmod +x install.sh
chmod +x uninstall.sh
chmod +x hooks/notify.sh
chmod +x cc-status cc-logs cc-diagnose cc-test-notify

# Generate a random password
RANDOM_PASS=$(openssl rand -base64 12 | tr -dc 'a-zA-Z0-9' | head -c12)
RANDOM_TOKEN=$(openssl rand -hex 16)

# Update config with random credentials
if [ -f config.yaml ]; then
    # Only update if still using defaults
    if grep -q "changeme" config.yaml; then
        echo -e "${YELLOW}Generating secure credentials...${NC}"

        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/changeme/$RANDOM_PASS/" config.yaml
            sed -i '' "s/your-secret-token-here/$RANDOM_TOKEN/" config.yaml
        else
            sed -i "s/changeme/$RANDOM_PASS/" config.yaml
            sed -i "s/your-secret-token-here/$RANDOM_TOKEN/" config.yaml
        fi

        echo -e "${GREEN}Generated new credentials:${NC}"
        echo -e "  Password: ${YELLOW}$RANDOM_PASS${NC}"
        echo -e "  Token:    ${YELLOW}$RANDOM_TOKEN${NC}"
        echo ""
        echo -e "${RED}Save these! They won't be shown again.${NC}"
    fi
fi

# Installation complete
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}Next steps:${NC}"
echo ""
echo -e "  1. Configure notifications in config.yaml"
echo -e "     - Set topic_prefix to something unique (e.g., claude-yourname)"
echo -e "     - For Pushover: add your API keys"
echo ""
echo -e "  2. Install ntfy app on your iPhone:"
echo -e "     https://apps.apple.com/app/ntfy/id1625396347"
echo -e "     Subscribe to: {topic_prefix}-claude (e.g., claude-yourname-claude)"
echo ""
echo -e "  3. Start the server:"
echo -e "     ${BLUE}./start.sh${NC}"
echo ""
echo -e "  4. Configure Claude Code hooks:"
echo -e "     Add hooks/notify.sh to ~/.claude/settings.json (see README)"
echo -e "     Then run ${BLUE}claude${NC} in any terminal"
echo ""

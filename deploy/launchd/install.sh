#!/usr/bin/env bash
# Install/uninstall the iMessage bot as a launchd agent (starts at login, restarts on crash).
#   ./deploy/launchd/install.sh            install + start
#   ./deploy/launchd/install.sh uninstall  stop + remove
set -euo pipefail

LABEL=com.charliegpt.imessage
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ "${1:-}" == "uninstall" ]]; then
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$DEST"
    echo "removed $LABEL"
    exit 0
fi

UV="$(command -v uv)" || { echo "uv not found (brew install uv)"; exit 1; }
mkdir -p "$HOME/Library/LaunchAgents"
sed -e "s|__UV__|$UV|g" -e "s|__REPO__|$REPO|g" -e "s|__HOME__|$HOME|g" \
    "$REPO/deploy/launchd/$LABEL.plist" > "$DEST"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$DEST"
echo "installed $LABEL → logs: ~/Library/Logs/charliegpt-imessage.log"
echo "Grant Full Disk Access to: $(cd "$REPO" && uv run python -c 'import sys,os; print(os.path.realpath(sys.executable))')"

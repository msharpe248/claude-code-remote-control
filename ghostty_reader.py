#!/usr/bin/env python3
"""
Ghostty Terminal Reader
Uses macOS Accessibility API to read Ghostty terminal content.

Requirements:
- pip install pyobjc-framework-ApplicationServices pyobjc-framework-Quartz
- Enable accessibility permissions in System Preferences > Privacy & Security > Accessibility
"""

import sys
import re
from dataclasses import dataclass
from typing import Optional, List

try:
    from ApplicationServices import (
        AXUIElementCopyAttributeValue,
        AXUIElementCopyAttributeNames,
        AXUIElementCreateApplication,
        kAXErrorSuccess,
    )
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
    )
except ImportError:
    print("Missing pyobjc packages. Install with:")
    print("  pip install pyobjc-framework-ApplicationServices pyobjc-framework-Quartz")
    sys.exit(1)


@dataclass
class GhosttyTab:
    """Represents a Ghostty tab."""
    title: str
    index: int
    is_selected: bool


@dataclass
class GhosttyTerminal:
    """Represents a terminal pane in Ghostty."""
    content: str
    tab_title: Optional[str] = None
    pane_position: Optional[str] = None


def get_running_apps():
    """Get list of running applications with windows."""
    windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    apps = {}
    for window in windows:
        owner = window.get('kCGWindowOwnerName', '')
        pid = window.get('kCGWindowOwnerPID', 0)
        if owner and pid:
            apps[owner] = pid
    return apps


def get_ax_attribute(element, attribute):
    """Get an accessibility attribute value."""
    err, value = AXUIElementCopyAttributeValue(element, attribute, None)
    if err == kAXErrorSuccess:
        return value
    return None


def get_ax_attributes(element):
    """Get list of available attributes on an element."""
    err, attrs = AXUIElementCopyAttributeNames(element, None)
    if err == kAXErrorSuccess:
        return list(attrs) if attrs else []
    return []


def find_ghostty_pid() -> Optional[int]:
    """Find Ghostty's process ID."""
    apps = get_running_apps()
    return apps.get("Ghostty")


def get_ghostty_app(pid: int):
    """Create accessibility element for Ghostty application."""
    app = AXUIElementCreateApplication(pid)
    # Verify we can access it
    attrs = get_ax_attributes(app)
    if not attrs:
        return None
    return app


def get_tabs(app) -> List[GhosttyTab]:
    """Get list of tabs in Ghostty."""
    tabs = []

    def find_tabs(element, depth=0):
        if depth > 5:
            return

        role = get_ax_attribute(element, "AXRole")

        # Find tab group
        if role == "AXTabGroup":
            tab_elements = get_ax_attribute(element, "AXTabs")
            if tab_elements:
                selected = get_ax_attribute(element, "AXValue")
                for i, tab_elem in enumerate(tab_elements):
                    title = get_ax_attribute(tab_elem, "AXTitle") or f"Tab {i+1}"
                    is_selected = (tab_elem == selected) if selected else (i == 0)
                    tabs.append(GhosttyTab(title=title, index=i, is_selected=is_selected))
            return  # Found tabs, no need to go deeper

        # Recurse into children
        children = get_ax_attribute(element, "AXChildren")
        if children:
            for child in children:
                find_tabs(child, depth + 1)

    find_tabs(app)
    return tabs


def get_terminal_contents(app) -> List[GhosttyTerminal]:
    """Get all terminal text areas from Ghostty."""
    terminals = []

    def find_terminals(element, depth=0, pane_desc=None):
        if depth > 10:
            return

        role = get_ax_attribute(element, "AXRole")
        desc = get_ax_attribute(element, "AXDescription")

        # Track pane position from split descriptions
        if desc and ("pane" in desc.lower() or "split" in desc.lower()):
            pane_desc = desc

        # Find text areas (terminal content)
        if role == "AXTextArea":
            value = get_ax_attribute(element, "AXValue")
            if value and len(str(value)) > 10:
                terminals.append(GhosttyTerminal(
                    content=str(value),
                    pane_position=pane_desc
                ))
            return  # Don't recurse into text areas

        # Recurse into children
        children = get_ax_attribute(element, "AXChildren")
        if children:
            for child in children:
                find_terminals(child, depth + 1, pane_desc)

    find_terminals(app)
    return terminals


def detect_claude_prompt(content: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Detect if Claude Code is waiting for input.
    Returns: (is_prompt, context, prompt_type)
    """
    lines = content.strip().split('\n')
    if len(lines) < 3:
        return False, None, None

    # Look at recent lines
    recent_lines = lines[-15:]
    recent_text = '\n'.join(recent_lines)

    # Check if busy (spinner active)
    for line in recent_lines[-3:]:
        line_lower = line.lower()
        if "esc to interrupt" in line_lower or "to interrupt" in line_lower:
            return False, None, None

    # Check for "Esc to cancel" - covers all question types
    if "Esc to cancel" in recent_text:
        context = '\n'.join(recent_lines[-8:])
        return True, context, 'question'

    # Check for idle prompt ("send" + "? for shortcuts")
    if "send" in recent_text and "? for shortcuts" in recent_text:
        context = '\n'.join(recent_lines[-8:])
        return True, context, 'idle'

    return False, None, None


def find_claude_sessions():
    """Find all Claude Code sessions in Ghostty terminals."""
    pid = find_ghostty_pid()
    if not pid:
        return None, "Ghostty not running"

    app = get_ghostty_app(pid)
    if not app:
        return None, "Cannot access Ghostty accessibility. Enable in System Preferences > Privacy & Security > Accessibility"

    tabs = get_tabs(app)
    terminals = get_terminal_contents(app)

    sessions = []
    for i, terminal in enumerate(terminals):
        # Check if this terminal has Claude running
        content = terminal.content
        is_prompt, context, prompt_type = detect_claude_prompt(content)

        # Look for Claude indicators
        is_claude = (
            "claude" in content.lower()[-2000:] or
            "anthropic" in content.lower()[-2000:] or
            "? for shortcuts" in content[-1000:] or
            "Esc to cancel" in content[-1000:]
        )

        if is_claude:
            sessions.append({
                'index': i,
                'pane': terminal.pane_position,
                'has_prompt': is_prompt,
                'prompt_type': prompt_type,
                'context': context,
                'content_preview': content[-500:] if len(content) > 500 else content
            })

    return {
        'tabs': [{'title': t.title, 'index': t.index, 'selected': t.is_selected} for t in tabs],
        'terminals': len(terminals),
        'claude_sessions': sessions
    }, None


def main():
    print("=" * 60)
    print("Ghostty Terminal Reader")
    print("=" * 60)

    result, error = find_claude_sessions()

    if error:
        print(f"\nError: {error}")
        return

    print(f"\nGhostty Status:")
    print(f"  Tabs: {len(result['tabs'])}")
    for tab in result['tabs']:
        selected = " (selected)" if tab['selected'] else ""
        print(f"    [{tab['index']}] {tab['title']}{selected}")

    print(f"  Terminal panes: {result['terminals']}")

    print(f"\nClaude Sessions Found: {len(result['claude_sessions'])}")
    for session in result['claude_sessions']:
        print(f"\n  Session {session['index']}:")
        if session['pane']:
            print(f"    Pane: {session['pane']}")
        print(f"    Has prompt: {session['has_prompt']}")
        if session['has_prompt']:
            print(f"    Prompt type: {session['prompt_type']}")
            print(f"    Context:")
            for line in (session['context'] or '').split('\n'):
                print(f"      {line}")


if __name__ == "__main__":
    main()

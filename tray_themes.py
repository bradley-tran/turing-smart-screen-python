#!/usr/bin/env python3
# tray_themes.py — Lightweight tray icon for theme switching.
# Run as your regular user (no sudo). It updates config.yaml and
# sends SIGUSR1 to the main.py process to trigger an auto-restart.
#
# Usage:  python3 tray_themes.py   (or .venv/bin/python tray_themes.py)

import os
import re
import signal
import sys
from pathlib import Path

import pystray
import yaml
from PIL import Image

MAIN_DIRECTORY = str(Path(__file__).parent.resolve()) + "/"
THEMES_DIR = os.path.join(MAIN_DIRECTORY, "res", "themes")
CONFIG_PATH = os.path.join(MAIN_DIRECTORY, "config.yaml")


def load_config():
    with open(CONFIG_PATH, "rt", encoding="utf8") as f:
        return yaml.safe_load(f)


def get_current_theme():
    return load_config()["config"]["THEME"]


def get_theme_display_info(theme_name):
    """Return (DISPLAY_SIZE, DISPLAY_ORIENTATION) for a theme."""
    theme_yaml = os.path.join(THEMES_DIR, theme_name, "theme.yaml")
    if os.path.isfile(theme_yaml):
        with open(theme_yaml, "rt", encoding="utf8") as f:
            data = yaml.safe_load(f)
        if data:
            d = data.get("display", {})
            return (d.get("DISPLAY_SIZE", '3.5"'), d.get("DISPLAY_ORIENTATION", "portrait"))
    return ('3.5"', "portrait")


def get_compatible_themes():
    """Return sorted list of theme names matching the current theme's size and orientation."""
    current_size, current_orient = get_theme_display_info(get_current_theme())
    result = []
    for name in os.listdir(THEMES_DIR):
        theme_yaml = os.path.join(THEMES_DIR, name, "theme.yaml")
        if os.path.isfile(theme_yaml):
            try:
                with open(theme_yaml, "rt", encoding="utf8") as f:
                    data = yaml.safe_load(f)
                if data:
                    d = data.get("display", {})
                    size = d.get("DISPLAY_SIZE", '3.5"')
                    orient = d.get("DISPLAY_ORIENTATION", "portrait")
                    if size == current_size and orient == current_orient:
                        result.append(name)
            except Exception:
                pass
    return sorted(result, key=str.casefold)


def find_main_pid():
    """Find the PID of the running main.py process."""
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().decode(errors="replace")
            if "main.py" in cmdline and "tray_themes" not in cmdline:
                return int(pid_dir.name)
        except (OSError, ValueError):
            pass
    return None


def switch_theme(theme_name):
    """Update config.yaml and signal main.py to restart."""
    # Read & replace the THEME line (preserves comments)
    with open(CONFIG_PATH, "rt", encoding="utf8") as f:
        content = f.read()
    content = re.sub(r"(?m)^(\s*THEME:\s*)(.+)$", r"\g<1>" + theme_name, content)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Theme set to '{theme_name}'")

    # Notify main.py to restart — prefer pipe fd, fall back to SIGUSR1
    restart_fd = os.environ.get('TRAY_RESTART_FD')
    if restart_fd:
        try:
            os.write(int(restart_fd), f"theme={theme_name}\n".encode())
            print("Sent restart via pipe")
            return
        except OSError as e:
            print(f"Pipe write failed: {e}, trying SIGUSR1")

    pid = find_main_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGUSR1)
            print(f"Sent SIGUSR1 to main.py (PID {pid})")
        except PermissionError:
            print("Cannot signal root process — run tray_themes.py via main.py instead")
    else:
        print("Warning: main.py process not found — restart it manually")


def build_tray():
    current_theme = get_current_theme()
    themes = get_compatible_themes()

    menu_items = []
    for t in themes:
        menu_items.append(
            pystray.MenuItem(
                text=t,
                action=(lambda name: lambda icon, item: (switch_theme(name), icon.stop()))(t),
                checked=(lambda name: lambda item: name == get_current_theme())(t),
            )
        )

    menu_items.append(pystray.Menu.SEPARATOR)
    menu_items.append(
        pystray.MenuItem("Quit Tray", lambda icon, item: icon.stop())
    )

    icon_path = os.path.join(MAIN_DIRECTORY, "res", "icons", "monitor-icon-17865", "64.png")
    if os.path.isfile(icon_path):
        icon_image = Image.open(icon_path)
    else:
        icon_image = Image.new("RGB", (64, 64), (0, 120, 255))

    tray = pystray.Icon(
        name="Turing Theme Switcher",
        title=f"Theme: {current_theme}",
        icon=icon_image,
        menu=pystray.Menu(*menu_items),
    )
    return tray


if __name__ == "__main__":
    print(f"pystray backend: {pystray.Icon.__module__}")
    print(f"Current theme: {get_current_theme()}")
    tray = build_tray()
    print("Tray icon running — click it to switch themes")
    tray.run()

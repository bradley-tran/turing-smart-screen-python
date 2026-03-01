# Racer display volume-knob → brightness control
#
# The Racer USB display has a SmartCloud AKP846 rotary encoder
# (VID 0x28E9, PID 0x3012) that the kernel exposes as a Consumer Control
# HID device.  Turning the knob emits KEY_VOLUMEUP / KEY_VOLUMEDOWN.
#
# This module grabs that device exclusively and re-interprets the
# events as brightness adjustments (5 % per click, range 5–100 %).

import os
import struct
import threading
from pathlib import Path

from library.log import logger

# ── Constants ───────────────────────────────────────────────────────────────

# SmartCloud AKP846 USB IDs
KNOB_VID = 0x28E9
KNOB_PID = 0x3012

# Linux input event ABI
EV_KEY = 0x01
KEY_VOLUMEDOWN = 0x72  # 114
KEY_VOLUMEUP = 0x73    # 115

# struct input_event: time_t, suseconds_t, __u16, __u16, __s32
# on 64-bit Linux: 8 + 8 + 2 + 2 + 4 = 24 bytes
INPUT_EVENT_FORMAT = "llHHi"
INPUT_EVENT_SIZE = struct.calcsize(INPUT_EVENT_FORMAT)

# EVIOCGRAB ioctl number  (_IOW('E', 0x90, int) = 0x40044590)
EVIOCGRAB = 0x40044590

# Brightness parameters
BRIGHTNESS_STEP = 5
BRIGHTNESS_MIN = 5
BRIGHTNESS_MAX = 100
BRIGHTNESS_DEFAULT = 70


# ── Device discovery ────────────────────────────────────────────────────────

def _read_sysfs_hex(path: Path) -> int | None:
    """Read a hex value from a sysfs file, return None on failure."""
    try:
        return int(path.read_text().strip(), 16)
    except (OSError, ValueError):
        return None


def _find_consumer_control_device() -> str | None:
    """Find the SmartCloud AKP846 Consumer Control evdev node.

    The AKP846 exposes multiple input interfaces (keyboard, mouse,
    system control, consumer control).  We match by VID/PID and pick
    the device whose name contains "Consumer Control".
    """
    sysfs = Path("/sys/class/input")
    if not sysfs.is_dir():
        return None

    for entry in sorted(sysfs.iterdir()):
        name = entry.name
        if not name.startswith("event"):
            continue

        id_dir = entry / "device" / "id"
        vid = _read_sysfs_hex(id_dir / "vendor")
        pid = _read_sysfs_hex(id_dir / "product")

        if vid != KNOB_VID or pid != KNOB_PID:
            continue

        # Read the device name to distinguish Consumer Control from Keyboard/Mouse
        try:
            dev_name = (entry / "device" / "name").read_text().strip()
        except OSError:
            continue

        if "Consumer Control" in dev_name:
            return f"/dev/input/{name}"

    return None


# ── Knob brightness controller ─────────────────────────────────────────────

class KnobBrightness:
    """Background thread: reads volume-knob events, maintains brightness level."""

    def __init__(self, initial: int = BRIGHTNESS_DEFAULT):
        self._brightness = max(BRIGHTNESS_MIN, min(BRIGHTNESS_MAX, initial))
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def brightness(self) -> int:
        """Current brightness percentage (5–100)."""
        with self._lock:
            return self._brightness

    def start(self) -> bool:
        """Start the knob-reader thread.  Returns False if device not found."""
        dev_path = _find_consumer_control_device()
        if dev_path is None:
            logger.warning("Knob: SmartCloud AKP846 Consumer Control not found")
            return False

        logger.info(f"Knob: found consumer control device at {dev_path}")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._event_loop,
            args=(dev_path,),
            name="knob-brightness",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self):
        """Signal the thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _event_loop(self, dev_path: str):
        """Read evdev events, updating brightness on volume key presses."""
        import fcntl

        try:
            fd = os.open(dev_path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as e:
            logger.error(f"Knob: cannot open {dev_path}: {e}")
            return

        try:
            # Grab exclusively — prevents volume events from reaching PulseAudio
            try:
                fcntl.ioctl(fd, EVIOCGRAB, 1)
                logger.info("Knob: grabbed device exclusively")
            except OSError as e:
                logger.warning(f"Knob: EVIOCGRAB failed (non-fatal): {e}")

            logger.info(f"Knob: entering event loop (brightness={self._brightness}%)")

            import select

            while not self._stop_event.is_set():
                # Poll with 1-second timeout so we notice stop_event promptly
                readable, _, _ = select.select([fd], [], [], 1.0)
                if not readable:
                    continue

                try:
                    data = os.read(fd, INPUT_EVENT_SIZE * 16)
                except OSError:
                    continue

                # Process all events in the read buffer
                offset = 0
                while offset + INPUT_EVENT_SIZE <= len(data):
                    _sec, _usec, ev_type, ev_code, ev_value = struct.unpack_from(
                        INPUT_EVENT_FORMAT, data, offset
                    )
                    offset += INPUT_EVENT_SIZE

                    # Only act on key-down events (value == 1)
                    if ev_type != EV_KEY or ev_value != 1:
                        continue

                    if ev_code == KEY_VOLUMEUP:
                        self._adjust(BRIGHTNESS_STEP)
                    elif ev_code == KEY_VOLUMEDOWN:
                        self._adjust(-BRIGHTNESS_STEP)

        finally:
            # Release grab and close
            try:
                fcntl.ioctl(fd, EVIOCGRAB, 0)
            except OSError:
                pass
            os.close(fd)
            logger.info("Knob: event loop stopped, device released")

    def _adjust(self, delta: int):
        """Adjust brightness by delta, clamping to [MIN, MAX]."""
        with self._lock:
            old = self._brightness
            self._brightness = max(BRIGHTNESS_MIN, min(BRIGHTNESS_MAX, old + delta))
            if self._brightness != old:
                logger.info(f"Knob: brightness {old}% → {self._brightness}%")

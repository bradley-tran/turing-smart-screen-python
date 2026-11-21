#!/usr/bin/env python3
"""
window_mirror.py — Mirror a window to the Racer USB display.

Uses xdg-desktop-portal ScreenCast to capture a selected window via PipeWire,
then streams scaled frames to the Racer USB display using LcdCommRacer.

Usage:
    sudo python3 window_mirror.py            # interactive window picker
    sudo python3 window_mirror.py --fps 15   # cap frame rate
"""

import argparse
import os
import signal
import sys
import time
import threading


def _check_environment():
    """Warn if running as root — D-Bus session bus won't work."""
    if os.getuid() == 0:
        print("ERROR: Do not run this script with sudo.")
        print("       The D-Bus session bus rejects root connections.")
        print()
        print("Instead, install a udev rule for the Racer USB device:")
        print('  echo \'SUBSYSTEM=="usb", ATTR{idVendor}=="34c7", '
              'ATTR{idProduct}=="2114", MODE="0666", TAG+="uaccess"\' '
              '| sudo tee /etc/udev/rules.d/99-racer-display.rules')
        print("  sudo udevadm control --reload-rules && sudo udevadm trigger")
        print()
        print("Then run without sudo:")
        print("  python3 window_mirror.py")
        sys.exit(1)


_check_environment()

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')

from gi.repository import Gst, GLib, Gio, GstApp
from PIL import Image

# ── Racer display imports ───────────────────────────────────────────────────
from library.lcd.lcd_comm_racer import LcdCommRacer
from library.lcd.lcd_comm import Orientation
from library.log import logger


# ── Portal D-Bus constants ──────────────────────────────────────────────────
PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
PORTAL_SCREENCAST = "org.freedesktop.portal.ScreenCast"
PORTAL_REQUEST = "org.freedesktop.portal.Request"

SOURCE_TYPE_WINDOW = 2  # bit 1 = window


class WindowMirror:
    """Captures a window via xdg-desktop-portal and mirrors it to the Racer display."""

    def __init__(self, fps: int = 20):
        self.fps = fps
        self.min_frame_interval = 1.0 / fps

        # GStreamer
        Gst.init(None)
        self.pipeline = None
        self.loop = GLib.MainLoop()

        # Portal state
        self.session_path = None
        self.pipewire_node_id = None
        self._request_counter = 0
        self._sender_name = None

        # Display
        self.lcd = None
        self._last_frame_time = 0.0
        self._running = True
        self._frame_count = 0
        self._start_time = 0.0

        # D-Bus
        self.bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)

    def _get_request_token(self) -> str:
        self._request_counter += 1
        return f"wmirror{self._request_counter}"

    def _get_sender_name(self) -> str:
        """Get the unique bus name, formatted for portal object paths."""
        if self._sender_name is None:
            name = self.bus.get_unique_name()
            # Convert ":1.234" → "1_234"
            self._sender_name = name.lstrip(":").replace(".", "_")
        return self._sender_name

    def _portal_call(self, method: str, params: GLib.Variant,
                     callback) -> None:
        """Call a portal method and subscribe to its Response signal."""
        token = self._get_request_token()
        sender = self._get_sender_name()
        request_path = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"

        # Subscribe to the Response signal BEFORE making the call
        self.bus.signal_subscribe(
            PORTAL_BUS,
            PORTAL_REQUEST,
            "Response",
            request_path,
            None,
            Gio.DBusSignalFlags.NONE,
            callback,
        )

        # Inject the handle_token into the options dict
        # The params are a tuple; the last element is the options dict
        self.bus.call_sync(
            PORTAL_BUS,
            PORTAL_PATH,
            PORTAL_SCREENCAST,
            method,
            params,
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None,
        )

    def _on_create_session(self, connection, sender, path, iface, signal, params):
        """Handle CreateSession response."""
        response, results = params.unpack()
        if response != 0:
            logger.error(f"CreateSession failed: response={response}")
            self.loop.quit()
            return

        self.session_path = results.get("session_handle", "")
        logger.info(f"Portal session created: {self.session_path}")

        # Now call SelectSources
        token = self._get_request_token()
        options = {
            "handle_token": GLib.Variant("s", token),
            "types": GLib.Variant("u", SOURCE_TYPE_WINDOW),
            "multiple": GLib.Variant("b", False),
        }

        sender = self._get_sender_name()
        request_path = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"

        self.bus.signal_subscribe(
            PORTAL_BUS, PORTAL_REQUEST, "Response", request_path,
            None, Gio.DBusSignalFlags.NONE, self._on_select_sources,
        )

        self.bus.call_sync(
            PORTAL_BUS, PORTAL_PATH, PORTAL_SCREENCAST, "SelectSources",
            GLib.Variant("(oa{sv})", (self.session_path, options)),
            None, Gio.DBusCallFlags.NONE, -1, None,
        )

    def _on_select_sources(self, connection, sender, path, iface, signal, params):
        """Handle SelectSources response."""
        response, results = params.unpack()
        if response != 0:
            logger.error(f"SelectSources failed: response={response}")
            self.loop.quit()
            return

        logger.info("Sources selected, starting stream...")

        # Now call Start
        token = self._get_request_token()
        options = {
            "handle_token": GLib.Variant("s", token),
        }

        sender = self._get_sender_name()
        request_path = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"

        self.bus.signal_subscribe(
            PORTAL_BUS, PORTAL_REQUEST, "Response", request_path,
            None, Gio.DBusSignalFlags.NONE, self._on_start,
        )

        self.bus.call_sync(
            PORTAL_BUS, PORTAL_PATH, PORTAL_SCREENCAST, "Start",
            GLib.Variant("(osa{sv})", (self.session_path, "", options)),
            None, Gio.DBusCallFlags.NONE, -1, None,
        )

    def _on_start(self, connection, sender, path, iface, signal, params):
        """Handle Start response — extract PipeWire node ID and launch pipeline."""
        response, results = params.unpack()
        if response != 0:
            logger.error(f"Start failed: response={response}")
            self.loop.quit()
            return

        streams = results.get("streams", [])
        if not streams:
            logger.error("No streams returned from portal")
            self.loop.quit()
            return

        # streams is an array of (node_id, properties)
        self.pipewire_node_id = streams[0][0]
        logger.info(f"PipeWire node ID: {self.pipewire_node_id}")

        # Initialize Racer display now
        self._init_display()

        # Build and start GStreamer pipeline
        self._start_pipeline()

    def _init_display(self):
        """Initialize the Racer USB display."""
        logger.info("Initializing Racer display...")
        self.lcd = LcdCommRacer(
            com_port="AUTO",
            display_width=320,
            display_height=480,
        )
        self.lcd.InitializeComm()
        self.lcd.SetOrientation(Orientation.LANDSCAPE)
        self.lcd.SetBrightness(level=50)
        logger.info("Racer display ready (480×320 landscape)")

    def _start_pipeline(self):
        """Build and start the GStreamer pipeline."""
        disp_w = self.lcd.get_width()   # 480 in landscape
        disp_h = self.lcd.get_height()  # 320 in landscape

        pipeline_str = (
            f"pipewiresrc path={self.pipewire_node_id} ! "
            f"videoconvert ! "
            f"videoscale method=bilinear ! "
            f"video/x-raw,format=RGB,width={disp_w},height={disp_h} ! "
            f"appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false"
        )

        logger.info(f"GStreamer pipeline: {pipeline_str}")
        self.pipeline = Gst.parse_launch(pipeline_str)

        # Connect appsink callback
        sink = self.pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_new_sample)

        # Watch for pipeline errors
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_pipeline_error)
        bus.connect("message::eos", self._on_pipeline_eos)

        self.pipeline.set_state(Gst.State.PLAYING)
        self._start_time = time.monotonic()
        logger.info("Pipeline started — mirroring window to display")

    def _on_new_sample(self, sink) -> Gst.FlowReturn:
        """Called by appsink for each new frame."""
        if not self._running:
            return Gst.FlowReturn.EOS

        # Rate limit
        now = time.monotonic()
        if now - self._last_frame_time < self.min_frame_interval:
            return Gst.FlowReturn.OK
        self._last_frame_time = now

        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        try:
            image = Image.frombytes("RGB", (width, height), map_info.data)
            self.lcd.DisplayPILImage(image)
            self._frame_count += 1

            if self._frame_count % 100 == 0:
                elapsed = time.monotonic() - self._start_time
                fps = self._frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Mirror: {self._frame_count} frames, {fps:.1f} avg FPS")
        except Exception as e:
            logger.error(f"Frame error: {e}")
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def _on_pipeline_error(self, bus, msg):
        err, debug = msg.parse_error()
        logger.error(f"Pipeline error: {err.message}")
        logger.debug(f"Pipeline debug: {debug}")
        self.stop()

    def _on_pipeline_eos(self, bus, msg):
        logger.info("Pipeline reached end-of-stream")
        self.stop()

    def start(self):
        """Start the portal session. The GLib MainLoop drives everything."""
        logger.info("Starting window mirror — requesting portal session...")

        token = self._get_request_token()
        options = {
            "handle_token": GLib.Variant("s", token),
            "session_handle_token": GLib.Variant("s", "wmirror_session"),
        }

        sender = self._get_sender_name()
        request_path = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"

        self.bus.signal_subscribe(
            PORTAL_BUS, PORTAL_REQUEST, "Response", request_path,
            None, Gio.DBusSignalFlags.NONE, self._on_create_session,
        )

        self.bus.call_sync(
            PORTAL_BUS, PORTAL_PATH, PORTAL_SCREENCAST, "CreateSession",
            GLib.Variant("(a{sv})", (options,)),
            None, Gio.DBusCallFlags.NONE, -1, None,
        )

        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """Clean shutdown."""
        if not self._running:
            return
        self._running = False

        if self._frame_count > 0:
            elapsed = time.monotonic() - self._start_time
            fps = self._frame_count / elapsed if elapsed > 0 else 0
            logger.info(f"Mirror stopped: {self._frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None

        if self.lcd:
            try:
                self.lcd.closeSerial()
            except Exception:
                pass
            self.lcd = None

        try:
            self.loop.quit()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Mirror a window to the Racer USB display"
    )
    parser.add_argument(
        "--fps", type=int, default=20,
        help="Maximum frame rate (default: 20)"
    )
    args = parser.parse_args()

    # Signal handling
    mirror = WindowMirror(fps=args.fps)

    def sighandler(signum, frame):
        logger.info("Signal received, shutting down...")
        mirror.stop()

    signal.signal(signal.SIGINT, sighandler)
    signal.signal(signal.SIGTERM, sighandler)
    if hasattr(signal, 'SIGQUIT'):
        signal.signal(signal.SIGQUIT, sighandler)

    mirror.start()


if __name__ == "__main__":
    main()

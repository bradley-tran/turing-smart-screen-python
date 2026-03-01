# Racer USB Display backend for turing-smart-screen-python
#
# Implements the Racer USB display protocol (VID 0x34C7, PID 0x2114) as an
# LcdComm backend. Encodes the screen as 32×8 JPEG strips and sends them
# via USB bulk transfers.
#
# Performance optimizations:
#   - TurboJPEG for ~10× faster JPEG encoding vs Pillow
#   - Block-level hash cache: only re-encode strips whose pixels changed
#   - Batched DisplayPILImage: composites are grouped, sent as one frame
#   - No wait_ready polling (device ignores GetStatus)

import hashlib
import io
import struct
import threading
import time
from typing import Optional, Tuple

import numpy as np
import usb.core
import usb.util
from PIL import Image, ImageEnhance

from library.lcd.lcd_comm import *
from library.log import logger

# Try to use TurboJPEG (10x faster), fall back to Pillow
_turbojpeg = None
try:
    from turbojpeg import TurboJPEG, TJSAMP_444, TJPF_RGB
    _turbojpeg = TurboJPEG()
    logger.info("Using TurboJPEG for JPEG encoding (fast path)")
except Exception:
    logger.info("TurboJPEG not available, using Pillow JPEG encoder (slower)")

# ── Device identification ───────────────────────────────────────────────────
VENDOR_ID = 0x34C7
PRODUCT_ID = 0x2114

# ── USB constants ───────────────────────────────────────────────────────────
BM_REQUEST_OUT = 0x41
BM_REQUEST_IN = 0xC1
CONTROL_TIMEOUT_MS = 500

REQ_GET_DEVICE_ID = 0x50
REQ_GET_MONITOR_FEATURE = 0x52
REQ_SET_RESOLUTION = 0x81
REQ_SET_QUANT_TABLE = 0x83
REQ_SET_DEVICE_ID = 0x8F

# ── Strip / encoding ────────────────────────────────────────────────────────
STRIP_WIDTH = 32
STRIP_HEIGHT = 8
BULK_OUT_EP = 0x01
BULK_PADDING_ALIGNMENT = 128
PADDING_FILL = b'\xFF\xD9\xFF\xFF'

DEFAULT_QUALITY = 100
FLUSH_INTERVAL_S = 0.05  # 50ms batch window


# ── Helper functions ────────────────────────────────────────────────────────

def _build_strip_header(jpeg_len: int, strip_index: int, is_last: bool) -> bytes:
    word_len = (jpeg_len + 3) // 4
    len_field = max(word_len - 1, 0) & 0xFF
    word = 0x01
    word |= (len_field << 2)
    word |= ((strip_index & 0xFFFF) << 10)
    word |= 0xD0_00_00_00
    if is_last:
        word |= 0x08_00_00_00
    return struct.pack('<I', word)


def _find_entropy_start(jpeg_data: bytes) -> int:
    """Find byte offset where entropy data begins (after SOS header)."""
    pos = 0
    while pos + 1 < len(jpeg_data):
        if jpeg_data[pos] != 0xFF:
            pos += 1
            continue
        marker = jpeg_data[pos + 1]
        if marker == 0xD8 or marker == 0x00:
            pos += 2
        elif marker == 0xDA:
            seg_len = struct.unpack('>H', jpeg_data[pos + 2:pos + 4])[0]
            return pos + 2 + seg_len
        else:
            if pos + 4 > len(jpeg_data):
                pos += 2
                continue
            seg_len = struct.unpack('>H', jpeg_data[pos + 2:pos + 4])[0]
            pos += 2 + seg_len
    raise ValueError("No SOS marker found")


def _pad_to_alignment(buf: bytearray):
    remainder = len(buf) % BULK_PADDING_ALIGNMENT
    if remainder == 0:
        return
    pad_bytes = BULK_PADDING_ALIGNMENT - remainder
    full_words = pad_bytes // 4
    extra = pad_bytes % 4
    for _ in range(full_words):
        buf.extend(PADDING_FILL)
    if extra > 0:
        buf.extend(PADDING_FILL[:extra])


def _extract_dqt_from_jpeg(jpeg: bytes) -> bytes:
    """Extract DQT marker segments from raw JPEG data."""
    dqt_data = bytearray()
    pos = 0
    while pos + 1 < len(jpeg):
        if jpeg[pos] != 0xFF:
            pos += 1
            continue
        marker = jpeg[pos + 1]
        if marker == 0xDB:
            if pos + 4 > len(jpeg):
                break
            seg_len = struct.unpack('>H', jpeg[pos + 2:pos + 4])[0]
            total = 2 + seg_len
            dqt_data.extend(jpeg[pos:pos + total])
            pos += total
        elif marker == 0xD8 or marker == 0x00:
            pos += 2
        elif marker == 0xDA:
            break
        else:
            if pos + 4 > len(jpeg):
                break
            seg_len = struct.unpack('>H', jpeg[pos + 2:pos + 4])[0]
            pos += 2 + seg_len

    if not dqt_data:
        raise ValueError("No DQT markers found")
    return bytes(dqt_data)


def _extract_dqt(quality: int) -> bytes:
    """Extract DQT from the SAME encoder that will be used for frames."""
    dummy_arr = np.full((STRIP_HEIGHT, STRIP_WIDTH, 3), 128, dtype=np.uint8)
    if _turbojpeg:
        jpeg = _turbojpeg.encode(dummy_arr, quality=quality,
                                  jpeg_subsample=TJSAMP_444, pixel_format=TJPF_RGB)
    else:
        dummy = Image.fromarray(dummy_arr, 'RGB')
        buf = io.BytesIO()
        dummy.save(buf, format='JPEG', quality=quality, subsampling=0)
        jpeg = buf.getvalue()
    return _extract_dqt_from_jpeg(jpeg)


def _probe_entropy_offset(quality: int) -> int:
    """Compute the fixed JPEG header length for 32×8 blocks at given quality."""
    dummy = Image.new('RGB', (STRIP_WIDTH, STRIP_HEIGHT), (128, 128, 128))
    buf = io.BytesIO()
    dummy.save(buf, format='JPEG', quality=quality, subsampling=0)
    return _find_entropy_start(buf.getvalue())


# ── LcdCommRacer ────────────────────────────────────────────────────────────

class LcdCommRacer(LcdComm):
    """Racer USB display backend with block-level caching and TurboJPEG."""

    def __init__(self, com_port: str = "AUTO", display_width: int = 320, display_height: int = 480,
                 update_queue: Optional[queue.Queue] = None):
        LcdComm.__init__(self, com_port, display_width, display_height, update_queue)

        self.quality = DEFAULT_QUALITY
        self.orientation = Orientation.PORTRAIT

        # Pre-compute the entropy offset for Pillow fallback
        self._entropy_offset = _probe_entropy_offset(self.quality)

        # Screen buffer
        self.screen_image = Image.new("RGB", (self.get_width(), self.get_height()), (0, 0, 0))

        # Block-level cache: maps block index → (pixel_hash, encoded_strip_bytes)
        self._strip_cache = {}

        # Batched update state
        self._dirty = False
        self._flush_lock = threading.Lock()
        self._flush_timer = None
        self._stopping = False

        # Software brightness (0–100 %)
        self._brightness = 100

        self.dev = None
        self._open_device()

    def _open_device(self):
        """Open (or re-open) the USB device."""
        # Release any previous handle
        if self.dev:
            try:
                usb.util.release_interface(self.dev, 0)
                usb.util.dispose_resources(self.dev)
            except Exception:
                pass
            self.dev = None

        self.dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        if self.dev is None:
            raise RuntimeError(
                f"Racer USB display not found (VID={VENDOR_ID:#06x} PID={PRODUCT_ID:#06x}). "
                "Is the device connected?"
            )

        logger.info(f"Found Racer display: bus={self.dev.bus} addr={self.dev.address}")

        for cfg in self.dev:
            for intf in cfg:
                try:
                    if self.dev.is_kernel_driver_active(intf.bInterfaceNumber):
                        self.dev.detach_kernel_driver(intf.bInterfaceNumber)
                except (usb.core.USBError, NotImplementedError):
                    pass

        try:
            self.dev.set_configuration(1)
        except usb.core.USBError:
            pass
        try:
            usb.util.claim_interface(self.dev, 0)
        except usb.core.USBError as e:
            raise RuntimeError(f"Cannot claim USB interface 0: {e}") from e

        logger.info("Racer USB display opened")

        # Read device ID
        try:
            device_id = self._get_device_id()
            if device_id == 0xFFFFFFFF or device_id == 0:
                new_id = int(time.time()) & 0x7FFFFFFF
                self._set_device_id(new_id)
            else:
                logger.info(f"Device ID: {device_id:#010x}")
        except usb.core.USBError:
            pass

        try:
            self._get_monitor_feature()
        except usb.core.USBError:
            pass

    def _reconnect(self):
        """Try to reconnect to the device after it disconnects/resets."""
        logger.warning("Device disconnected, attempting reconnection...")
        for attempt in range(30):  # retry for up to 30 seconds
            time.sleep(1)
            try:
                self._open_device()
                # Re-send quant tables and resolution
                self._send_quant_tables(self.quality)
                w, h = self.screen_image.size
                try:
                    self._set_resolution(w, h)
                except usb.core.USBError:
                    pass
                self._strip_cache.clear()  # force full re-encode
                logger.info(f"Reconnected after {attempt + 1}s")
                return True
            except Exception as e:
                if attempt % 5 == 4:
                    logger.debug(f"Reconnect attempt {attempt + 1}: {e}")
        logger.error("Failed to reconnect after 30 attempts")
        return False

    def __del__(self):
        try:
            self.closeSerial()
        except Exception:
            pass

    @staticmethod
    def auto_detect_com_port() -> Optional[str]:
        return None

    def closeSerial(self):
        self._stopping = True
        if self._flush_timer:
            self._flush_timer.cancel()
        if self.dev:
            usb.util.release_interface(self.dev, 0)
            usb.util.dispose_resources(self.dev)
            self.dev = None

    # ── Control transfers ───────────────────────────────────────────────────

    def _control_in(self, req, wVal, wIdx, length):
        return self.dev.ctrl_transfer(BM_REQUEST_IN, req, wVal, wIdx, length, timeout=CONTROL_TIMEOUT_MS)

    def _control_out(self, req, wVal, wIdx, data):
        self.dev.ctrl_transfer(BM_REQUEST_OUT, req, wVal, wIdx, data, timeout=CONTROL_TIMEOUT_MS)

    def _get_device_id(self):
        data = self._control_in(REQ_GET_DEVICE_ID, 0, 0, 4)
        return struct.unpack('<I', bytes(data))[0]

    def _set_device_id(self, did):
        self._control_out(REQ_SET_DEVICE_ID, 0, 0, struct.pack('<I', did))

    def _get_monitor_feature(self):
        return self._control_in(REQ_GET_MONITOR_FEATURE, 0, 0, 1)[0]

    def _set_resolution(self, w, h):
        self._control_out(REQ_SET_RESOLUTION, 0, 0, struct.pack('<HH', w, h))

    def _send_quant_tables(self, quality):
        dqt = _extract_dqt(quality)
        self._control_out(REQ_SET_QUANT_TABLE, 1, 0, dqt)
        logger.info(f"SetQuantTable: {len(dqt)}B (quality={quality})")

    # ── JPEG encoding with caching ──────────────────────────────────────────

    def _encode_block_turbo(self, block_arr: np.ndarray) -> bytes:
        """Encode a 32×8 block with TurboJPEG, return stripped entropy data."""
        jpeg = _turbojpeg.encode(block_arr, quality=self.quality,
                                  jpeg_subsample=TJSAMP_444, pixel_format=TJPF_RGB)
        return jpeg[_find_entropy_start(jpeg):]

    def _encode_block_pillow(self, block_arr: np.ndarray) -> bytes:
        """Encode a 32×8 block with Pillow, return stripped entropy data."""
        block_img = Image.fromarray(block_arr, 'RGB')
        buf = io.BytesIO()
        block_img.save(buf, format='JPEG', quality=self.quality, subsampling=0)
        return buf.getvalue()[self._entropy_offset:]

    def _encode_frame(self, image: Image.Image) -> bytes:
        """Encode frame as single contiguous payload with block-level caching."""
        w, h = image.size
        cols = w // STRIP_WIDTH
        rows = h // STRIP_HEIGHT
        total_blocks = cols * rows

        pixels = np.ascontiguousarray(np.array(image))
        encode_fn = self._encode_block_turbo if _turbojpeg else self._encode_block_pillow

        payload = bytearray()
        cache_hits = 0

        for idx in range(total_blocks):
            row = idx // cols
            col = idx % cols
            y0 = row * STRIP_HEIGHT
            x0 = col * STRIP_WIDTH
            is_last = (idx == total_blocks - 1)

            block = pixels[y0:y0 + STRIP_HEIGHT, x0:x0 + STRIP_WIDTH]
            block_hash = hashlib.md5(block.tobytes()).digest()

            cached = self._strip_cache.get(idx)
            if cached and cached[0] == block_hash:
                stripped = cached[1]
                cache_hits += 1
            else:
                stripped = encode_fn(np.ascontiguousarray(block))
                self._strip_cache[idx] = (block_hash, stripped)

            header = _build_strip_header(len(stripped), idx, is_last)
            payload.extend(header)
            payload.extend(stripped)
            pad = (4 - (len(stripped) % 4)) % 4
            if pad:
                payload.extend(b'\x00' * pad)

        _pad_to_alignment(payload)

        if total_blocks > 0:
            logger.debug(f"Cache: {cache_hits}/{total_blocks} hits ({100*cache_hits/total_blocks:.0f}%), payload={len(payload)/1024:.0f}KB")

        return bytes(payload)

    # ── Bulk transfer ───────────────────────────────────────────────────────

    def _send_payload(self, payload: bytes):
        """Send entire frame payload in one bulk transfer."""
        w, h = self.screen_image.size
        try:
            self._set_resolution(w, h)
        except usb.core.USBError:
            pass

        for attempt in range(2):
            try:
                # Fire-and-forget: device DMA's the data (~10ms for 280KB at USB 2.0).
                # Transfer always "times out" because device never ACKs the last packet.
                self.dev.write(BULK_OUT_EP, payload, timeout=50)
            except usb.core.USBTimeoutError:
                return  # Expected — device accepted the data
            except usb.core.USBError as e:
                if e.errno == 19:  # ENODEV
                    self._reconnect()
                    return
                elif e.errno == 12 and attempt == 0:  # ENOMEM — retry once
                    logger.debug("ENOMEM on bulk write, retrying after 100ms...")
                    time.sleep(0.1)
                    continue
                else:
                    logger.warning(f"Bulk write error: {e}")
                    try:
                        self.dev.clear_halt(BULK_OUT_EP)
                    except usb.core.USBError:
                        pass
                    return
            return  # write() returned normally (rare but possible)

    # ── Batched flush ───────────────────────────────────────────────────────

    def _schedule_flush(self):
        if self._flush_timer is not None:
            return
        self._flush_timer = threading.Timer(FLUSH_INTERVAL_S, self._do_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _do_flush(self):
        if self._stopping:
            return
        with self._flush_lock:
            self._flush_timer = None
            if not self._dirty:
                return
            self._dirty = False
            image = self.screen_image.copy()

            # Apply software brightness before encoding
            if self._brightness < 100:
                factor = self._brightness / 100.0
                image = ImageEnhance.Brightness(image).enhance(factor)

            try:
                t0 = time.monotonic()
                payload = self._encode_frame(image)
                t_enc = time.monotonic() - t0
                self._send_payload(payload)
                t_total = time.monotonic() - t0
                logger.debug(f"Frame: {len(payload)/1024:.0f}KB, encode={t_enc*1000:.0f}ms, total={t_total*1000:.0f}ms")
            except Exception as e:
                logger.error(f"Failed to send frame: {e}")

    def _flush_sync(self):
        if self._flush_timer:
            self._flush_timer.cancel()
            self._flush_timer = None
        self._dirty = False

        # Apply software brightness before encoding
        image = self.screen_image
        if self._brightness < 100:
            factor = self._brightness / 100.0
            image = ImageEnhance.Brightness(image).enhance(factor)

        try:
            t0 = time.monotonic()
            payload = self._encode_frame(image)
            t_enc = time.monotonic() - t0
            self._send_payload(payload)
            t_total = time.monotonic() - t0
            logger.debug(f"Frame (sync): {len(payload)/1024:.0f}KB, encode={t_enc*1000:.0f}ms, total={t_total*1000:.0f}ms")
        except Exception as e:
            logger.error(f"Failed to send frame: {e}")

    # ── LcdComm interface ───────────────────────────────────────────────────

    def InitializeComm(self):
        for attempt in range(3):
            try:
                self._send_quant_tables(self.quality)
                w, h = self.get_width(), self.get_height()
                self._set_resolution(w, h)
                logger.info(f"Racer display initialized (q{self.quality}, {w}×{h})")
                return
            except usb.core.USBError as e:
                logger.warning(f"InitializeComm attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    try:
                        self._reconnect()
                    except Exception:
                        time.sleep(1)
        logger.error("InitializeComm failed after 3 attempts")

    def Reset(self):
        pass

    def Clear(self):
        self.screen_image = Image.new("RGB", (self.get_width(), self.get_height()), (0, 0, 0))
        self._strip_cache.clear()
        self._flush_sync()

    def ScreenOff(self):
        pass

    def ScreenOn(self):
        pass

    def SetBrightness(self, level: int = 25):
        level = max(0, min(100, level))
        if level != self._brightness:
            logger.info(f"Racer: brightness {self._brightness}% → {level}%")
            self._brightness = level
            self._strip_cache.clear()  # force full re-encode at new brightness
            if not self._stopping:
                self._dirty = True
                self._schedule_flush()

    def SetBackplateLedColor(self, led_color: Tuple[int, int, int] = (255, 255, 255)):
        pass

    def SetOrientation(self, orientation: Orientation = Orientation.PORTRAIT):
        self.orientation = orientation
        w, h = self.get_width(), self.get_height()
        self.screen_image = Image.new("RGB", (w, h), (0, 0, 0))
        self._strip_cache.clear()
        try:
            self._set_resolution(w, h)
        except usb.core.USBError:
            pass
        logger.info(f"Orientation set: {w}×{h}")

    def DisplayPILImage(
            self,
            image: Image.Image,
            x: int = 0, y: int = 0,
            image_width: int = 0,
            image_height: int = 0
    ):
        if not image_height:
            image_height = image.size[1]
        if not image_width:
            image_width = image.size[0]

        if image.size[1] > self.get_height():
            image_height = self.get_height()
        if image.size[0] > self.get_width():
            image_width = self.get_width()

        if image_width != image.size[0] or image_height != image.size[1]:
            image = image.crop((0, 0, image_width, image_height))

        with self.update_queue_mutex:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            self.screen_image.paste(image, (x, y))
            self._dirty = True
            self._schedule_flush()

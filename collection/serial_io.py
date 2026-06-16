"""
Lectura del stream binario del firmware ESP32 + ADS1256.

Protocolo (ver sketchs/esp32/src/main.cpp y monitor.py):
    Frame de 27 bytes:
      [0..1]   magic 0xAA 0x55
      [2..5]   t_micros uint32 LE
      [6..9]   emg0     int32  LE   (extensor del indice, ADS AIN0/AIN1)
      [10..13] emg1     int32  LE   (flexor   del indice, ADS AIN2/AIN3)
      [14..17] emg2     int32  LE   (extensor del medio,  ADS AIN4/AIN5)
      [18..21] emg3     int32  LE   (flexor   del medio,  ADS AIN6/AIN7)
      [22..23] joy_a    uint16 LE   (joystick A, ESP32 GPIO5)
      [24..25] joy_b    uint16 LE   (joystick B, ESP32 GPIO6)
      [26]     CRC-8 sobre [2..25]   (poly 0x07, init 0x00)
"""
import struct
from dataclasses import dataclass, field
from typing import Dict, Optional

import serial


# ----- Protocol constants -----
FRAME_SIZE = 27
MAGIC = b"\xAA\x55"
PAYLOAD_FMT = "<I4iHH"   # uint32 + 4 int32 + 2 uint16


def _crc8_table_init() -> bytes:
    out = bytearray(256)
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if (c & 0x80) else (c << 1) & 0xFF
        out[i] = c
    return bytes(out)


_CRC8_TABLE = _crc8_table_init()


def crc8(data: bytes) -> int:
    c = 0
    for b in data:
        c = _CRC8_TABLE[c ^ b]
    return c


@dataclass
class Sample:
    t_micros: float
    emg0: float
    emg1: float
    emg2: float
    emg3: float
    joy_a: float
    joy_b: float
    extra: Dict[str, float] = field(default_factory=dict)


class SerialReader:
    """
    Lee frames binarios del stream del ESP32 y devuelve Sample.

    El protocolo trae magic header + CRC para resincronizacion en caso de
    perder bytes (cable inestable, restart del MCU, etc). Bytes fuera del
    formato se contabilizan en dropped_bytes; frames con CRC malo en
    bad_lines.
    """

    def __init__(
        self,
        port: str,
        baud: int = 921600,
        timeout: float = 0.05,
    ):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.buf = bytearray()
        self.good_lines = 0       # nombre heredado del legacy CSV-based reader
        self.bad_lines = 0
        self.dropped_bytes = 0

    def _find_frame(self) -> Optional[bytes]:
        while True:
            idx = self.buf.find(MAGIC)
            if idx < 0:
                if len(self.buf) > 1024:
                    self.dropped_bytes += len(self.buf) - 1
                    del self.buf[:-1]
                return None
            if idx > 0:
                self.dropped_bytes += idx
                del self.buf[:idx]
            if len(self.buf) < FRAME_SIZE:
                return None
            frame = bytes(self.buf[:FRAME_SIZE])
            if frame[26] == crc8(frame[2:26]):
                del self.buf[:FRAME_SIZE]
                return frame
            # CRC bad: el magic fue casualidad. Avanzar 1 byte y reintentar.
            self.bad_lines += 1
            del self.buf[0]

    def read(self) -> Optional[Sample]:
        nwait = self.ser.in_waiting
        if nwait:
            self.buf.extend(self.ser.read(nwait))
        else:
            # Lectura bloqueante chica para no quemar CPU; el timeout del
            # puerto serial decide cuanto esperar como maximo.
            chunk = self.ser.read(FRAME_SIZE)
            if chunk:
                self.buf.extend(chunk)
        frame = self._find_frame()
        if frame is None:
            return None
        t_us, e0, e1, e2, e3, j_a, j_b = struct.unpack(PAYLOAD_FMT, frame[2:26])
        self.good_lines += 1
        return Sample(
            t_micros=float(t_us),
            emg0=float(e0),
            emg1=float(e1),
            emg2=float(e2),
            emg3=float(e3),
            joy_a=float(j_a),
            joy_b=float(j_b),
        )

    def flush_input(self) -> None:
        self.ser.reset_input_buffer()
        self.buf.clear()

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass

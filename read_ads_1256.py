import serial
import struct
import time
import csv

PORT = "/dev/cu.usbmodem21101"
BAUD = 2_000_000

SYNC = b"\x5A\xA5"  # ESP32 little-endian de 0xA55A
PACKET_SIZE = 16    # uint16 + uint32 + uint32 + int32 + uint16

FMT = "<HIIiH"

def checksum16(data: bytes) -> int:
    return sum(data) & 0xFFFF

ser = serial.Serial(PORT, BAUD, timeout=1)

buffer = b""
last_seq = None
drops = 0
count = 0
t0 = time.time()

with open("ads_emg_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["host_t", "seq", "t_us", "emg_raw"])

    while True:
        buffer += ser.read(4096)

        while True:
            idx = buffer.find(SYNC)
            if idx < 0:
                buffer = buffer[-1:]
                break

            if idx > 0:
                buffer = buffer[idx:]

            if len(buffer) < PACKET_SIZE:
                break

            pkt = buffer[:PACKET_SIZE]
            buffer = buffer[PACKET_SIZE:]

            sync, seq, t_us, emg, chk = struct.unpack(FMT, pkt)

            if chk != checksum16(pkt[:-2]):
                continue

            if last_seq is not None and seq != last_seq + 1:
                drops += seq - last_seq - 1

            last_seq = seq
            count += 1

            writer.writerow([time.time(), seq, t_us, emg])

            if count % 1000 == 0:
                print(emg)
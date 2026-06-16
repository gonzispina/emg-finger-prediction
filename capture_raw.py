"""
Captura raw del stream binario del ESP32 a CSV.

Para debug del hardware, sin protocolo de tareas. Graba N segundos del
stream y guarda como CSV con las 7 columnas del schema nuevo. Despues
se inspecciona offline con plot_raw.py.

Uso:
    python capture_raw.py --port /dev/cu.usbserial-XXX --seconds 30 \\
                          --out data/debug/test_001.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

from collection.serial_io import SerialReader


CSV_COLUMNS = ["pc_timestamp", "t_micros",
               "emg0", "emg1", "emg2", "emg3",
               "joy_a", "joy_b"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--seconds", type=float, required=True,
                   help="duracion de la captura (s)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--countdown", type=int, default=3,
                   help="segundos de countdown antes de empezar a grabar")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Conectando a {args.port} @ {args.baud}...")
    reader = SerialReader(args.port, args.baud)
    time.sleep(2)
    reader.flush_input()
    print("Listo.")

    if args.countdown > 0:
        print(f"\nEmpieza en:")
        for n in range(args.countdown, 0, -1):
            print(f"  {n}...", flush=True)
            time.sleep(1)
        print(f"  GO. Capturando {args.seconds:.0f} s.\n")

    start = time.time()
    next_tick = start + 1.0
    written = 0
    try:
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            while True:
                now = time.time()
                if now - start >= args.seconds:
                    break
                if now >= next_tick:
                    remaining = max(0, int(round(args.seconds - (now - start))))
                    print(f"  ...{remaining} s   ({written} samples, "
                          f"{written / max(now - start, 0.001):.0f} Hz)", flush=True)
                    next_tick += 1.0
                s = reader.read()
                if s is None:
                    continue
                writer.writerow([now, s.t_micros,
                                 s.emg0, s.emg1, s.emg2, s.emg3,
                                 s.joy_a, s.joy_b])
                written += 1
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        reader.close()

    elapsed = time.time() - start
    print(f"\nFin. {written} samples en {elapsed:.1f} s "
          f"(fs efectiva = {written / elapsed:.0f} Hz)")
    print(f"Buenos: {reader.good_lines}, "
          f"CRC malos: {reader.bad_lines}, "
          f"bytes dropeados: {reader.dropped_bytes}")
    print(f"CSV escrito en {args.out}")


if __name__ == "__main__":
    main()

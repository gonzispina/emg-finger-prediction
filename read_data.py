import serial
import csv
import argparse
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", required=True)
parser.add_argument("--port", default="/dev/cu.usbserial-2110")
parser.add_argument("--baud", type=int, default=230400)
parser.add_argument("--cycles", type=int, default=20)
args = parser.parse_args()

ser = serial.Serial(args.port, args.baud, timeout=1)
time.sleep(2)

protocol = []

for i in range(args.cycles):
    protocol.append(("rest", random.uniform(1.5, 2.5)))

    protocol.append(("prepare_up", 1.5))
    protocol.append(("up", random.uniform(1.0, 2.5)))

    protocol.append(("rest", random.uniform(1.0, 2.0)))

    protocol.append(("prepare_down", 1.5))
    protocol.append(("down", random.uniform(1.0, 2.5)))

print("Preparado.")

for n in [3, 2, 1]:
    print(n)
    time.sleep(1)

print("START!")

try:
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "pc_time",
            "t_micros",
            "y_top",
            "y_bottom",
            "emg_ext",
            "emg_flex",
            "command",
            "block_id",
        ])

        block_id = 0
        counter = 0

        for command, duration in protocol:
            block_id += 1

            if command == "prepare_up":
                print("\nPREPARE TO PUSH UP")
            elif command == "prepare_down":
                print("\nPREPARE TO PUSH DOWN")
            else:
                print(f"\n{command.upper()}... {duration:.2f}s")

            block_start = time.time()

            while time.time() - block_start < duration:
                raw = ser.readline().decode(errors="ignore").strip()

                if not raw:
                    continue

                parts = raw.split(",")

                if len(parts) != 5:
                    continue

                try:
                    t = int(parts[0])
                    y_top = int(parts[1])
                    y_bottom = int(parts[2])
                    emg_ext = int(parts[3])
                    emg_flex = int(parts[4])
                except ValueError:
                    continue

                writer.writerow([
                    time.time(),
                    t,
                    y_top,
                    y_bottom,
                    emg_ext,
                    emg_flex,
                    command,
                    block_id,
                ])

                counter += 1

                if counter % 100 == 0:
                    f.flush()

        print("\nProtocolo terminado.")

except KeyboardInterrupt:
    print("\nGrabación detenida manualmente.")

finally:
    ser.close()
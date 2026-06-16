"""
Dry-run del stream serial. Valida puerto, parseo y rangos de canales
ANTES de correr una sesion real. No escribe CSV.

Uso:
    python check_serial.py
    python check_serial.py --port /dev/cu.usbserial-2110 --baud 230400 --seconds 10
"""
import argparse
import glob
import sys
import time
from collections import defaultdict

from collection.serial_io import SerialReader, parse_5field


CHANNELS = ["t_micros", "y_top", "y_bottom", "flexor", "extensor"]
ADC_MAX = 1023  # Arduino Uno/Nano 10-bit ADC


def _list_ports():
    return sorted(glob.glob("/dev/cu.*"))


def _fmt_stats(name, values):
    if not values:
        return f"  {name:>10}: (sin muestras)"
    mn, mx = min(values), max(values)
    mean = sum(values) / len(values)
    span = mx - mn
    flags = []
    if name in ("y_top", "y_bottom", "flexor", "extensor"):
        if mx == 0 or mn == ADC_MAX:
            flags.append("STUCK")
        if span < 2:
            flags.append("FLAT")
        if mx >= ADC_MAX:
            flags.append("SATURATED")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""
    return f"  {name:>10}: min={mn:>7.1f}  max={mx:>7.1f}  mean={mean:>7.1f}  span={span:>6.1f}{flag_str}"


def run_dry(port: str, baud: int, seconds: float):
    print(f"\nDRY RUN — {port} @ {baud}  durante {seconds:.0f}s\n")
    print("Mientras corre: movee el dedo arriba/abajo y contrae el antebrazo")
    print("para verificar que joysticks y EMG responden.\n")

    try:
        reader = SerialReader(port, baud, parser=parse_5field)
    except Exception as e:
        print(f"ERROR al abrir {port}: {e}\n")
        ports = _list_ports()
        if ports:
            print("Puertos disponibles:")
            for p in ports:
                print(f"  {p}")
        else:
            print("No hay puertos /dev/cu.* — el Arduino no esta conectado.")
        sys.exit(1)

    time.sleep(2)
    reader.flush_input()

    samples = defaultdict(list)
    start = time.time()
    last_print = start
    last_sample = None

    try:
        while time.time() - start < seconds:
            s = reader.read()
            if s is None:
                # Mostramos algo aunque sea para no parecer colgado
                now = time.time()
                if now - last_print >= 0.5:
                    elapsed = now - start
                    print(
                        f"\r  t={elapsed:5.1f}s  good={reader.good_lines:>6}  "
                        f"bad={reader.bad_lines:>4}  rate={reader.good_lines/max(elapsed, 1e-3):>6.1f} Hz  "
                        f"(esperando...)",
                        end="", flush=True,
                    )
                    last_print = now
                continue
            last_sample = s
            for c in CHANNELS:
                samples[c].append(getattr(s, c))
            now = time.time()
            if now - last_print >= 0.5:
                elapsed = now - start
                rate = reader.good_lines / max(elapsed, 1e-3)
                print(
                    f"\r  t={elapsed:5.1f}s  good={reader.good_lines:>6}  "
                    f"bad={reader.bad_lines:>4}  rate={rate:>6.1f} Hz  "
                    f"y_top={s.y_top:>4.0f} y_bot={s.y_bottom:>4.0f} "
                    f"flx={s.flexor:>4.0f} ext={s.extensor:>4.0f}",
                    end="", flush=True,
                )
                last_print = now
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        reader.close()

    elapsed = time.time() - start
    total = reader.good_lines + reader.bad_lines
    bad_pct = (reader.bad_lines / total * 100) if total else 0.0
    rate = reader.good_lines / max(elapsed, 1e-3)

    print("\n\nResumen:")
    print(f"  duracion        : {elapsed:.2f} s")
    print(f"  buenas / malas  : {reader.good_lines} / {reader.bad_lines}  ({bad_pct:.2f}% malas)")
    print(f"  rate efectiva   : {rate:.1f} Hz")

    print("\nRangos por canal:")
    for c in CHANNELS:
        print(_fmt_stats(c, samples[c]))

    print("\nDiagnostico:")
    if reader.good_lines == 0:
        print("  X NO se recibio ninguna linea valida. Revisar firmware/cable/baud.")
        sys.exit(2)
    if bad_pct > 1.0:
        print(f"  ! {bad_pct:.1f}% de lineas corruptas — posible baud mal o ruido en el USB.")
    if rate < 500:
        print(f"  ! rate {rate:.0f} Hz bajo (esperado ~800). Verificar firmware loop / serial buffer.")
    elif rate > 1200:
        print(f"  ! rate {rate:.0f} Hz alto. Esperado ~800. Algo cambio en el firmware?")
    if not samples["y_top"] or max(samples["y_top"]) - min(samples["y_top"]) < 5:
        print("  ! y_top sin variacion: movete arriba o revisa joystick superior.")
    if not samples["y_bottom"] or max(samples["y_bottom"]) - min(samples["y_bottom"]) < 5:
        print("  ! y_bottom sin variacion: movete abajo o revisa joystick inferior.")
    if not samples["flexor"] or max(samples["flexor"]) - min(samples["flexor"]) < 5:
        print("  ! flexor sin variacion: contrae el flexor o revisa electrodo A1.")
    if not samples["extensor"] or max(samples["extensor"]) - min(samples["extensor"]) < 5:
        print("  ! extensor sin variacion: contrae el extensor o revisa electrodo A0.")
    print()


def main():
    p = argparse.ArgumentParser(description="Dry-run del stream serial")
    p.add_argument("--port", default="/dev/cu.usbserial-2110")
    p.add_argument("--baud", type=int, default=230400)
    p.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args()
    run_dry(args.port, args.baud, args.seconds)


if __name__ == "__main__":
    main()

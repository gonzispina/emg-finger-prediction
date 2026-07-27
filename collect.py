"""
Entry point para correr una sesion de captura.

Uso:
    python collect.py                       # pregunta todo por consola
    python collect.py -s P01 -n 1 -p P01-A  # bypass de prompts
    python collect.py --mini -s P01_test -n 0 -p P01-A   # prueba ~2 min
"""
import argparse
from pathlib import Path

from collection.protocols import mini_protocol, protocol_1finger, protocol_1finger_noise, protocol_free_eval
from collection.runner import run_session


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{label}{suffix}: ").strip()
    return val or default


def main():
    p = argparse.ArgumentParser(description="Captura EMG/finger para TCN")
    p.add_argument("--subject", "-s", help="subject_id, ej P01")
    p.add_argument("--session", "-n", type=int, help="session_number (1-10)")
    p.add_argument("--placement", "-p", help="placement_id, ej P01-A")
    p.add_argument("--port", default="/dev/cu.usbserial-2110")
    p.add_argument("--baud", type=int, default=1000000,
                   help="baud rate (default 1000000 para Arduino; 2000000 para ESP32)")
    p.add_argument("--protocol", choices=["auto", "1finger", "1finger-noise", "free-eval"], default="auto",
                   help="protocolo: auto=segun session_number, 1finger=solo indice (~14 min), "
                        "1finger-noise=solo indice + distractores (~16 min), "
                        "free-eval=rest 15s + free_controlled 100s (~2 min, solo evaluacion)")
    p.add_argument("--output-dir", default="data/raw", type=Path)
    p.add_argument("--notes", default=None)
    p.add_argument("--mini", action="store_true",
                   help="protocolo mini (~2 min) para validar pipeline. "
                        "NO usar para entrenar.")
    args = p.parse_args()

    subject_id = args.subject or _prompt("subject_id (ej P01)")
    if not subject_id:
        raise SystemExit("subject_id es requerido")

    session_number = args.session
    if session_number is None:
        session_number = int(_prompt("session_number (1-10)", "1"))

    placement_id = args.placement or _prompt(
        "placement_id (ej P01-A)", f"{subject_id}-A"
    )
    notes = args.notes if args.notes is not None else _prompt("optional_notes (Enter para vacio)")

    if args.mini:
        blocks_override = mini_protocol()
        print("\n[MINI] Protocolo mini activado (~2 min). NO usar este CSV "
              "para entrenamiento.\n")
    elif args.protocol == "1finger":
        blocks_override = protocol_1finger()
        print("\n[1FINGER] Protocolo solo-indice (~14 min).\n")
    elif args.protocol == "1finger-noise":
        blocks_override = protocol_1finger_noise()
        print("\n[1FINGER-NOISE] Protocolo solo-indice con distractores (~16 min).\n")
    elif args.protocol == "free-eval":
        blocks_override = protocol_free_eval()
        print("\n[FREE-EVAL] Protocolo de evaluacion: rest 15s + free_controlled 100s. "
              "NO usar para entrenar.\n")
    else:
        blocks_override = None

    run_session(
        subject_id=subject_id,
        session_number=session_number,
        placement_id=placement_id,
        port=args.port,
        baud=args.baud,
        output_dir=args.output_dir,
        notes=notes,
        blocks_override=blocks_override,
    )


if __name__ == "__main__":
    main()
